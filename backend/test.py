import os
import json 
import pandas as pd
from langchain_openai import OpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI

from langchain_core.globals import set_verbose, set_debug

# Disable verbose logging
set_verbose(False)

# Disable debug logging
set_debug(False)



llm = OpenAI()

def load_data(data_path):
    """
    Load data from the given path, replace spaces in column names with underscores, 
    and add a primary key column if none exists.

    :param data_path: Path to the CSV file.
    :return: DataFrame with modified column names and a primary key column if needed.
    """
    data = pd.read_csv(data_path, encoding='utf-8')

    # Replace spaces in column names with underscores
    data.columns = data.columns.str.replace(" ", "_", regex=False)

    unique_columns = data.nunique() == len(data)

    # Check if there is a primary key (assuming primary key means unique values in one column)
    if not any(data.nunique() == len(data)):
        # Add a primary key column with incremental numbers
        data.insert(0, "Primary_Key", range(1, len(data) + 1))
    elif unique_columns.any():
        # Rename the first column with all unique values to "Primary_Key"
        primary_key_column = data.columns[unique_columns.argmax()]
        data.rename(columns={primary_key_column: "Primary_Key"}, inplace=True)

    return data

data = ""

relationship_prompt = f"""
    You are a helpful agent designed to create a json output that will be helpful for finding relationships and can be used for creating graph db later.
    
    given below is the top 2 rows of a dataframe from where you need to extract the json
    <<data>>
    
    Based on the above data you need to find the cypher (neo4j) relationships, entity_value, entity_type and description. It should be in following json format:
        [
            {{ "relationship": "", "entity_type": "", "entity_value":"", "description": "", "unique_identifier":"Primary_Key" }}
        ]

    For example, if titanic data is uploaded below can be an example json:
        [
            {{ "relationship": "has_gender", "entity_type": "gender", "entity_value": "Sex", "description": "Defines the gender of a passenger.", "unique_identifier": "Primary_Key" }},
            {{ "relationship": "has_age", "entity_type": "age", "entity_value": "Age", "description": "Defines the age of a passenger.", "unique_identifier": "Primary_Key" }},
            {{ "relationship": "has_family", "entity_type": "family", "entity_value": "SibSp, Parch", "description": "Defines the family relationships of a passenger.", "unique_identifier": "Primary_Key" }},
        ]
    entity_value represents the column name from which the value should be extracted.
    unique_identifier will always have value Primary_Key
    make sure entity_type has no spaces.

    I have already got following relationships. Don't include these combinations. Find more.
    <<final_results>>
    
    Extract as many relationships as possible, based on what a user can ask and query based on this data. Give top 3 relationships. Make sure the response is in strict json only.
"""

def get_relationships(relationship_prompt, data):
    results = llm.invoke(relationship_prompt)
    final_results = json.loads(results)
    return(final_results)

def create_relationships_df(final_results, data):
    output_rows = []
    unique_identifier_name = ""
    master_columns = data.columns
    print(master_columns)
    print(final_results)
    try:
        for index, row in data.iterrows():
            for item in final_results[1:]:
                values = [row[item["entity_value"]]]   
                output_rows.append({
                "relationship": item["relationship"],
                "entity_type": item["entity_type"],
                "entity_value": values[0],
                "description": item["description"],
                item["unique_identifier"]: str(row[item["unique_identifier"]]),
                **{column: str(row[column]) for column in master_columns if column not in [item["unique_identifier"], "index"]}  # Add all other columns dynamically
            })
                unique_identifier_name = item["unique_identifier"]
    
    except Exception as e:
        print(str(e))

    # Create a DataFrame from the output
    output_df = pd.DataFrame(output_rows)
    return output_df, unique_identifier_name

def connect_to_db(db_name):
    url = "bolt://localhost:7687"
    username ="neo4j"
    password = "12345678"
    graph = Neo4jGraph(
        url=url, 
        username=username, 
        password=password, 
        database=db_name
    )
    return graph

def insert_to_db(graph, output_df, unique_identifier_name):
    print(unique_identifier_name)
    def sanitize(text):
        # Sanitize text to remove special characters
        return str(text).replace("'", "").replace('"', "").replace("{", "").replace("}", "")

    # Loop through each row in the Titanic DataFrame and add them to the database
    i = 1
    for index, row in output_df.iterrows():
        i += 1
        excluded_columns = {"unique_identifier_name", "entity_type", "entity_value", "relationship"}

        # Dynamically construct ON CREATE SET clause
        on_create_set = ",\n        ".join(
            [
                f'analysis.{sanitize(key)} = "{sanitize(value)}"'
                for key, value in row.items()
                if key not in excluded_columns
            ]
        )

        # Complete query
        query = f'''
            MERGE (analysis:Entity {{id: "{sanitize(row[unique_identifier_name])}"}})
            ON CREATE SET 
                analysis.{unique_identifier_name} = "{sanitize(row[unique_identifier_name])}",
                {on_create_set}
            MERGE (entity:{sanitize(row['entity_type'])} {{value: "{sanitize(row['entity_value'])}"}})
            MERGE (analysis)-[:{sanitize(row['relationship'])}]->(entity)
        '''
        # Execute the query in Neo4j
        graph.query(query)
    return graph, query

from neo4j import GraphDatabase

def create_database(uri, username, password, database_name):
    """
    Create a new database in Neo4j.
get_categorical_columns_data
    :param uri: Bolt URI (e.g., "bolt://localhost:7687")
    :param username: Neo4j admin username
    :param password: Neo4j admin password
    :param database_name: Name of the database to be created
    :return: None
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        with driver.session() as session:
            # Check if the database already exists
            result = session.run(f"SHOW DATABASES WHERE name='{database_name}'")
            if result.single():
                print(f"Database '{database_name}' already exists.")
                return
            
            # Create the new database
            session.run(f"CREATE DATABASE {database_name}")
            print(f"Database '{database_name}' has been created successfully.")
            return "success"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "failure"
    finally:
        driver.close()


def create_context(final_results):
    print(final_results)
    entity_types = {}
    relation_types = {}
    entity_relationship_match = {}
    for item in final_results:
        description = item.get("description", "")
        if description == "":
            continue
        entity_types[item["entity_type"]] = description
        relation_types[item["relationship"]] = description
        entity_relationship_match[item["entity_type"]] = item["relationship"]
    
    return entity_types, relation_types, entity_relationship_match

def get_categorical_columns_data(df):
    """
    Takes a DataFrame and outputs a JSON where keys are column names with categorical values 
    and values are lists of unique values in those columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        str: JSON string with column names and their unique values.
    """
    categorical_columns = {}

    for column in df.columns:
        if column == "relationship" or column == "entity_value" or column == "entity_type" or column == "description":
            continue
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            # Check if the column truly contains categorical data
            try:
                # Try converting to numeric; if successful, skip the column
                pd.to_numeric(df[column])
                continue
            except ValueError:
                # If conversion fails, treat it as categorical
                categorical_columns[column] = df[column].unique().tolist()

    

    return json.dumps(categorical_columns)

def get_results(entity_types, relation_types, entity_relationship_match, graph, unique_identifier, data, additional_context, query, categorical_information):
    from langchain_core.prompts import PromptTemplate
    user_input = query
    cypher_prompt = PromptTemplate(
        template='''
        You are an expert at generating Cypher queries for Neo4j.
        Use the following schema to generate a Cypher query that answers the given question.
        Make the query flexible by using case-insensitive matching and partial string matching where appropriate.

        To create the db I used below query:
        <<additional_context>>
        
        The neo4j graph database has following entity types:
        <<entity_types>>
        
        Each link has one of the following relationships. Use only below relationships. If you are using a relationship not present, then return relationship not found.
        <<relation_types>>

        The relationship match details is given below.
        <<entity_relationship_match>>

        The database has following property keys:
        <<unique_identifier>>

        Given below is the top two rows of the data:
        <<data>>

        Given below are the unique categorical values present in our data with column names:
        <<categorical_information>>

        Schema:
        {schema}
        
        Question: {question}
        
        Cypher Query: <enter the cypher query here for sure>
    '''
        .replace("<<additional_context>>", json.dumps(additional_context).replace('{', '{{').replace('}', '}}'))
        .replace("<<entity_types>>", json.dumps(entity_types).replace('{', '{{').replace('}', '}}'))
        .replace("<<relation_types>>", json.dumps(relation_types).replace('{', '{{').replace('}', '}}'))
        .replace("<<entity_relationship_match>>", json.dumps(entity_relationship_match).replace('{', '{{').replace('}', '}}'))
        .replace("<<unique_identifier>>", unique_identifier)
        .replace("<<data>>", data)
        .replace("<<categorical_information>>", categorical_information.replace('{', '{{').replace('}', '}}')),
        input_variables=["schema", "question"]
    )

    qa_prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following Cypher query results to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise. If topic information is not available, focus on the paper titles.
        
        Question: {question} 
        Cypher Query: {query}
        Query Results: {context} 
        
        Answer:""",
        input_variables=["question", "query", "context"],
    )

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=1), qa_llm=ChatOpenAI(temperature=1) , graph=graph, verbose=True, allow_dangerous_requests=True, validate_cypher=True, return_intermediate_steps=True, cypher_prompt=cypher_prompt, return_direct=True, qa_prompt=qa_prompt
    )
    
    results = chain.invoke({"query": user_input})
    return results

if __name__ == '__main__':
    print("loading data")
    data = load_data("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    print("data loaded. creating relationships")
    final_results = get_relationships(relationship_prompt.replace("<<data>>", data.head(2).to_string()), data)
    print("relationships created. creating dataframe")
    output_df, unique_identifier_name = create_relationships_df(final_results, data)
    print("dataframe created. creating graph")
    create_database("bolt://localhost:7687", "neo4j", "12345678", "titanic")
    graph = connect_to_db("titanic")
    print("graph created. inserting to graph")
    graph, query = insert_to_db(graph, output_df[:1000], unique_identifier_name)
    print("successfully inserted. creating context")
    entity_types, relation_types, entity_relationship_match = create_context(final_results)
    print("context created. replying to your query")
    results = get_results(entity_types, relation_types, entity_relationship_match, graph, unique_identifier_name, data=output_df.head(2).to_string(), additional_context=query, query="Average Age of people who survived. Exclude nan. Give descriptive answer")
    print(results)




