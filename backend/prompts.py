from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core.llms import ChatMessage
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant which answers general question asked by a user."
)
SYSTEM_PROMPT_LANGCHAIN = """
You are a helpful assistant which answers general question asked by a user.
Current conversation:
{history}
Human: {input}
AI Assistant:
"""

SYSTEM_PROMPT_LANGCHAIN_2 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def ner_prompt_generator(question, model_type):
    if model_type == "langchain":
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"NER for the question in one or two words. Give only the word and don't start it with some pre words like Named Entity - question - {question}"
            ),
        ]
    elif model_type == "llamaindex":
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"NER for the question in one or two words. Give only the word and don't start it with some pre words like Named Entity - question - {question}",
            ),
        ]
    elif model_type == "openai":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"NER for the question in one or two words. Give only the word and don't start it with some pre words like Named Entity - question - {question}",
            },
        ]
    else:
        raise ValueError("Invalid model type")

RELATIONSHIPS_PROMPT = f"""
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
    
    Give 3 more relationships apart from above relationships. Make sure the response is in strict json only.
"""

from langchain_core.prompts import PromptTemplate


CYPHER_PROMPT = '''
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

QA_PROMPT = """You are an assistant for question-answering tasks. 
        Use the following Cypher query results to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise. If topic information is not available, focus on the paper titles.
        
        Question: {question}
        Query Results: {context} 
        
        Answer:<Explain answer in a sentence>"""

DATA_PREFIX_PROMPT = """
Your are an expert in answering questions based on the dataframe provided to you.
Given below are the unique categorical values present in our data with column names:
<<categorical_information>>
"""

DATA_SUFFIX_PROMPT = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""