from abc import ABC, abstractmethod

from fastapi import HTTPException

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI as OAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.llms.openai import OpenAI as OpenAILlama

from config import *
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_LANGCHAIN,SYSTEM_PROMPT_LANGCHAIN_2, ner_prompt_generator, RELATIONSHIPS_PROMPT, CYPHER_PROMPT, QA_PROMPT, DATA_PREFIX_PROMPT, DATA_SUFFIX_PROMPT
from answer_generator import AnswerGenerator
import os
import json 
import pandas as pd
from langchain_openai import OpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain_core.globals import set_verbose, set_debug
from answer_generator import DataAnswerGenerator

set_verbose(False)
set_debug(False)

import redis

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_langchain = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
client_llamaindex = OpenAILlama(model=MODEL, api_key=OPENAI_API_KEY)
client_langchain_with_prompt_template = OAI()

cache = {}


class OpenAIAnswerGenerator(AnswerGenerator):
    def generate_answer(self, question: str, session_id: str):
        try:
            if self.session_id != session_id:
                self.session_id = session_id
                try:
                    session_name = client_openai.chat.completions.create(model=MODEL, messages=ner_prompt_generator(question, "openai"))
                    self.set_index(session_name.choices[0].message.content,session_id,"all_sessions_data_openai")
                except Exception as e:
                    print(str(e))
                    
            self.load_from_redis()
            self.message_history.append({"role": "user", "content": question})
            
            answer = client_openai.chat.completions.create(model=MODEL, messages=self.message_history)
            
            self.message_history.append({"role": "assistant", "content": answer.choices[0].message.content})
            self.save_to_redis()

            return answer.choices[0].message.content
        
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with OpenAI"
            )


class LangChainAnswerGenerator(AnswerGenerator):
    def __init__(self):
        super().__init__()

    def get_redis_history(self, session_id: str):
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

    def generate_answer(self, question: str, session_id: str):
        try:
            print(question, session_id, self.session_id)
            if self.session_id != session_id:
                self.session_id = session_id
                session_name = client_langchain.invoke(ner_prompt_generator(question, "langchain"))
                self.set_index(session_name.content, session_id, "all_sessions_data_langchain")
            print("till here")
            chain = SYSTEM_PROMPT_LANGCHAIN_2 | client_langchain
            print("till here 2")
            conversation = RunnableWithMessageHistory(chain, self.get_redis_history, input_messages_key="input", history_messages_key="history")
            answer = conversation.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
            return answer.content

        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail="Error generating answer with LangChain")


class LlamaIndexAnswerGenerator(AnswerGenerator):
    def __init__(self):
        super().__init__()
    
    def generate_answer(self, question: str, session_id: str):
        try:
            if self.session_id != session_id:
                self.session_id = session_id
                session_name = client_llamaindex.chat(ner_prompt_generator(question, "llamaindex"))
                self.set_index(str(session_name).replace("assistant: ", ""),session_id,"all_sessions_data_llama")
            
            chat_store = RedisChatStore(redis_url=REDIS_URL)
            chat_memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=session_id)
            conversation = SimpleChatEngine.from_defaults(memory=chat_memory, llm=client_llamaindex)
            answer = conversation.chat(question)
            return answer.response
        
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with Llama Index"
            )

class KGDataAnswerGenerator(DataAnswerGenerator):
    def __init__(self):
        super().__init__()

    def load_data(self):
        # read the csv file
        data = pd.read_csv(self.data_path, encoding='utf-8')

        # make sure that there are no spaces in column name as it can cause error in building relationships
        data.columns = data.columns.str.replace(" ", "_", regex=False)

        # find the column which has all elements has unique
        unique_columns = data.nunique() == len(data)

        # this column we will mark as primary key and rename it as well
        if not any(data.nunique() == len(data)):
            data.insert(0, "Primary_Key", range(1, len(data) + 1))
        
        elif unique_columns.any():
            primary_key_column = data.columns[unique_columns.argmax()]
            data.rename(columns={primary_key_column: "Primary_Key"}, inplace=True)
        return data

    def get_relationships(self, relationship_prompt):
        try:
            results = client_langchain_with_prompt_template.invoke(relationship_prompt)
            final_results = json.loads(results)
            return final_results
        except Exception as e:
            print(results)
            print(str(e))
            return "failure"
    
    def create_relationships_df(self, final_results, data):
        """
            This function is used to create a dataframe from relationships and original data frame
            where for each row we will generate all the relationships.
        """
        output_rows = [] # we use this to create the processed rows and later convert to df
        master_columns = data.columns # get all the columns from the data
        try:
            # iterate through each row in data
            for index, row in data.iterrows():
                # extract each relationship for each row
                for item in final_results[1:]:
                    output_rows.append({
                    "relationship": item["relationship"],
                    "entity_type": item["entity_type"],
                    "entity_value": row[item["entity_value"]],
                    "description": item["description"],
                    item["unique_identifier"]: str(row[item["unique_identifier"]]),
                    **{column: str(row[column]) for column in master_columns if column not in [item["unique_identifier"], "index"]}  # Add remaining columns in the data as well
                })
        
        except Exception as e:
            print(str(e))

        output_df = pd.DataFrame(output_rows)
        return output_df, "Primary_Key"
    
    def insert_to_db(self, output_df, unique_identifier_name):
        i = 1
        for index, row in output_df.iterrows():
            # we will iterate through our relationships df one by one
            i += 1
            # excluding columns that are not required as a node in graph db
            excluded_columns = {"unique_identifier_name", "entity_type", "entity_value", "relationship"}
            
            # query to create nodes
            on_create_set = ",\n        ".join(
                [
                    f'analysis.{self.sanitize(key)} = "{self.sanitize(value)}"'
                    for key, value in row.items()
                    if key not in excluded_columns
                ]
            )
            # this query will create nodes, relationships and keys
            insert_query = f'''
                MERGE (analysis:Entity {{id: "{self.sanitize(row[unique_identifier_name])}"}})
                ON CREATE SET 
                    analysis.{unique_identifier_name} = "{self.sanitize(row[unique_identifier_name])}",
                    {on_create_set}
                MERGE (entity:{self.sanitize(row['entity_type'])} {{value: "{self.sanitize(row['entity_value'])}"}})
                MERGE (analysis)-[:{self.sanitize(row['relationship'])}]->(entity)
            '''

            # this will start the run of the query
            self.graph.query(insert_query)

            # store the query to put in the context later
            self.insert_query = insert_query

    def create_context(self, final_results):
        """
            We will use this function to get all our entities and relationships at one place.
            This will be used to create a final prompt for the user query
        """
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

    def get_categorical_columns_data(self, df):
        categorical_columns = {}

        for column in df.columns:
            if column == "relationship" or column == "entity_value" or column == "entity_type" or column == "description":
                continue
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                try:
                    pd.to_numeric(df[column])
                    continue
                except ValueError:
                    categorical_columns[column] = df[column].unique().tolist()

        return json.dumps(categorical_columns)
    
    def get_results(self, entity_types, relation_types, entity_relationship_match, graph, unique_identifier, data, additional_context, query, categorical_information):
        user_input = query
        input_variables = ["schema", "question"]
        replacements = {
            "<<additional_context>>": json.dumps(additional_context),
            "<<entity_types>>": json.dumps(entity_types),
            "<<relation_types>>": json.dumps(relation_types),
            "<<entity_relationship_match>>": json.dumps(entity_relationship_match),
            "<<unique_identifier>>": unique_identifier,
            "<<data>>": data,
            "<<categorical_information>>": categorical_information

        }
        parameters = {
            "input_variables": input_variables,
            "replacements": replacements
        }

        # replace the values in cypher prompt
        cypher_prompt = self.get_prompt(CYPHER_PROMPT, **parameters)
        
        input_variables=["question", "context"]
        parameters = {
            "input_variables" : input_variables,
            "replacements" : {}
        }

        # create the qa prompt
        qa_prompt = self.get_prompt(QA_PROMPT, **parameters)
        
        # start the agent. It will understand user query and generate the cypher
        # next it will execute the cypher
        # finally it will provide answer to the user
        chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=1), qa_llm=ChatOpenAI(temperature=1) , graph=graph, verbose=True, allow_dangerous_requests=True, cypher_prompt=cypher_prompt, qa_prompt=qa_prompt
        )
        
        results = chain.invoke({"query": user_input})
        return results

    def dump_to_cache(self, entity_types, relation_types, entity_relationship_match, unique_identifier_name, output_df):
        self.cache["entity_types"] = entity_types
        self.cache["relation_types"] = relation_types
        self.cache["entity_relationship_match"] = entity_relationship_match
        self.cache["unique_identifier_name"] = unique_identifier_name
        self.cache["output_df"] = output_df.head(2).to_string()
        self.cache["additional_context"] = self.insert_query
        self.cache["categorical_information"] = self.get_categorical_columns_data(output_df)

    def generate_multiple_relationships(self, data, num_iter=10, retry=5):
        # Creating an empty dictionary. This will tell about the relationships between columns. It will also tell the respective column name (entity type) and the value.
        # We will provide the primary key column name in unique identifier
        final_results = [{ "relationship": "", "entity_type": "", "entity_value": "", "unique_identifier": "Primary_Key"}]
        
        # Next we will initialize the prompt for getting these relationships
        # We will give first two rows of our data for the context
        # we will also provide our empty json so that it keeps on generating new relationships and adding it
        replacements = {
            "<<data>>": data.head(2).to_string(),
            "<<final_results>>": json.dumps(final_results)
        }
        parameters = {
            "input_variables" : [],
            "replacements" : replacements
        }
        print("getting relationship prompt")
        # created a reusable function that will do the replacements in prompt for us
        relationship_prompt = self.get_prompt(template=RELATIONSHIPS_PROMPT, **parameters)
        print("received relationship prompt")
        print("************************************************************************")
        
        # Now we will run the loop 5 times
        # Each time 3 relationships are generated
        # Relationships are appended to our final_results dictionary
        loop = 0
        while True:
            if loop == num_iter:
                break
            temp = self.get_relationships(relationship_prompt)
            if isinstance(temp, str):
                # if llm gives error 5 times, then we will continue with already generated relationships
                if retry == 0:
                    break
                else:
                    retry -= 1
                    continue
            final_results.extend(temp)
            print(f"generated {loop+1} relationships")
            loop+=1
        return final_results

    def generate_answer_from_data(self, query, session_id, regenerate, data_path, database_name):
        # initialize the variables based on the request parameters received
        self.session_id = session_id
        self.data_path = data_path
        self.database = database_name
        self.query = query
        print(self.session_id,
        self.data_path,
        self.database,
        self.query,
        )
        # if new session then create the instance in Redis
        if self.session_id != session_id:
            self.session_id = session_id
            try:
                print("Creating new session")
                csv_name  = os.path.splitext(os.path.basename(data_path))[0]
                session_name = f"Chat with {csv_name}"
                self.set_index(session_name, session_id," all_sessions_data_langchain")
            except Exception as e:
                print(str(e))

        try:
            # if older session, retrieve information from the redis
            if not regenerate:
                print("loading from redis")            
                self.load_from_redis()
                print("************************************************************************")
                print("loaded from redis")
                self.connect_to_db()
            # if not then start the new flow
            else:
                data = self.load_data()
                print("************************************************************************")
                print("loaded new data")
                final_results = self.generate_multiple_relationships(data, num_iter=3, retry=5)
                output_df, unique_identifier_name = self.create_relationships_df(final_results, data) 
                print("************************************************************************")
                print("generated exhaustive relationships")
                status = self.create_database()
                if status == "success":
                    print("************************************************************************")
                    print("databased created")
                else:
                    print("************************************************************************")
                    print("using older database")
                self.connect_to_db()
                print("************************************************************************")
                print("connected to db")  
                self.insert_to_db(output_df[:1000], unique_identifier_name)
                print("************************************************************************")
                print("relationships dumped to db")
                print("final_results", final_results)
                entity_types, relation_types, entity_relationship_match = self.create_context(final_results)
                print("************************************************************************")
                print("created additional context for prompt")
                print(entity_types, relation_types, entity_relationship_match )
                self.dump_to_cache(entity_types, relation_types, entity_relationship_match, unique_identifier_name, output_df)
                print("************************************************************************")
                print("initialized cache")
            
            self.cache["message_history"].append({"role": "user", "content": query}) 
            results = self.get_results(
                    entity_types=self.cache["entity_types"], 
                    relation_types=self.cache["relation_types"], 
                    entity_relationship_match=self.cache["entity_relationship_match"], 
                    graph=self.graph,
                    unique_identifier=self.cache["unique_identifier_name"], 
                    data=self.cache["output_df"],
                    additional_context=self.cache["additional_context"], 
                    query=self.query,
                    categorical_information = self.cache["categorical_information"]
                )
            self.cache["message_history"].append({"role": "assistant", "content": results["result"]})
            self.save_to_redis()
            return results["result"]
        
        except Exception as e:
            return str(e)
        

class LlamaIndexAnswerGenerator(AnswerGenerator):
    def __init__(self):
        super().__init__()
    
    def generate_answer(self, question: str, session_id: str):
        try:
            if self.session_id != session_id:
                self.session_id = session_id
                session_name = client_llamaindex.chat(ner_prompt_generator(question, "llamaindex"))
                self.set_index(str(session_name).replace("assistant: ", ""),session_id,"all_sessions_data_llama")
            
            chat_store = RedisChatStore(redis_url=REDIS_URL)
            chat_memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=session_id)
            conversation = SimpleChatEngine.from_defaults(memory=chat_memory, llm=client_llamaindex)
            answer = conversation.chat(question)
            return answer.response
        
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with Llama Index"
            )

class PandasDataAnswerGenerator(DataAnswerGenerator):
    def __init__(self):
        super().__init__()

    def load_data(self):
        data = pd.read_csv(self.data_path, encoding='utf-8')
        return data
    
    def get_categorical_columns_data(self, df):
        """
            This function is used to extract information about categorical columns present in the data. It returns the categorical column name with
            all the unique categories present inside it.
        """
        categorical_columns = {}

        for column in df.columns:
            # check if a column is an object or category
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                try:
                    # try to convert the column to numeric. If converted then its not a category and hence skip this loop
                    pd.to_numeric(df[column])
                    continue
                except ValueError:
                    # if not converted, find the unique values of  this column
                    categorical_columns[column] = df[column].unique().tolist()

        return json.dumps(categorical_columns)

    def get_results(self, data_prefix, categorical_columns, data, query):
        replacements = {
            "<<categorical_information>>": categorical_columns
        }
        parameters = {
            "input_variables" : [],
            "replacements" : replacements
        }
        data_prefix = self.get_prompt(template=data_prefix, **parameters)
        print(data_prefix)
        agent = create_pandas_dataframe_agent(
                llm=client_langchain,
                df=data,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                prefix=data_prefix,
                include_df_in_prompt=True,
                number_of_head_rows=5,
                allow_dangerous_code=True,
            )
        results = agent.invoke(query)
        return results

    def dump_to_cache(self, data_prefix, data, categorical_columns):
        self.cache["data_prefix"] = data_prefix
        self.cache["data"] = data.to_json()
        self.cache["categorical_columns"] = categorical_columns

    
    def generate_answer_from_data(self, query, session_id, regenerate, data_path, database_name):
        self.data_path = data_path
        self.database = database_name
        self.query = query
        if self.session_id != session_id:
            self.session_id = session_id
            try:
                print("Creating new session")
                csv_name  = os.path.splitext(os.path.basename(data_path))[0]
                session_name = f"Chat with {csv_name}"
                self.set_index(session_name, session_id,"all_sessions_data_langchain")
            except Exception as e:
                print(str(e))

        try:
            if not regenerate:        
                if isinstance(self.data, str):
                    print("Found empty values")
                    print("loading from redis")
                    self.load_from_redis()
                    self.data  = pd.read_json(self.cache["data"])
                print("************************************************************************")
                print("loaded from redis")
            else:
                self.data = self.load_data()
                print("************************************************************************")
                print("loaded new data")

                data_prefix = DATA_PREFIX_PROMPT
                categorical_columns = self.get_categorical_columns_data(self.data)
                self.dump_to_cache(data_prefix, self.data, categorical_columns)
                print("************************************************************************")
                print("initialized cache")
            
            self.cache["message_history"].append({"role": "user", "content": query})            
            results = self.get_results(
                    data_prefix = self.cache["data_prefix"],
                    categorical_columns=self.cache["categorical_columns"],
                    data = self.data,
                    query=query
                )
            self.cache["message_history"].append({"role": "assistant", "content": results["output"]})
            self.save_to_redis()
            return {"status": "success", "results": results["output"]}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}