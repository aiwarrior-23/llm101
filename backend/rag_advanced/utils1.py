from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import concurrent.futures
import tqdm
import pandas as pd
from ragas import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.run_config import RunConfig
import warnings
import random
import time
from langchain_chroma import Chroma
from pydantic import BaseModel, Field


from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy, Faithfulness

from langchain_core.output_parsers import JsonOutputParser

import os
from config import HF_TOKEN, open_ai_key
import warnings
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Ignore all the warnings given by different packages
warnings.filterwarnings('ignore')

# Retrieve the open ai and hf tokens from the configuration file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["OPENAI_API_KEY"] = open_ai_key

# Select the GPU as the device
model_kwargs = {'device': 'cuda'}

# Required for fast similarity computations
encode_kwargs = {'normalize_embeddings': True}

# Initialize Embeddings, GPT and Ollama Model
llm_gpt = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=open_ai_key)

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5,
    num_thread = 2,
    format = ''
)

model_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

class QueryCategory(BaseModel):
        query: dict = Field(description="New query enhanced")


parser_query = JsonOutputParser(pydantic_object=QueryCategory)

class QueryEnhancer:
    def __init__(self):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance and rewrite this factual question for better information retrieval from knowledge base: {query}\nGive information strictly in given json format {{'query': str}}. Don't deviate from the format. No extra keys"
        )
        self.parser_query = JsonOutputParser(pydantic_object=QueryCategory)
        self.chain = self.prompt | self.llm | self.parser_query


    def get_query(self, query):
        return self.chain.invoke({'query':query})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_response(query, vectordb):
    # convert vector db to retriever to perform similarity search
    retriever = vectordb.as_retriever()

    # pull the prompt from HF hub
    prompt = hub.pull("rlm/rag-prompt")

    # create a chain with prompt and llama 3.2 1B llm
    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

    # pass the retriever and chain and get the response along with the retrieve documents
    rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)
    
    result = rag_chain_with_source.invoke(query)
    
    return result["answer"], result["context"]

def process_llm_response(k, v, vectordb):
    """Function to fetch LLM response in parallel."""
    enhanced_query = QueryEnhancer()
    new_query = enhanced_query.get_query(query = v["question"])['query']
    answer, context = get_llm_response(new_query, vectordb)
    return k, {"question": v["question"], "answer": answer, "context": [s.page_content for s in context]}

def get_response(cleaned_data, vectordb):
    llm_response_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_llm_response, k, v, vectordb): k for k, v in cleaned_data.items()}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            k, result = future.result()
            llm_response_data[k] = result
        return llm_response_data

warnings.filterwarnings('ignore')
        
def save_evaluator_results(result, name):
    result.to_csv(f"evaluation_{name}.csv")

def evaluator(name, vectordb, cleaned_data):
    data_list = []
    
    # Get Response from the selected Vector DB
    llm_response_data = get_response(cleaned_data, vectordb)

    # Persist the response for later usage
    with open(f"llm_response_data_{name}.json", "w") as file:
        json.dump(llm_response_data, file, indent=4)

    # Create an evaluation dataset
    for k, v in llm_response_data.items():
        row = SingleTurnSample(
            user_input=v["question"],
            retrieved_contexts=v["context"],
            response=v["answer"],
            reference=cleaned_data[k]["answer"],
            reference_contexts = [cleaned_data[k]["answer"]]
        )
    
        data_list.append(row)
    dataset = EvaluationDataset(samples=data_list)
    

    # To make the evaluation faster, we will go with two metrics at a time
    result_1 = evaluate(dataset, llm=llm_gpt, embeddings=model_embeddings, run_config=RunConfig(max_workers=10, max_retries=20, timeout=180), 
                      metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()]).to_pandas()

    # Adding a sleep timer to avoid rate limit or token limit error
    time.sleep(30)
    
    result_2 = evaluate(dataset, llm=llm_gpt, embeddings=model_embeddings, run_config=RunConfig(max_workers=10, max_retries=20, timeout=180), 
                      metrics=[ResponseRelevancy(), Faithfulness()]).to_pandas()

    # Merge all the metrics
    result = pd.merge(result_1, result_2[['user_input', 'answer_relevancy', 'faithfulness']], on='user_input')

    # Persist the metrics result for the experiment
    save_evaluator_results(result, name)
    
    return result

def create_or_load_vector_db(chunked_documents, persist_directory, load=True):
    if load:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=model_embeddings,
        )

    else:
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=model_embeddings,
            persist_directory=persist_directory,
        )
    
    return vectordb