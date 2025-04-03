from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_openai import ChatOpenAI
import os
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from dotenv import load_dotenv
from config import *

load_dotenv()

class AgenticChunker:
    def __init__(self, openai_api_key=None):
        self.chunks = {}
        self.id_truncate_limit = 5

        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = False

        if openai_api_key is None:
            openai_api_key = open_ai_key

        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=open_ai_key, temperature=0)

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition):
        if self.print_logging:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return
        
        # if the proposition is not new, we find if its similar to any other chunk or not
        chunk_id = self._find_relevant_chunk(proposition)
        
        if chunk_id == None:
            # If a chunk wasn't found (That means there is no chunk similar), then create a new one
            self._create_new_chunk(proposition)
        else:
            # If a relevant chunk is found, then just append it with new proposition
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        

    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        }).content

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        }).content

        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title


    def _create_new_chunk(self, proposition):
        # Create a new ID for your chunk
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]

        # extract summary of the proposition
        new_chunk_summary = self._get_new_chunk_summary(proposition)

        # Give a title to the propisition
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        # Create the chunk with the above attributes
        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""
        # We will iterate over all the chunks and create an outline of them
        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        # Get the outline of all the chunks created till now
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk of their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content
        
        
        if chunk_found == "No chunks":
            return None

        # Pydantic data class
        class ChunkID(BaseModel):
            """Extracting the chunk id"""
            chunk_id: Optional[str]
            
        # Extraction to catch-all LLM responses. This is a bandaid
        extraction_chain = self.llm.with_structured_output(ChunkID)
        extraction_found = extraction_chain.invoke(chunk_found)
        if extraction_found:
            chunk_found = extraction_found.chunk_id

        # If you got a response that isn't the chunk id limit, chances are it's a bad response or it found nothing
        # So return nothing
        if len(chunk_found) != self.id_truncate_limit:
            return None

        return chunk_found
    
    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print (f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        pass
        # print ("Chunk Outline\n")
        # print(self.get_chunk_outline())