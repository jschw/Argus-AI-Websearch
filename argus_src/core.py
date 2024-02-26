from argus_src.llm_inference import LlmConfiguration, LlmInferenceEngine, LlmInfTypes
from argus_src.datatypes_llm import Conversation, Message, MsgType
from argus_src.utils import crawl_website

from googlesearch import search


import re

import pandas as pd  # requirements: pip install pyarrow
from pandas import DataFrame

from urllib.parse import urlparse, urljoin

import nest_asyncio

import os
import sys

import json

import openai

import langchain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Document, MarkdownHeaderTextSplitter

from langchain_community.vectorstores import Qdrant  # requirements: pip install qdrant-client
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings # requirements: pip install sentence-transformers

from langchain_community.document_loaders import AsyncChromiumLoader # requirements: pip install playwright / playwright install
from langchain_community.document_transformers import Html2TextTransformer  # requirements: pip install html2text

# Unicode-Symboltabelle:
# https://www.gaijin.at/en/infos/unicode-character-table-dingbats#U2700


# ==== Settings ====

# os.environ['HF_TOKEN'] = "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"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

enableVerboseOutput = True
enable_context = True

config_file = "config.json"

# stage 1: Generating search queries
# stage 2: Aggregating data with websearch + crawling
# stage 3: Build vecstore + get relevant context with similarity search
# stage 4: Summarize context / perform task described in original prompt

stage_1_depth = 3  # Generate 3 search queries
stage_2_depth = 5  # Load 5 top rated websites
stage_3_chunksize = 700  # Size of each content chunk is 500 Token
stage_3_depth = 3  # Add 3 chunks of each vecstore search result to content.
                   # Example: stage_3_depth * stage_3_depth = chunk count in context
                   # Example: 3 * 3 = 9 chunks * 500 token = 4500 token context

enableDryRun = False

class ArgusWebsearch():

    def __init__(self):

        # ==== Load configuration file ====

        self.llm_config = LlmConfiguration(config_file)

        # ==== Init inference engine ====

        self.llm = LlmInferenceEngine(LlmInfTypes.OPENAI, api_key = self.llm_config.get_api_key())

        self.tokens_used_total = 0

        self.vectorstore = None

        self.rag_context = None


    def run_stage1(self, prompt:str) -> list:
        # Create conversation object for stage 1
        self.conversation_stage1 = Conversation()

        # System role information
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = f"This is an application that formulates {stage_1_depth} websearch input queries based on the given input.\n"))
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = f"All {stage_1_depth} queries are different and each one focuses on another aspect given in the user input.\n"))
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = "All queries should not be longer than absolutely necessary.\n"))
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = "The query with the highest overall relevancy for the search topic should be the first one in the output.\n"))
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = "The application respects the following rules of formulating a perfect websearch query which are separated by ## :\n"))

        tmp_string = "## The phrase should be clear and concise, avoiding ambiguity. Use straightforward language that clearly conveys what you are looking for.\n"
        tmp_string += "## Be as specific as possible about the topic or information you're seeking. General queries often yield broad results, while specificity narrows down the focus.\n"
        tmp_string += "## Ensure that the keywords or terms you include are directly related to your query. Irrelevant terms can lead to off-topic results.\n"
        tmp_string += "## Incorporate relevant keywords or key phrases that are likely to appear in the content you are seeking. Think about the terms someone might use when discussing your topic.\n"
        tmp_string += "## Unless searching within a specialized field, avoid using technical jargon or industry-specific terms that may not be widely understood.\n"
        tmp_string += "## If seeking an answer to a specific question, consider structuring your search phrase as a question. This can help in obtaining direct and concise answers.\n"
        tmp_string += "## Exclusion of Unwanted Information. Use modifiers like \"not,\" \"-\" (minus), or \"exclude\" to remove irrelevant information. For example, if you're looking for information about apples but not the fruit, you might search for \"apple -fruit.\"\n"
        tmp_string += "## Enclosing phrases in quotation marks can help search engines find exact matches for that phrase, which is especially useful when looking for specific terms or titles.\n"
        tmp_string += "## If applicable, include different variations or synonyms of your search terms to capture a broader range of relevant content. This is particularly useful when terms may have multiple meanings.\n"
        tmp_string += "## Consider the context of your search. If you're looking for recent information, include a timeframe or specific year. Likewise, if location is relevant, include geographic modifiers to narrow down results.\n"
        tmp_string += "## The search query is as short as possible and do not contain additional words.\n"
        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = tmp_string))

        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = "The application outputs only the search queries formatted in JSON format and nothing else and without special characters.\n"))

        tmp_string = "The application outputs the search queries in the following JSON:\n"
        tmp_string += """[
            {
                "search_query": "<output_query_1>"
            },
            {
                "search_query": "<all following queries>"
            }
        ]"""

        self.conversation_stage1.add_message(Message(type=MsgType.SYSTEM, msg = tmp_string))


        # ==== LLM inference ====

        text_out = ""

        # Append context if activated
        # prompt = prompt_messages_stage1
        # prompt.append({"role": "user", "content": input_prompt})

        self.conversation_stage1.add_message(Message(type=MsgType.USER, msg = prompt), finish_sequence=True)

        # Inference stage 1
        if enableDryRun:
            # Dryrun
            text_out = """[
                {
                    "search_query": "world's tallest building 2023"
                },
                {
                    "search_query": "tallest building completion date 2023"
                },
                {
                    "search_query": "skyscraper height record 2023"
                }
            ]"""

            text_out = """[
                {
                    "search_query": "CO2 footprint of battery electric vehicles compared to modern Diesel engine vehicles"
                },
                {
                    "search_query": "Comparison of carbon emissions between battery electric vehicles and modern Diesel engine cars"
                },
                {
                    "search_query": "Environmental impact of battery electric cars versus modern Diesel engine automobiles"
                }
            ]"""
            tokens_used_stage1 = 500
        else:
            text_out, tokens_used_stage1 = self.llm.run_inference(self.conversation_stage1.create_prompt_dict(), self.llm_config.get_config())

        # Parse JSON output
        tmp_json_result = json.loads(text_out)

        generated_search_queries = []
        for doc in tmp_json_result:
            generated_search_queries.append(doc['search_query'])

        # Debug output of parsed JSON
        if enableVerboseOutput:
            print("--> Generated search queries: ")
            i = 1
            for query in generated_search_queries:
                print(f"{i}: {query}")
                i += 1

            print(f"\n--> Tokens used for stage 1 inference: {tokens_used_stage1}")

        print(f"\n==> Finished stage 1\n")

        return generated_search_queries, self.llm.token_used_last


    def run_stage2(self, queries:list) -> list :
        # Perform websearch with the generated queries -> google
        result_urls = []

        for query in queries:

            urls_tmp = search(query, tld="co.in", num=stage_2_depth, stop=stage_2_depth)
            urls_tmp = list(urls_tmp)

            for url in urls_tmp:
                result_urls.append(url)

        if enableVerboseOutput:
            print("--> Top rated result URLs:\n")
            url_num = 1
            for url in result_urls:
                print(f"{url_num}: {url}")
                url_num += 1
            print("")

        # Scrape websites + Convert website content to markdown
        content_list = []

        for url in result_urls:
            tmp_website_content = crawl_website(url=url)
            if tmp_website_content != None:
                content_list.append([url, tmp_website_content])

        print(f"\n==> Finished stage 2\n")

        return content_list

    def run_stage3(self, content:list, queries:list) -> DataFrame:
        # Create Embeddings and pass to vectorstore
        new_doc = []

        i = 0
        for doc in content:
            new_doc.append(Document(page_content=doc[1], metadata={'url': doc[0]}, type="Document"))

            i += 1

        text_splitter_text = RecursiveCharacterTextSplitter(chunk_size=stage_3_chunksize, chunk_overlap=100)
        source_docs_urls = text_splitter_text.split_documents(new_doc)

        # print(source_docs_urls[1])

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create on-memory Qdrant instance from website content
        self.vectorstore = Qdrant.from_documents(
            source_docs_urls,
            embeddings, 
            location = ":memory:",
            collection_name="websearch_results",
        )

        # Get RAG input documents for each search query
        rag_aggregated_results = []

        query_num = 1
        for query in queries:
            tmp_results = self.vectorstore.similarity_search(query, k=stage_3_depth)
            for res in tmp_results:
                rag_aggregated_results.append([res.metadata['url'], res.page_content])

            if enableVerboseOutput:
                print(f"--> Performing search for query {query_num}: '{query_num}'")

            query_num += 1

        # Convert list to pandas dataframe
        df_rag_aggregated_results = pd.DataFrame(rag_aggregated_results, columns =['URL', 'Content'])

        # Delete duplicate chunks
        df_rag_aggregated_results.drop_duplicates(subset=['Content'], keep='first', inplace=True, ignore_index=True)

        if enableVerboseOutput:
            print(df_rag_aggregated_results)
            df_rag_aggregated_results.to_csv('RAG_output.csv')

        print(f"\n==> Finished stage 3, {df_rag_aggregated_results.shape[0]} document chunks in context\n")

        return df_rag_aggregated_results


    def run_stage4(self, context:DataFrame, prompt:str) -> str:
        # ======== Stage 4 - Performing Task of original prompt ==========

        self.conversation_stage4 = Conversation()

        # System role information
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "This is a helpful assistant that compiles information, answers questions or generally carries out what is requested in the user input.\n"))
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "The assistant uses the information provided in the following context to perform the task requested in the user input.\n"))
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "The context contains several pieces of information, which are separated by ## .\n"))
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "Each context block contains its number, the source URL and the source content. The source content is the information which should be used to perform the task.\n"))

        tmp_str = "Context:"

        for index, doc in context.iterrows():
            tmp_str += "\n\n##\n"
            tmp_str += f"Source No: {index+1}\n"
            tmp_str += f"Source URL: {doc['URL']}\n"

            tmp_str += f"Source Content: {doc['Content']}\n"

        # Add information context
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg =  tmp_str))

        # Add formatting instructions
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "Provide step by step reasoning and mark all the places in the answer with the numbers of the information used.\n"))
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "Do not refer to the source directly or output its URL, only output the number of the source in brackets like (2).\n"))
        self.conversation_stage4.add_message(Message(type=MsgType.SYSTEM, msg = "Provide a continuous text with linebreaks in a scientific tone.\n"))

        # Add user question                 
        self.conversation_stage4.add_message(Message(type=MsgType.USER, msg = prompt))

        # Inference stage 4
        text_out = ""
        text_out, tokens_used_stage4 = self.llm.run_inference(self.conversation_stage4.create_prompt_dict(), self.llm_config.get_config())

        # Add answer to conversation
        self.conversation_stage4.add_message(Message(type=MsgType.ASSISTANT, msg = text_out), finish_sequence=True)

        print(f"\n==> Finished stage 4, going to output results.\n")

        return text_out, self.llm.token_used_last


    def run_full_research(self, input_prompt:str) -> str:

        generated_queries, tokens_stage1 = self.run_stage1(prompt=input_prompt)

        website_content = self.run_stage2(queries=generated_queries)

        self.rag_context = self.run_stage3(content=website_content, queries=generated_queries)

        llm_output, tokens_stage4 = self.run_stage4(context=self.rag_context, prompt=input_prompt)

        self.tokens_used_total = self.llm.token_used_total

        return llm_output, self.tokens_used_total
    

    def append_and_run(self, input_prompt:str) -> str:
        # Add new input to conversation
        self.conversation_stage4.add_message(Message(type=MsgType.USER, msg = input_prompt))

        # Run inference
        llm_output, tokens_actual = self.llm.run_inference(self.conversation_stage4.create_prompt_dict(), self.llm_config.get_config())
        self.conversation_stage4.add_message(Message(type=MsgType.ASSISTANT, msg = llm_output))

        self.tokens_used_total = self.llm.token_used_total

        return llm_output, self.llm.token_used_last

