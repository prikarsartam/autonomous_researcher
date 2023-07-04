from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoTokenizer
import time

from nltk.tokenize import word_tokenize

import pdfkit # You'll need this for the PDF creation feature


import os
import openai
from dotenv import find_dotenv, load_dotenv
import requests
import json 
import streamlit as st
from serpapi import GoogleSearch

# Path to your .env file
dotenv_path = '.env'

# Load the environment variables from the .env file
load_dotenv(dotenv_path)

chunk_size = 3000
chunk_overlap = 50


SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
print('\n\n')
def count_tokens(string, tokenizer):
    tokens = tokenizer.tokenize(string)
    return len(tokens)

def search(query):
    params = {
    "engine": "google",
    "q": f"{query}",
    "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    print("search results: ", organic_results)
    return organic_results


# llm to choose the best articles

def find_best_article_urls(response_data, query):
    # Turn JSON into string
    response_str = json.dumps(response_data)

    # Create LLM to choose the best articles
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    template = """
    You are a world-class journalist and researcher. You are extremely good at finding the most relevant articles on a certain topic.
    {response_str}
    Above is the list of search results for the query "{query}".
    Please choose the best 20 articles from the list and return ONLY an array of the URLs. Do not include anything else.
    """

    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template)

    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=True)

    urls = article_picker_chain.predict(response_str=response_str, query=query)

    # Convert string to list
    url_list = json.loads(urls)
    print(url_list)

    return url_list


# get content from each article & create a vector database

def get_content_from_urls(urls):   
    # use unstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data

def summarise(data, query):
    # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2500, chunk_overlap=400, length_function=len)
    # text = text_splitter.split_documents(data) 

    text_splitter = TokenTextSplitter( chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text = text_splitter.split_text(str(data))   

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create an information thread about {query}
    Please follow all of the following rules:

    1/ Make sure the content is engaging, informative with reliable data
    2/ Make sure the content is very descriptive and prescriptive
    3/ The content should address the {query} topic very precisely
    4/ The content needs to be rigorous and direct
    5/ The content needs to be written in a way that is transparent and unambiguous
    6/ The content needs to give audience actionable advice & insights too

    SUMMARY:
    """

    prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries

def summarise_1(text, query):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_text(text)

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create an information thread about {query}
    Please follow all of the following rules:

    1/ Make sure the content is engaging, informative with good data
    2/ Make sure the content is not containing any irrelevant and prejudiced informations
    3/ The content should address the {query} topic very wells
    4/ The content needs to be written in a way that is easy to read and understand
    5/ The content needs to give audience actionable advice & insights too

    SUMMARY:
    """

    prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(texts):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    summaries = str(' '.join(summaries))
    return summaries

# Turn summarization into twitter thread
def generate_thread(summaries, query):

    text  = ' '.join(summaries)
    text = str(text)

    summaries = summarise_1(text, query)


    test_template = f"""
    {summaries}

    You are a world class journalist with expertise in braod aspects, and your task is to execute research on {query} using the text above, and generate information thread and following all rules below:
    1/ The thread needs to be engaging, informative with good and reliable data
    2/ The thread needs to be relevant, precise and interesting
    3/ The thread needs to address the topic - '{query}' very well
    4/ The thread needs to be written in a way that is transparent and unambiguous
    5/ The thread needs to give audience actionable advice & insights too with provocative measures

    Information THREAD:
    """ 

    token_count = count_tokens(test_template, tokenizer)
    if token_count > 4090:
        summaries = summarise_1(test_template, query)


    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.5)
    template = """
    {summaries_str}

    You are a world class journalist with expertise in braod aspects, and your task is to execute research on {query} using the text above, and generate information thread and following all rules below:
    1/ The thread needs to be engaging, informative with good and reliable data
    2/ The thread needs to be relevant, precise and interesting
    3/ The thread needs to address the {query} topic very well
    4/ The information and text provided must be informative, referential, reliable and unambiguous
    5/ The information must capture exact sequence of reasoning to conclude any statement in exact reliable fashion.


    Information THREAD:
    """

    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    info_thread_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    info_thread = info_thread_chain.predict(summaries_str=summaries, query=query)

    return info_thread

def connect_words_with_underscores(text):
    words = text.split()
    connected_string = '_'.join(words)
    return connected_string


# def main():
#     load_dotenv(find_dotenv())

#     st.set_page_config(page_title="Autonomous researcher", page_icon=":bird:")

#     st.header("Autonomous researcher")
#     query = st.text_input("Topic of information thread")

#     openai.api_key = OPENAI_API_KEY

#     if query:
#         print(f'{query}\n\n')
#         with st.spinner('Wait for the agent to study the internet . . . '):

#             st.write("Generating information thread for: ", query)
            
#             search_results = search(query)
#             urls = find_best_article_urls(search_results, query)
#             data = get_content_from_urls(urls)
#             summaries = summarise(data, query)
#             thread = generate_thread(summaries, query)

#         with st.expander("thread"):
#             st.info(str(thread))
#         with st.expander("best urls"):
#             st.info(urls)
#         with st.expander("search results"):
#             st.info(search_results)



# def main():
#     load_dotenv(find_dotenv())

#     st.set_page_config(page_title="Autonomous Researcher", page_icon=":bird:")

#     st.header("Autonomous researcher")

#     openai.api_key = OPENAI_API_KEY

#     with st.form(key='research_form'):
#         query = st.text_input("Topic of information thread")
#         submit_button = st.form_submit_button('Research')

#         if submit_button and query:
#             print(f'{query}\n\n')
#             with st.spinner('Wait for the agent to study the internet . . . '):

#                 st.write("Generating information thread for: ", query)
                
#                 search_results = search(query)
#                 urls = find_best_article_urls(search_results, query)
#                 data = get_content_from_urls(urls)
#                 summaries = summarise(data, query)
#                 thread = generate_thread(summaries, query)

#             with st.expander("thread"):
#                 st.info(str(thread))
#             with st.expander("best urls"):
#                 st.info(urls)
#             with st.expander("search results"):
#                 st.info(search_results)


# def main():
#     load_dotenv(find_dotenv())

#     st.set_page_config(page_title="Autonomous Researcher", page_icon=":bird:")

#     st.header("Autonomous researcher")

#     openai.api_key = OPENAI_API_KEY

#     with st.form(key='research_form'):
#         # query = st.text_input("Topic of information thread", on_change=reset_process) # We'll define the reset_process function below

#         query = st.text_input("Topic of information thread") # We'll define the reset_process function below
#         submit_button = st.form_submit_button('Research') # Added on_enter=True

#         if submit_button and query:
#             print(f'\n\n{query}\n\n')
#             with st.spinner('Wait for the agent to study the internet . . . '):

#                 st.write("Generating information thread for: ", query)
                
#                 search_results = search(query)
#                 urls = find_best_article_urls(search_results, query)
#                 data = get_content_from_urls(urls)
#                 summaries = summarise(data, query)
#                 thread = generate_thread(summaries, query)

#             with st.expander("thread", expanded=True): # Added expanded=True
#                 st.info(str(thread))
#             with st.expander("best urls", expanded=True): # Added expanded=True
#                 st.info(urls)
#             with st.expander("search results", expanded=True): # Added expanded=True
#                 st.info(search_results)

#             # Create a PDF of the webpage
#             pdfkit.from_string(str(thread), 'output.pdf')

#             # Create a button to download the PDF
#             if st.button('Download PDF'):
#                 st.download_button('Download PDF', f'{connect_words_with_underscores(query)}.pdf', 'application/pdf')

# def reset_process():
#     st.experimental_rerun() # This will reset the process if a new string is input


def main():
    load_dotenv(find_dotenv())
    print('\n\n')

    st.set_page_config(page_title="Autonomous Researcher", page_icon=":bird:")

    st.header("Autonomous Researcher")

    openai.api_key = OPENAI_API_KEY

    with st.form(key='research_form'):
        # Removed 'on_change=reset_process'
        query = st.text_input("Topic of information thread") 
        submit_button = st.form_submit_button('Research') 

        if submit_button and query:
            print(f'\n\n{query}\n\n')
            with st.spinner('Wait for the agent to study the internet . . . '):

                st.write("Generating information thread for: ", query)
                
                search_results = search(query)
                urls = find_best_article_urls(search_results, query)
                data = get_content_from_urls(urls)
                summaries = summarise(data, query)
                thread = generate_thread(summaries, query)

            # Added 'expanded=True' for all expanders
            with st.expander("thread", expanded=True): 
                st.info(str(thread))
            with st.expander("best urls", expanded=True):
                st.info(urls)
            with st.expander("search results", expanded=True):
                st.info(search_results)

            # Create a PDF of the webpage
            pdfkit.from_string(str(thread), f'{connect_words_with_underscores(query)}.pdf')

            # Create a button to download the PDF
            # Changed filename to use a helper function 'connect_words_with_underscores(query)'
            time.sleep(5)
            # st.download_button('Download PDF', f'{connect_words_with_underscores(query)}.pdf', 'application/pdf')
            
    

    # Download_button = st.button('Download PDF')
    # if Download_button:
    #     st.download_button('Download PDF', f'{connect_words_with_underscores(query)}.pdf', 'application/pdf')


if __name__ == '__main__':
    main()