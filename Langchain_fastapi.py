from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from Bio import Entrez
from datetime import datetime
from fastapi import FastAPI
from langserve import add_routes
import os
import re
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]



Template2 = """

{text}
------------------
using the text above, answer the following question in short and no more than 500 words.
Question: {question}
-------------------------
if the question can not be answered, dont make up answers and just write that the answer to your question is not withing the document.
"""
prompt = ChatPromptTemplate.from_template(template=Template2)


def get_entrez(keywords: list):
    # Set your email for PubMed requests (required)
    Entrez.email = "your_email"

    # Keywords list to search in PubMed
    keywords_list = keywords

    # Combine keywords with OR operator for PubMed query. you can also use AND
    keywords_query = ' AND '.join(keywords_list)

    # Get today's date in for the text file's name (YYYY/MM/DD)
    today_date = datetime.today().strftime('%Y-%m-%d')

    # All you need to search is the keywords query
    search_query = f'({keywords_query})'
    search_results = Entrez.read(
        Entrez.esearch(db="pubmed", term=search_query, retmax=10, datetype="pdat", reldate=30, usehistory="y"))
    webenv = search_results['WebEnv']
    query_key = search_results['QueryKey']
    id_list = search_results['IdList']
    all_summaries = []
    # Step 2: EFetch to retrieve titles based on the UIDs
    for i in id_list:
        fetch_handle = Entrez.efetch(db="pubmed", id=i, rettype="abstract", retmode="text", webenv=webenv,
                                     query_key=query_key)
        fetch_content = fetch_handle.read()
        all_summaries.append(fetch_content)  # Store title along with summary
    output = ''.join(all_summaries)
    return output


scrape_and_summarize = RunnablePassthrough.assign(
    text=lambda x: get_entrez(x['keywords'])[:30000]
) | prompt | ChatOpenAI(model='gpt-4-1106-preview', temperature=1) | StrOutputParser()

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 Pubmed search queries to search online that form an "
            "objective medical opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

KEY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 Pubmed search queries to search online that form an "
            "objective medical opinion from the following:{question}\n"
            "then extract important keyword combination from the each question"
            "You must respond with a list of strings in the following format: "
            '["keyword_combination 1", "keyword_combination 2", "keyword_combination 3"].',
        ),
    ]
)


def parse_keywords(output):
    # Initialize an empty set to store unique keywords
    unique_keywords = set()

    # Split the output into lines
    lines = output['question'].strip().split('\n')

    # Iterate through each line
    for line in lines:
        # Extract keywords using regular expression
        keywords = re.findall(r'"([^"]*)"', line)

        # Add keywords to the set
        unique_keywords.update(keywords)

    # Convert the set to a list and return
    return list(unique_keywords)


key_search = KEY_PROMPT | ChatOpenAI(model='gpt-4-1106-preview', temperature=1) | {
    'question': StrOutputParser()} | RunnablePassthrough.assign(
    text=lambda x: get_entrez(parse_keywords(x))[:40000]) | prompt |ChatOpenAI(model='gpt-4-1106-preview', temperature=1) | StrOutputParser()



WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


chain = RunnablePassthrough.assign(
    research_summary=key_search | collapse_list_of_lists
) | prompt | ChatOpenAI(model='gpt-4-1106-preview', temperature=1) | StrOutputParser()

system_message = st.text_area("Enter your question")
analyze_button = st.button("Answer")
if analyze_button:
    answer= chain.invoke(
        {
            "question":system_message
        }
    )
    st.write(answer)



