import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_text
from portfolio import Portfolio

def create_streamlit_app(llm, portfolio, clean_text):
    st.title(" Cold Email Generator ")
    url_input = st.text_input("Enter a URL:",value="")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                skills_query = ", ".join(skills)  # 🔥 Fix here!
                links = portfolio.query_links(skills_query)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An error occured while processing {e}")

if __name__ == '__main__':
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide",page_title="Cold Email Generator")
    create_streamlit_app(chain, portfolio, clean_text)