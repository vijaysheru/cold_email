import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE: 
            {page}
            ### INSTRUCTION : 
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following
            keys: 'role', 'experience', 'skills', and 'description'.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE)
            """
        )

        formatted_prompt = prompt_extract.format(page=cleaned_text)
        res = self.llm.invoke(formatted_prompt)

        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Content too big or LLM output not in valid JSON format")

        return parsed if isinstance(parsed, list) else [parsed]

    def write_email(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Vijay Sheru, an AI Engineer and Software Developer with a strong background in building and deploying AI-powered applications and scalable automation systems.
            You specialize in AI & Software solutions that help businesses streamline operations, reduce costs, and scale effectively. With expertise in Python, TensorFlow, PyTorch, Hugging Face, LangChain, FastAPI, Flask, AWS, and MLOps, you design and implement cutting-edge AI tools tailored to business needs.
            Add the Dear Hiring Manager for every job and then remove brackets where you don't know what to fill.
            Over your career, you have successfully:
            Developed advanced NLP systems such as AI Response Aggregators that detect AI-generated content using state-of-the-art transformer models (BERT, GPT-4) and integrate humanization layers using LSTM and BERT-based text generation techniques.
            Engineered deep learning applications like Neural Style Transfer (NST) with MobileNetV2, delivering optimized, real-time AI-generated artwork through GPU-accelerated inference.
            Built fraud detection systems using neural networks and reinforcement learning, significantly improving fraud detection accuracy while reducing false positives.
            Led the development of APIs and automation systems, optimized microservices, and deployed production-grade solutions on AWS EC2, S3, and platforms like Railway and Vercel.
            You bring a unique combination of software engineering (Java, Spring Boot, REST APIs) and AI/ML expertise, allowing you to build end-to-end solutions from model training to cloud deployment.

            Your role is to craft a cold email to prospective clients based on the provided job description, showcasing your ability to deliver impactful AI & software solutions that align with their project needs.

            Additionally, reference the most relevant projects and case studies from the following links: {link_list}.

            Use a professional, confident tone to position yourself as an expert AI & Software Engineer who delivers measurable business outcomes.
            ### EMAIL (NO PREAMBLE):

            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))