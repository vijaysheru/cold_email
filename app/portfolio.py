import pandas as pd
import chromadb
import uuid
import os


class Portfolio:
    def __init__(self, file_path="resource/my_portfolio.csv"):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.file_path = os.path.join(current_dir, file_path)

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"CSV file not found at {self.file_path}")

            self.data = pd.read_csv(self.file_path)
            self.chroma_client = chromadb.PersistentClient(path="vectorstore")
            self.collection = self.chroma_client.get_or_create_collection(name='portfolio')


    def load_portfolio(self):
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                techstack = row.get("Techstack", "")
                link = row.get("Links", "")
                if techstack and link:
                    self.collection.add(
                        documents=[techstack],
                        metadatas={"links": link},
                        ids=[str(uuid.uuid4())]
                    )

    def query_links(self, skills):
        if isinstance(skills, list):
            skills = ", ".join(skills)
        response = self.collection.query(
            query_texts=[skills],
            n_results=2
        )
        return response.get('metadatas', [])


