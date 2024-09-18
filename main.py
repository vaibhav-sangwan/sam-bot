import os
import requests
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: 

{question}
---
"""
load_dotenv()
HF_API_KEY = os.getenv("L38BINSTRUCT_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Step 4: Function to generate chat response
def get_chat_response(query):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=HuggingFaceEmbeddings())

    # Search the DB for relevant context
    results = db.similarity_search_with_score(query=query, k=5)
    
    # Combine the context from the top results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    
    response = bot_response(
        payload={
            "inputs": prompt,
            "parameters": {"temperature": 0.8, "max_new_tokens": 2048, "seed": 1},
        }
    )
    return response[0]['generated_text'].split("---")[-1]

def bot_response(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

if __name__ == '__main__':
    response = get_chat_response("How do I get better at understanding sign language?")
    print(response)
