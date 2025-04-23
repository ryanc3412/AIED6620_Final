import openai
import faiss
import pandas as pd
import numpy as np
import os


openai.api_key = "Your-API-Key-Here"
client = openai.OpenAI(api_key=openai.api_key)

#loads the data from a text file
def load_data(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if ":" in line:
            code, content = line.strip().split(":", 1)
            data.append({"code": code.strip(), "content": content.strip()})
    return pd.DataFrame(data)

#embeds the data using OpenAI's embedding model
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def embed_data(df):
    df["embedding"] = df["content"].apply(get_embedding)
    return df

#faiss index creation to be able to search through vectors
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    matrix = np.array(embeddings).astype("float32")
    index.add(matrix)
    return index

#searching the index for the best match
def search_index(query, index, df):
    query_code = query.strip().upper()

    #try exact match first
    exact_match = df[df["code"] == query_code]
    if not exact_match.empty:
        return exact_match.iloc[0]

    #fall back to semantic search
    query_vec = np.array(get_embedding(query)).astype("float32")
    D, I = index.search(np.array([query_vec]), k=1)
    best_index = I[0][0]
    return df.iloc[best_index]
