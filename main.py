import openai
import faiss
import pandas as pd
import numpy as np
import os

openai.api_key = "your_api_key"
client = openai.OpenAI(api_key=openai.api_key)

#loads the data from a text file
#test
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
    query_vec = np.array(get_embedding(query)).astype("float32")
    D, I = index.search(np.array([query_vec]), k=1)
    best_index = I[0][0]
    return df.iloc[best_index]

#main
def main():
    file_path = "diagnostic_data.txt"
    print("Loading and parsing data...")
    df = load_data(file_path)

    print("Generating embeddings from OpenAI...")
    df = embed_data(df)

    print("Building FAISS index...")
    index = build_faiss_index(df["embedding"].tolist())

    print("\nSetup complete! You can now ask questions like 'How do I fix P0016?'\n")

    while True:
        query = input("Enter your error code question (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = search_index(query, index, df)

        #store original context separately for re-use
        original_context = (
            f"You are a Jeep diagnostic assistant. This is the info for code {result['code']}:\n\n"
            f"{result['content']}\n\nAnswer user questions based on that. Keep it focused and practical."
        )

        chat_history = [
            {"role": "user", "content": f"What's the first thing I should check for code {result['code']}?"}
        ]

        while True:
            messages = [{"role": "system", "content": original_context}] + chat_history
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5
            )
            reply = response.choices[0].message.content
            print(f"\n{reply}\n")

            follow_up = input("You: ")
            if follow_up.lower() in ("exit", "new code", "quit"):
                print("Ending this conversation.")
                break
            chat_history.append({"role": "user", "content": follow_up})
            chat_history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
