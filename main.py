import openai
import faiss
import pandas as pd
import numpy as np
import os
from data_and_rag import load_data, embed_data, build_faiss_index, search_index, client


#main
def main():
    file_path = "diagnostic_data.txt"
    print("Loading and parsing data...")
    df = load_data(file_path)

    print("Generating embeddings from OpenAI...")
    df = embed_data(df)

    print("Building FAISS index...")
    index = build_faiss_index(df["embedding"].tolist())

    while True:
        query = input("Enter your error code question (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = search_index(query, index, df)

        #store original context separately for re-use
        original_context = (
            f"You are a Jeep diagnostic assistant. This is the info for code {result['code']}:\n\n"
            f"{result['content']}\n\nAnswer user questions based on that. Keep it focused and practical,"
            f"and don't ask them to make sure to enter more OBD code, they will if they need to."
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