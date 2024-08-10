import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import create_history_aware_retriever
from langchain import hub
from langchain_community.retrievers import PubMedRetriever
import os

# Pull the rephrase prompt from LangChain's hub
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["openai"].get("api_key")

# Initialize the OpenAI model
llm = OpenAI(api_key=api_key)

# Initialize the PubMed retriever
retriever = PubMedRetriever()

# Create the history-aware retriever chain
chat_retriever_chain = create_history_aware_retriever(
    prompt=rephrase_prompt,
    llm=llm,
    retriever=retriever
)

def write_result_to_text(result, filename='output_result.txt'):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            if isinstance(result, list):
                if not result:
                    file.write("Result is an empty list. No data to write.\n")
                else:
                    for item in result:
                        file.write(str(item) + '\n')
            else:
                file.write("Result format is not supported for text output.\n")
        st.success(f"Result has been written to {filename}")
    except Exception as e:
        st.error(f"An error occurred while writing to text file: {e}")

def main():
    st.title("History-Aware Retriever")

    # Input for query
    query = st.text_input("Enter your query:", "What are the mental health issues in adults?")

    # Input for chat history
    chat_history = st.text_area("Enter chat history (comma-separated):", "Tell me more about the health issues.")
    
    # Button to generate result
    if st.button("Generate Result"):
        if query:
            # Invoke the chain with input and chat history
            result = chat_retriever_chain.invoke({
                "input": query,
                "chat_history": chat_history.split(',')
            })
            
            # Display the result
            st.subheader("Result")
            st.write(result)

            # Save the result to a text file
            write_result_to_text(result)
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()
