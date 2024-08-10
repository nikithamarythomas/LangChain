import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate


# Retrieve the API key from Streamlit secrets
api_key = st.secrets["openai"].get("api_key")

# Initialize OpenAI LLM
llm = OpenAI(api_key=api_key)

# Define your prompt template (use 'context' as the input variable)
prompt_template = PromptTemplate(
    template="Please summarize the following document:\n\n{context}",
    input_variables=["context"]
)


# Create the chain
stuff_documents_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

def main():
    st.title("Document Summarizer")

    # Upload file
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    
    if uploaded_file is not None:
        # Read file content
        document_text = uploaded_file.read().decode("utf-8")

        # Create a Document object
        documents = [Document(page_content=document_text)]

        # Execute the chain with documents
        result = stuff_documents_chain.invoke({"context": documents})
        
        # Display the result
        st.subheader("Summary")
        st.write(result)

if __name__ == "__main__":
    main()
