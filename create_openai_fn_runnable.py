import streamlit as st
from typing import Optional
from langchain.chains.structured_output import create_openai_fn_runnable
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the Pydantic models
class RecordPerson(BaseModel):
    '''Record some identifying information about a person.'''
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

class RecordDog(BaseModel):
    '''Record some identifying information about a dog.'''
    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["openai"].get("api_key")


# Initialize the OpenAI model
llm = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0)

# Create a structured output runnable
structured_llm = create_openai_fn_runnable([RecordPerson, RecordDog], llm)

def main():
    st.title("Structured Output with OpenAI")

    # Input for person record
    st.subheader("Person Record")
    input_text_person = st.text_area("Enter a description of a person:")
    
    if st.button("Generate Person Record"):
        if input_text_person:
            result_person = structured_llm.invoke(input_text_person)
            st.write("Person Record:", result_person)
        else:
            st.error("Please enter a description for the person.")

    # Input for dog record
    st.subheader("Dog Record")
    input_text_dog = st.text_area("Enter a description of a dog:")
    
    if st.button("Generate Dog Record"):
        if input_text_dog:
            result_dog = structured_llm.invoke(input_text_dog)
            st.write("Dog Record:", result_dog)
        else:
            st.error("Please enter a description for the dog.")

if __name__ == "__main__":
    main()
