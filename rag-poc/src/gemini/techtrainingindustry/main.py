import streamlit as st
from helper import get_qa_chain, create_vector_db

st.title("Code basics QA :  ")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()
question=st.text_input("Questions : ")
if question:
    chain = get_qa_chain()
    response = chain(question)
    print(response)
    st.header("Answer ")
    st.write(response["result"])

