import os

from langchain_community.llms.google_palm import GooglePalm
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#llm=genai.GenerativeModel('gemini-pro')
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
instructor_embeddings=HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
vector_db_filepath="faiss_index"


def create_vector_db():
    loader=CSVLoader(file_path="codebasics_faq.csv",source_column="prompt")
    data=loader.load()
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    vectordb.save_local(vector_db_filepath)

def get_qa_chain():
    vectordb=FAISS.load_local(vector_db_filepath,instructor_embeddings,allow_dangerous_deserialization=True)
    retriever=vectordb.as_retriever(score_threshold=0.7)
    prompt_template="""Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much as text possible from "response" section in the source document context without making much changes.
    if the answer is not found in the context, kindly say "I don't know". Don't try to make up an answer.

    CONTEXT: {context}
    
    QUESTIION: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":prompt})
    return chain


if __name__ == "__main__":
    #create_vector_db()
    chain = get_qa_chain()
    print(chain("what is the duration of this bootcamp"))

