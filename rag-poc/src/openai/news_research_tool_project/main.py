import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Sample web based assistant")
st.sidebar.title("Please enter News Article URLS")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"USL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process the above provided URLS")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm=OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started......")
    data=loader.load()

    #split the loaded data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=800
    )
    main_placeholder.text("Text splitting happening ......")
    docs=text_splitter.split_documents(data)

    #create embeddings and save to FAISS index
    embedddings = OpenAIEmbeddings()

    vectorstore_openai = FAISS.from_documents(docs,embedddings)

    main_placeholder.text("Embedding vector started building .... ")
    time.sleep(3)

    #save the FAISS index to a pickle file
#    with open(file_path,"wb") as f:
 #       pickle.save_local(vectorstore_openai,f)
    # Save the vectorstore object locally
    vectorstore_openai.save_local("vectorstore")

query = main_placeholder.text_input("Question: ")
if query:
    print("================")
    print(query)
   # if os.path.exists(file_path):
    #    with open(file_path, "rb") as f:
     #       vectorstore = pickle.load_local(f)

    x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)


    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=x.as_retriever())
    result = chain({"question":query}, return_only_outputs=True)
    print(result)

    st.header("Answer  ")
    st.write(result["answer"])

    sources = result.get("sources","")

    if sources:
        st.subheader("Sources:")
        sources_list=sources.split("\n")
        for source in sources_list:
            st.write(source)

