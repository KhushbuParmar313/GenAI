from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
# from langchain.embeddings import  GPT4AllEmbeddings
from langchain.chains import RetrievalQA
import time

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from nemoguardrails import LLMRails, RailsConfig
# import asyncio 
import nest_asyncio

nest_asyncio.apply()

ollama = Ollama(base_url='http://localhost:11434',
model="llama2")

config = RailsConfig.from_path("./ins_assistant/config")
app = LLMRails(config)

# @st.cache_resource()
#Extract JSON file
def get_json_text(json_files):
    for files in json_files:
        loader = JSONLoader(file_path=files.name, jq_schema=".", text_content=False)
    documents = loader.load()
    
    print("\nJSON DATA",documents)

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=documents, embedding=oembed, persist_directory="./chroma_db")
    vectorstore.persist()
    # db = Chroma.from_documents(documents, embedding_function)

#Extracting Text of Pdf
def get_pdf_text(pdf_docs):
    # print(pdf_docs)
    total_data = []
    for files in pdf_docs:
        loader = PyPDFLoader(files.name)
        data = loader.load()
        total_data.extend(data)
    # print("\n\n***************************\nPDF data:",type(data))

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(total_data)

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory="./chroma_db")
    vectorstore.persist()
    
    
def get_url_text(url):
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    # client = chromadb.HttpClient(host='127.0.0.1', port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory="./chroma_db")
    vectorstore.persist()

def load_url_query(question):

    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=oembed)
    
     # Prompt
    template = """ Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    
    Answer:
    """

    # config = RailsConfig.from_path("./ins_assistant/config")
    # app = LLMRails(config)
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever(),chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # app.register_action(qachain, name="qa_chain")

    # result = app.generate(messages=question)
    result = qachain.invoke({"query": question})
    # print(result)
    st.write("Reply: ",result["result"])

def main():
    st.set_page_config("Chat URL")
    st.header("Chat with your data using LLM 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        start = time.time()
        load_url_query(user_question)
        end = time.time()
        st.write(round(end-start,2), "seconds")

    with st.sidebar:
        st.title("Menu:")
        url_docs = st.text_input("Add your url")
        # pdf_docs = st.file_uploader("Upload your Link", accept_multiple_files=True)
        

        all_files = st.file_uploader("Upload your Files ", accept_multiple_files=True)
        if st.button("Submit"):
            # print("Path:",pdf_docs)
            with st.spinner("Processing..."):
                pdf_list = []
                json_list = []
                for file in all_files:
                    if file.name.endswith('pdf'):
                        pdf_list.append(file)
                    elif file.name.endswith('json'):
                        json_list.append(file)
                
                print('\n**********************\n URL UPLOADED:', url_docs)

                if pdf_list != []:
                    get_vector_store = get_pdf_text(pdf_list)
                # st.write(get_vector_store)      

                #if json
                if json_list !=  []:
                    get_vector_store = get_json_text(json_list)
                # st.write(get_vector_store)
               

                if url_docs != '':
                    get_vector_store = get_url_text(url_docs)

                st.success("Done")

if __name__ == "__main__":
    main()