from llama_index.retrievers.bm25 import BM25Retriever

import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import time
from llama_index.core import PromptTemplate

Settings.llm  = Ollama(model="llama2", request_timeout=200000.0, temperature=0)

Settings.embed_model = OllamaEmbedding(base_url="http://localhost:11434", 
                                       model_name="llama2")
Settings.chunk_size = 256

# Ollama.additional_kwargs = {"top_p": 0.65}

import os
import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# @st.cache_resource()

st.set_page_config(page_title="GenAI Assistant", layout="wide")

def read_files(docs):
    
    for files in os.listdir('files'):
        if files not in docs:
            os.remove('./files/'+files)
        
    documents = SimpleDirectoryReader('./files').load_data()
    # print("DOCUMENTS",documents)
    splitter = SentenceSplitter(chunk_size=256)
    start_time = time.time()
    index = VectorStoreIndex.from_documents(documents, transformations = [splitter])
    end_time = time.time()
    # print("INDEXS",index)
    index.storage_context.persist("doc_index") 
    st.write(round(end_time - start_time,2))
    return index

def retriver(user_question, reference):
    storage_context = StorageContext.from_defaults(persist_dir="doc_index")
    index = load_index_from_storage(storage_context)
    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore,similarity_top_k=5)
   
    # retriever1 = bm25_retriever.retrieve(user_question)
    # retriever2 = vector_retriever.retrieve(user_question)
    # st.write(len(retriever1))
    # st.write('bm25:', retriever1[0].metadata)
    # st.write('vector:', retriever2)

    retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],
                                    retriever_weights=[0.25, 0.75],
                                    similarity_top_k=6,
                                    num_queries=1,  # set this to 1 to disable query generation
                                    mode="relative_score",
                                    use_async=False,
                                    verbose=True,)
    
    similar_chunks = retriever.retrieve(user_question)
    # top_chunks = similar_chunks[:2]
    if reference == True:
        page_index = []
        similar_chunks = retriever.retrieve(user_question)
        result_source = {}
        # chunks = [chunk for chunk in similar_chunks if chunk.score>=0.65]
        # st.write('Chunk type:',i)
        for i in similar_chunks:
            pdf_name = i.metadata['file_name']
            page_label = i.metadata['page_label']
            result_source[pdf_name] = page_label
            k+=1
            if k < 3:
                break
        st.write(result_source)
    # st.write('fusion:',len(retriever.retrieve(user_question)))
    
    # st.write('fusion:',type(retriever.retrieve(user_question)))
    query_engine = RetrieverQueryEngine.from_args(retriever)
    # st.write("QUERY_ENGINE",query_engine)
    response = query_engine.query(user_question)
    return response

def main():

    model_name = st.selectbox('Choose a model',
    ('llama2:7b', 'llama2:13b', 'mistral','phi','llava', 'llama3'))
    if model_name == 'llama2:7b':
        model_name = 'llama2'

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    reference = st.checkbox('Get page index')
    if user_question:
        # Rebquery_engineuild storage context
        # storage_context = StorageContext.from_defaults(persist_dir="doc_index")
        # index = load_index_from_storage(storage_context)
        # query_engine = index.as_query_engine()
        # response = query_engine.query(user_question)
            # print(type(response), response)    
        
        # st.write("Reply: ", str(response))
        start = time.time()
        response = retriver(user_question,reference)
        end = time.time()
        st.write("Reply", str(response))
        st.write(round(end-start,2), "seconds")

        
    with st.sidebar:
        st.title("Menu:")
        url_docs = st.text_input("Add your url")
        # pdf_docs = st.file_uploader("Upload your Link", accept_multiple_files=True)

        all_files = st.file_uploader("Upload your Files ", accept_multiple_files=True)
        if st.button("Submit"):
            # print("Path:",pdf_docs)
            with st.spinner("Processing..."):
                files_list = []
                for file in all_files:
                    saved_file_path  = os.path.join('./files', file.name)
                    with open(saved_file_path, mode='wb') as w:
                        w.write(file.getvalue())
                    if file.name.endswith('pdf') or file.name.endswith('json') or  file.name.endswith('txt'):
                        files_list.append(file.name)

                if files_list != []:
                    read_files(files_list)      

                

                st.success("Done")

if __name__ == "__main__":
    main()