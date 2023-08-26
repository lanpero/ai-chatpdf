# from dotenv import load_dotenv
# load_dotenv()
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings

# # Loader
# loader = PyPDFLoader("luckyday.pdf")
# pages = loader.load_and_split()

# # Split
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 300,
#     chunk_overlap  = 20,
#     length_function = len,
#     is_separator_regex = False,
# )
# texts = text_splitter.split_documents(pages)

# # Embedding
# embeddings_model = OpenAIEmbeddings()

# # load it into Chroma
# db = Chroma.from_documents(texts, embeddings_model)

# #  pip install chromadb

# print(pages[0])

# cmd : 
# python .\main.py

#---------pdf 1
# from dotenv import load_dotenv
# load_dotenv()
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# #Loader
# loader = PyPDFLoader("unsu.pdf")
# pages = loader.load_and_split()

# #Split
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 300,
#     chunk_overlap  = 20,
#     length_function = len,
#     is_separator_regex = False,
# )
# texts = text_splitter.split_documents(pages)

# #Embedding
# from langchain.embeddings import OpenAIEmbeddings
# embeddings_model = OpenAIEmbeddings()

# from dotenv import load_dotenv
# load_dotenv()
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.chains import RetrievalQA
# import streamlit as st
# # import pandas as pd
# # from io import StringIO
# import tempfile
# import os

# # title
# st.title("ChatPDF")
# st.write("---")

# # upload file

# uploaded_file = st.file_uploader("Choose a file")
# st.write("---")

# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     loader = PyPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages

# #업로드 되면 동작하는 코드
# if uploaded_file is not None:
#     pages = pdf_to_document(uploaded_file)
    

# #  업로드 되면 동작하는 코드
# # if uploaded_file is not None:
# #     # To read file as bytes:
# #     bytes_data = uploaded_file.getvalue()
# #     st.write(bytes_data)

# #     # To convert to a string based IO:
# #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
# #     st.write(stringio)

# #     # To read file as string:
# #     string_data = stringio.read()
# #     st.write(string_data)

# #     # Can be used wherever a "file-like" object is accepted:
# #     dataframe = pd.read_csv(uploaded_file)
# #     st.write(dataframe)

# # 위 코드에 들어가 있어
# # #Loader
# # loader = PyPDFLoader("luckyday.pdf")
# # pages = loader.load_and_split()

# #Split
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 300,
#     chunk_overlap  = 20,
#     length_function = len,
#     is_separator_regex = False,
# )
# texts = text_splitter.split_documents(pages)

# #Embedding
# embeddings_model = OpenAIEmbeddings()

# # load it into Chroma
# db = Chroma.from_documents(texts, embeddings_model)

# #Question
# st.header("PDF에게 질문해보세요!!")
# question = st.text_input('질문을 입력하세요.')

# if st.button('질문하기'):
#     with st.spinner('Wait for it...'):
    
#     # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
#         result = qa_chain({"query": question})
#         st.write(result["result"])
    
#     # streamlit run .\main.py


# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# # from dotenv import load_dotenv
# # load_dotenv()
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# # from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.chains import RetrievalQA
# import streamlit as st
# import tempfile
# import os

# #제목
# st.title("ChatPDF")
# st.write("---")

# #파일 업로드
# uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
# st.write("---")

# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     loader = PyPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages

# #업로드 되면 동작하는 코드
# if uploaded_file is not None:
#     pages = pdf_to_document(uploaded_file)

#     #Split
#     text_splitter = RecursiveCharacterTextSplitter(
#         # Set a really small chunk size, just to show.
#         chunk_size = 300,
#         chunk_overlap  = 20,
#         length_function = len,
#         is_separator_regex = False,
#     )
#     texts = text_splitter.split_documents(pages)

#     #Embedding
#     embeddings_model = OpenAIEmbeddings()

#     # load it into Chroma
#     db = Chroma.from_documents(texts, embeddings_model)

#     #Question
#     st.header("PDF에게 질문해보세요!!")
#     question = st.text_input('질문을 입력하세요')

#     if st.button('질문하기'):
#         with st.spinner('Wait for it...'):
#             llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#             qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
#             result = qa_chain({"query": question})
#             st.write(result["result"])

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])