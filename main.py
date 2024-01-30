from dotenv import load_dotenv
load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile

st.title("Chatpdf")
st.write("pdf 파일을 업로드하면, 그 안에 있는 내용을 읽어서 질문에 답해줍니다.")
st.write("---")

# Load the PDF file
uploaded_file = st.file_uploader("Choose a PDF file",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages

#if file is uploaded and working
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,  # 100 characters
        chunk_overlap=20, # 겹치는 부분
        length_function=len, # len() 함수를 이용하여 길이를 계산
        is_separator_regex=False, # 구분자가 정규표현식인지 여부
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chorma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("PDF 파일을 업로드하고, 질문을 입력하세요.")
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        with st.spinner('질문 중...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
