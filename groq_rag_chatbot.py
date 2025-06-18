import os
import warnings
import logging
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('ALchatty')

if 'messages' not in st.session_state:
    st.session_state.messages = []

#  Let user upload a PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

#  Process uploaded PDF
def build_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    return index.vectorstore

vectorstore = None
if uploaded_file:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    vectorstore = build_vectorstore("temp_uploaded.pdf")

# Display all messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Ask something about your uploaded PDF')

if prompt and vectorstore:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are very smart at everything, you always give the best, most accurate and precise answers.
        Answer the following Question: {user_prompt}. Start the answer directly. No small talk please.
    """)

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True
    )

    try:
        result = chain({"query": prompt})
        response = result["result"]
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
    except Exception as e:
        st.error(f"Error: {str(e)}")

elif prompt:
    st.error("Please upload a PDF first!")


