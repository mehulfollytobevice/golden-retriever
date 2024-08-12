import streamlit as st
import time
import os

# langchain imports
from langchain import hub
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_core.documents import Document
import json
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# UTILITY FUNCTIONS
class CustomParentDocumentRetriever(ParentDocumentRetriever):
    def invoke(self, query, config=None, **kwargs):
        results = super().invoke(query, config=config, **kwargs)
        documents = []
        for result in results:
            try:
                if isinstance(result, bytes):
                    json_string = result.decode('utf-8')
                elif isinstance(result, str):
                    json_string = result
                else:
                    print(f"Unexpected result type: {type(result)}")
                    continue

                deserialized_result = json.loads(json_string)
                
                # Extract metadata and page_content from the deserialized result
                metadata = deserialized_result['kwargs']['metadata']
                page_content = deserialized_result['kwargs']['page_content']

                doc = Document(metadata=metadata, page_content=page_content)
                documents.append(doc)
            except Exception as e:
                print(f"Error processing result: {e}")
                print(f"Problematic result: {result}")

        return documents


# NOTES:
# 1. Place options for choosing model
# 2. Format markdown output

#  get GROQ API KEY 
os.environ["GROQ_API_KEY"]  = "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"
# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatGroq(model="llama-3.1-70b-versatile")

# load embedding model
model_name = "Alibaba-NLP/gte-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"trust_remote_code": True})

# locate parent and child documents
parent_doc_path = "/Users/mehuljain/Documents/course_related/Capstone/rag_ds5500/vector_db/parent_docs"
child_doc_path = "./vector_db/child_docs"

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# The vectorstore to use to index the child chunks

# Load Chroma vectorstore
loaded_vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory= child_doc_path
)
 
# Load parent document store
loaded_file_store = LocalFileStore(parent_doc_path)

# Use the custom retriever
loaded_retriever = CustomParentDocumentRetriever(
    vectorstore=loaded_vectorstore,
    docstore=loaded_file_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(loaded_retriever, question_answer_chain)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, loaded_retriever, contextualize_q_prompt
)

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# function to answer using RAG:
def generate_answer(question):

    # return the references
    docs_ret= loaded_retriever.invoke(question)
    # print(question)
    # print(docs_ret)
    ref= [doc.metadata['source'] for doc in docs_ret]
    
    # generate the answer
    answer= conversational_rag_chain.invoke(
    {"input": question},
    config={
        "configurable": {"session_id": "abc122"}
    },  # constructs a key "abc123" in `store`. This should be a dynamic 6 digit key that should change every time 
    )["answer"]

    # print(answer)

    # put the references
    for i, j in enumerate(ref):
        answer= answer + f" \n Link {i+1}: {j} \n "

    return ref, answer

# STREAMLIT UI code 
st.set_page_config(page_title="The Knowledge Keeper", layout="wide")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: darkorange;
        font-size: 44px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
st.markdown("<div class='title'>In-house RAG System</div>", unsafe_allow_html=True)
 
 
if 'history' not in st.session_state:
    st.session_state.history = []
if 'focus_index' not in st.session_state:
    st.session_state.focus_index = -1
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = ""
 
 
def display_chat():
    with chat_container:
        for i, entry in enumerate(st.session_state.history):
            if entry['type'] == 'question':
                if i == st.session_state.focus_index:
                    st.write(f"**You:** {entry['text']} ▲")
                else:
                    st.write(f"**You:** {entry['text']}")
            elif entry['type'] == 'answer':
                st.write(f"**Answer:** {entry['text']}")
            if i == st.session_state.focus_index:
                st.markdown(f"<a name='answer_{i}'></a>", unsafe_allow_html=True)
 
 
st.sidebar.title("Navigation")
st.sidebar.header("Query History")

#st.sidebar.write("Query History:")
if st.session_state.history:
    for i, entry in enumerate(st.session_state.history):
        if entry['type'] == 'question':
            if st.sidebar.button(entry['text'], key=f"button_{i}"):
                st.session_state.focus_index = i
                st.session_state.button_pressed = entry['text']
                st.session_state.query_params = st.query_params
                st.query_params.update(st.session_state.query_params)

# Add a reset button to clear the chat history
if st.sidebar.button("Reset Chat"):
    st.session_state.history = []
    st.session_state.focus_index = -1
    st.session_state.query_input = ""
    st.session_state.button_pressed = ""
 
 
st.header("Chat")
 
 
chat_container = st.container()
 
 
st.write("<br>", unsafe_allow_html=True)
 
 
with st.form(key='query_form', clear_on_submit=True):
    query_input = st.text_input("Enter your query:", value=st.session_state.query_input)
    submit_button = st.form_submit_button("▲ Send")
 
    if submit_button:
        prompt = query_input or st.session_state.button_pressed
 
        if prompt:
        
            message_placeholder = st.empty()

            # query the RAG system and get the answer
            ref, answer = generate_answer(prompt)
 
            st.session_state.history.append({'type': 'question', 'text': prompt})
          
            typing_message = ""
            for chunk in answer.split():
                typing_message += chunk + " "
                message_placeholder.markdown(typing_message + "▌")
                time.sleep(0.05)
 
            # add text answer to the history  
            st.session_state.history.append({'type': 'answer', 'text': typing_message.strip()})

            # add references to the history
            # st.session_state.ref_list= ref
            
            st.session_state.query_input = ""
            st.session_state.button_pressed = ""
 
            
            st.session_state.query_params = st.query_params
            st.query_params.update(st.session_state.query_params)
 
 
display_chat()