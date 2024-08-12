import streamlit as st
import json
import time
import os
import re

# langchain imports
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# set page config
st.set_page_config(page_title="The Golden Retriever", layout="wide")

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
# 4. Pass previous generated response to embedding search

@st.cache_resource 
def load_embeddings(_model_name):
    return HuggingFaceEmbeddings(model_name=_model_name,model_kwargs={"trust_remote_code": True})

#  get GROQ API KEY 
os.environ["GROQ_API_KEY"]  = "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"

# choice of LLM 
# llm = ChatGroq(model="llama-3.1-8b-instant")
# llm = ChatGroq(model="mixtral-8x7b-32768")
llm = ChatGroq(model="llama-3.1-70b-versatile")

# load embedding model
model_name = "Alibaba-NLP/gte-large-en-v1.5"
embeddings = load_embeddings(_model_name=model_name)

# locate parent and child documents
parent_doc_path = "/Users/mehuljain/Documents/course_related/Capstone/rag_ds5500/vector_db/parent_docs"
child_doc_path = "./vector_db/child_docs"

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Load Chroma vectorstore
loaded_vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory= child_doc_path
)
 
# Load parent document store
loaded_file_store = LocalFileStore(parent_doc_path)

# cache the loaded retriever
@st.cache_resource
def store_retreiver(_vs, _fs, _cs , _ps):

    return CustomParentDocumentRetriever(
        vectorstore=_vs,
        docstore=_fs,
        child_splitter=_cs,
        parent_splitter=_ps
        )

# Use the custom retriever
loaded_retriever =store_retreiver(
    _vs= loaded_vectorstore, 
    _fs= loaded_file_store, 
    _cs= child_splitter, 
    _ps= parent_splitter
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

# question answer chain without history 
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

# contextualize the question answering prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# create a history aware retriever
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

# Q&A prompt
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
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@st.cache_resource
def load_sim_embed():
    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    return model

# load similarity model
sim_embed= load_sim_embed()

# function to measure the similarity between two embeddings
def measure_similarity(model,question, prev_answer):

    # Get embeddings for the paragraphs
    embeddings = model.encode([question, prev_answer], batch_size=2, convert_to_tensor=False)

    # Get embeddings for the paragraphs
    embeddings1 = embeddings[0]
    embeddings2 = embeddings[1]

    # calculate similarity
    similarity = 1 - cosine(embeddings1, embeddings2)

    return similarity

# function to retrieve the documents
def retreive_docs(question):

    # return the references
    if "abc126" in st.session_state.store:
        prev_message= st.session_state.store["abc126"].messages[-1].content
        print(f"Similarity between current question and past answer:{measure_similarity(sim_embed, question, prev_message)}")
        if measure_similarity(sim_embed, question, prev_message)>0.4:
            question= question + " " + prev_message

    docs_ret= loaded_retriever.invoke(question)
    # print(question)
    # print(docs_ret)
    ref= [doc.metadata['source'] for doc in docs_ret]
    ref= list(set(ref))

    # put the references
    refs=[]
    for i, j in enumerate(ref):
        refs.append(f"**Link {i+1}:** {j}")

    return refs

# function to answer using RAG:
def generate_answer(question):

    # retrieve documents
    refs= retreive_docs(question)
    
    # generate the answer
    answer= conversational_rag_chain.invoke(
    {"input": question},
    config={
        "configurable": {"session_id": "abc126"}
    },  # constructs a key "abc123" in `store`. This should be a dynamic 6 digit key that should change every time 
    )["answer"]

    # convert the answer into markdown format 
    words_with_spaces = re.findall(r'\S+|\s+', answer)
    # print(words_with_spaces)

    # yield the answer word by word
    for word in words_with_spaces:
        yield word
        time.sleep(0.05)
    
    yield '\n\n'
    yield '**References:**'
    yield '\n\n'

    for r in refs:
        yield r
        yield '\n\n'

def generate_llm(prompt):

    # invoke the LLM
    answer =llm.invoke(prompt).content

    # generate answer
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)

# STREAMLIT CODE UI

st.title("Golden Retriever 🐕: In-House RAG System")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            
            response = generate_answer(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

            # check out the output session state
            # st.write(st.session_state.store)

    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)