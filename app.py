import streamlit as st
from langchain.chains import LLMChain

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

embedding_models = OpenAIEmbeddings()

chroma_client = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_models,
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = chroma_client.as_retriever(search_kwargs={"k": 2})

prompt_template = """Provided information is below
    ---------------------
    {context}
    ---------------------
    Given the provided information and no prior knowledge, 
    answer the question: {question}
    Answer:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question_with_context(question):
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model_name="gpt-3.5-turbo")
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

with st.sidebar:
    st.header("Ask Me About FC Cincinnati!")

st.title("Ask Me About FC Cincinnati!")

if "messages" not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Ask me questions related to Ask Me About FC Cincinnatis 2023 Season!'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_question := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': user_question})
    st.chat_message("user").write(user_question)
    answer = ask_question_with_context(user_question)
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
