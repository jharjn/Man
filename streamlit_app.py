import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os



embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local("embedding_data",embeddings,allow_dangerous_deserialization=True)

retriever = db.as_retriever()
llm=ChatGroq(groq_api_key="")


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
    llm, retriever, contextualize_q_prompt
)

system_prompt = (

 """You are a mental health symptom tracking chatbot. Using the information provided, converse with the user.
  Do not talk with the user about anything unrelated to mental health. As you converse with them, collect symptoms, and once you have enough, return a list for a doctor to use in a diagnosis
  if the user at any time says something along the lines of "Thank you, im done", then end early and diplay the list of symptoms, and a mental health issue they might have
Use three sentences maximum and keep the answer concise.\
Case Studies:
{context}"""



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

if 'store' not in st.session_state:
    st.session_state.store={}

def get_session_history(session:str)->BaseChatMessageHistory:
      if session_id not in st.session_state.store:
          st.session_state.store[session_id]=ChatMessageHistory()
      return st.session_state.store[session_id]
      
conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )














st.set_page_config(page_title="Symptom-chatbot")
with st.sidebar:
    st.title('Hi there! I am mental health symptom analyzing chatbot!')
# Function for generating LLM response
def conv_past(inp):
    ret = []
    for num,comb in enumerate(inp):
        ret.append(f"Message {num%2} by the {comb['role']}: {comb['content']}\n")
    return ret

def afterRes(input_string):
    # Find the index of "Question:"
    question_index = input_string.find("Question:")
    if question_index == -1:
        return input_string
        #return "No 'Question:' found in the input string"
    # Find the index of "Answer:"
    answer_index = input_string.find("Answer:", question_index)
    if answer_index == -1:
        return input_string
        #return "No 'Answer:' found in the input string"
    # Extract the text after "Answer:"
    text_after_answer = input_string[answer_index + len("Answer:"):].strip()
    return text_after_answer
        
        
    #return response
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi there!"}]
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
         st.write(input)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating.."):
            response = conversational_rag_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}},
              )["answer"] 
            response = afterRes(response)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
