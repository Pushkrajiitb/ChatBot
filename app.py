import os
from langchain import PromptTemplate, LLMChain
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from streamlit_chat import message

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.title('Campus Connect V1')

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  
#repo_id="facebook/blenderbot-400M-distill"
model = HuggingFaceEndpoint(  
    repo_id=repo_id,  
    max_length=128,  
    temperature=0.5,  
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  
)  
# Initialize the session state to store chat history
if 'history' not in st.session_state:
    st.session_state.history = []
# Define the prompt
# prompt = PromptTemplate(  
#     input_variables=["question"],  
#     template="As a helpful assistant you have to give the answer to user query {question}")



user_input = st.chat_input("You: ")
chat_history = st.session_state.get('chat_history', [])
#print(llm_chain.invoke({"question": question}))
# Display the chatbot response
# Process user input and generate response
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"You are a helpful assistant. Answer all questions to the best of your ability.user input is {user_input}",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the LLMChain
chain = prompt | model
if user_input:
    chat_history.append({'role': 'user', 'content': user_input})
    st.session_state.chat_history = chat_history
    # Display user's message
    #message(user_input, is_user=True, key=f"user_{len(chat_history)}")

    with st.spinner('Generating response..'):
        response = chain.invoke({"messages": [HumanMessage(content=user_input)]})
    
        bot_reply = response
        # Add assistant's message to chat history
        chat_history.append({'role': 'assistant', 'content': bot_reply})
        st.session_state.chat_history = chat_history
        # Display assistant's message
        #message(bot_reply,is_user=False,key=f"assistant_{len(chat_history)}")

# Display previous messages
for idx, message_data in enumerate(chat_history):
    role, content = message_data['role'], message_data['content']
    if role == 'user':
        message(content, is_user=True, key=f"user_{idx}_{len(chat_history)}")
    elif role == 'assistant':
        message(content, is_user=False, key=f"assistant_{idx}_{len(chat_history)}")