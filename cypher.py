#streamlit run cypher.py --server.port 8506
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from streamlit_chat import message

def create_cypher_agent(direct=False):
    password = os.getenv('NEO4J_PASSWORD')
    graph = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password=password)

    if direct:
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, model="gpt-4"), graph=graph, 
            verbose=False,
            return_direct=True,
            top_k = 100
        )
    else:
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, model="gpt-4"), graph=graph, 
            verbose=False, 
            top_k = 100
        )
    return chain
st.set_page_config(layout="centered")
st.title("Cyber Assistant")

if 'cypher_agent' not in st.session_state:
    st.session_state.cypher_agent = create_cypher_agent()
if 'cypher_agent_direct' not in st.session_state:
    st.session_state.cypher_agent_direct = create_cypher_agent(direct=True)
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []    

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user

        try:
           
            answer = st.session_state.cypher_agent.run(prompt)
            if "I'm sorry" in answer:
                output = st.session_state.cypher_agent_direct.run(prompt)
                print (output)
                answer = ''
                for o in output:
                    answer += list(o.values())[0]
                    answer += ', '

        except:
            answer = 'The model failed to provide an answer :('

            
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)    
    
st.text_input("Ask a question:", key="user")
st.button("Send", on_click=send_click)

# col1.write(response)
if st.session_state.prompts:
    for i in range(len(st.session_state.responses)-1, -1, -1):
        message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=100)
        message(st.session_state.responses[i], key=str(i), seed=42)