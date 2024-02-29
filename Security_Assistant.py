#streamlit run SecurityAssistant.py --server.port 8507
import streamlit as st
import os, re, json
#os.environ["OPENAI_API_KEY"] = ''
import glob
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import asyncio 
from threading import Thread
# from langchain_mistralai.chat_models import ChatMistralAI
# mistral_api_key = os.environ["MISTRAL_API_KEY"]


def create_chat():
    persist_directory = './chromadb/'
    embed_path  = '/embeddings/'
    
    if st.session_state.selected_model == 'mistral':
        llm = ChatMistralAI(temperature=0.0)
    else:
        llm = ChatOpenAI(model_name = st.session_state.selected_model,  temperature=0)
    
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)


    client = chromadb.PersistentClient(path= persist_directory)
    db = Chroma(
        client=client,
    collection_name = st.session_state.collection_name,
        embedding_function=embeddings)
    retriever = db.as_retriever(search_type = 'similarity', search_kwargs={'k':30})


    combine_template = '''
     Use the following pieces of context to answer the question at the end. 
    If the context does not have the answer, just say that no information has been provided. 
    {selected_answer_option}
    If you don't know the answer, just say that you don't know, don't try to make up an answer and don't return any sources.
    Answer the question in {selected_language}
    
    \n\n{context}\n\nQuestion: {question}\nHelpful Answer:
    '''
    if st.session_state.use_json:
        combine_template = 'You are assistant. Only reply with JSON. \n' + combine_template
        
    combine_docs_chain = StuffDocumentsChain(
     llm_chain=LLMChain( prompt=PromptTemplate(input_variables=['context', 'question', 'selected_language', 'selected_answer_option'], 
                                               template=combine_template), llm=llm), document_variable_name='context')

    template = '''
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:'
    '''

    question_generator  = LLMChain( prompt=PromptTemplate(input_variables=['chat_history', 'question'], 
    template=template), llm=llm)

    cqa5 = ConversationalRetrievalChain(retriever=retriever, return_source_documents=True,
                           combine_docs_chain=combine_docs_chain, question_generator=question_generator)
    
    #print (cqa5)
    return cqa5

 
st.set_page_config(layout="wide")
st.title("Security Assistant")

col1, col2 = st.columns([1,3])


if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'source_texts' not in st.session_state:
    st.session_state.source_texts = []    
if 'source_metadata' not in st.session_state:
    st.session_state.source_metadata = []    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
if 'collection_name' not in st.session_state:
    st.session_state.collection_name='dora_1000'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model='gpt-4-turbo-preview'
if 'use_json' not in st.session_state:
    st.session_state.use_json=False
if 'cqa' not in st.session_state:
    st.session_state.cqa = create_chat()
if 'selected_answer_option' not in st.session_state:
    st.session_state.selected_answer_option = 'short' 
if 'question' not in st.session_state:
    st.session_state.questioin = '' 
    
with col1:
    language_options = ['English',  'Hindi', 'Spanish', 'French', 'German',  'Russian']
    selected_language = st.selectbox('Choose the answer language:', language_options)
    #st.write(f'You selected: {selected_language}')
    model_options = ['gpt-4-turbo-preview',  'gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-16k']
    st.session_state.selected_model = st.selectbox('Choose the LLM:', model_options)
    source_options = ['DORA',  'EU AI Act']
    selected_source = st.selectbox('Choose the data source:', source_options)
    if selected_source == 'DORA':
        st.session_state.collection_name = 'dora_1000'
    elif selected_source == 'EU AI Act':
        st.session_state.collection_name = 'eu_ai_1000'
#    st.session_state.use_json = st.checkbox("Use JSON", value=False)

    answer_options = ['short',  'detailed']
    st.session_state.selected_answer_option = st.selectbox('Choose the answer option:', answer_options)
    if st.session_state.selected_answer_option == 'short':
        st.session_state.selected_answer_option = 'short, succinct, and to the point answer without bullet points. '
    elif st.session_state.selected_answer_option == 'detailed':
        st.session_state.selected_answer_option = 'detailed answer formatted as a bulletted point list. '
    st.session_state.cqa = create_chat()
    
async def async_cqa():
    result = st.session_state.cqa({'question':st.session_state.question, 'chat_history':st.session_state.chat_history, 
                                           'selected_language':selected_language, 
                                           'selected_answer_option':st.session_state.selected_answer_option})
    return result
    
def run_in_thread(loop, coroutine):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coroutine)    

def start_async_task():
    coroutine = async_cqa()
    loop = asyncio.new_event_loop()
    t = Thread(target=run_in_thread, args=(loop, coroutine))
    t.start()
    return coroutine
    
async def run_async_cqa():
    if st.session_state.user != '':
        prompt = st.session_state.user
        #st.session_state.user = ''
 
        question = prompt + f'  Answer the question in {selected_language}'
        st.session_state.question = question
        try:
            result = await async_cqa()
 
            answer = result['answer']
        except:
            result = None
            answer = 'The model failed to provide an answer :('

        if selected_language == 'English':
            st.session_state.chat_history.append((question, answer))
        else:
            st.session_state.chat_history.append((question, ''))        

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)

with col2:

    # show user input
    st.text_input("Ask a question:", key="user")
    st.button("Send", on_click=asyncio.run(run_async_cqa()))

    # col1.write(response)
    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', 
                     logo = "https://t3.ftcdn.net/jpg/05/13/20/00/360_F_513200016_YTvJGWkVAV53Bl9a5FOFNw3BLE4Pbjdd.jpg")
 
            message(st.session_state.responses[i], key=str(i), 
                    logo = "https://styles.redditmedia.com/t5_3b9u5/styles/communityIcon_d49a7viby3db1.png")
            
                
