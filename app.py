#streamlit run app.py --server.port 8505
import streamlit as st
import os, re
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

collection_name = 'crowdstrike_cortex_1000'
def create_chat(collection_name):
    persist_directory = './chromadb/'
    embed_path  = '/embeddings/'
    
    llm = ChatOpenAI(model_name = 'gpt-4-1106-preview', 
                 temperature=0)
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)


    client = chromadb.PersistentClient(path= persist_directory)
    db = Chroma(
        client=client,
    collection_name = collection_name,
        embedding_function=embeddings)
    retriever = db.as_retriever(search_type = 'similarity', search_kwargs={'k':20})


    combine_template = '''
     Use the following pieces of context to answer the question at the end. 
    If the context does not have the answer, just say that no information has been provided. 
    If it does try to provide a detailed answer with at least 3 sentences. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer and don't return any sources.
    If you do  know the answer return  the answer and the references to the sources used by using the following notation: 
    If the answer was found in the first piece of context, return the answer and "[1]", 
    If the answer was found in the second piece of context, return the answer and "[2]", 
    If the answer was found in the first and second piece of context, return the answer and "[1], [2]"
    If the question is to compare two things, please compare Cortex and Crowdstrike and answer which is one is better.
    Answer the question in {selected_language}
    
    \n\n{context}\n\nQuestion: {question}\nHelpful Answer:
    '''

    combine_docs_chain = StuffDocumentsChain(
     llm_chain=LLMChain( prompt=PromptTemplate(input_variables=['context', 'question', 'selected_language'], template=combine_template), 
    llm=llm), document_variable_name='context')

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


def return_sources(result):
    for r in result['source_documents']:
        print (len(r.page_content))
        
    source_texts = source_metadata = []
    try:
        answer = result['answer']
        source_index = re.findall(r'\[(\d+)\]', answer)

        source_texts = []
        source_metadata = []
        for index in source_index:
            r = result['source_documents'][int(index)-1]
            source_texts.append(r.page_content)
            source_metadata.append(r.metadata)
    except:
        pass
    return source_texts, source_metadata
    

def answer_question(question):
    try:
        result = st.session_state.cqa({'question':question, 'chat_history':st.session_state.chat_history, 
                                       'selected_language':selected_language})
        answer = re.sub(r'\[(\d+)\]', '',result['answer'])   
    except:
        result = None
        answer = 'The model failed to provide an answer :('
#    if (len(answer))< 3 or (', , , , , , , , , , ' in answer): answer = 'I do not know'
    
    if selected_language == 'English':
        st.session_state.chat_history.append((question, answer))
    else:
        st.session_state.chat_history.append((question, ''))        
    return result
 
st.set_page_config(layout="wide")
st.title("Cyber Assistant")

col1, col2 = st.columns([1,1])


with col1:
    language_options = ['English',  'Hindi', 'Spanish', 'French', 'German',  'Russian']
    selected_language = st.selectbox('Choose the answer language:', language_options), 
    #st.write(f'You selected: {selected_language}')


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
if 'cqa' not in st.session_state:
    st.session_state.cqa = create_chat(collection_name)
    

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        st.session_state.user = ''
        source_texts = []
        source_metadata = []
        st.session_state.source_texts = []
        st.session_state.source_metadata = []   
        question = prompt + f'  Answer the question in {selected_language}'
        try:
            response = answer_question(question)
            answer = re.sub(r'\[(\d+)\]', '',response['answer'])  
#             if (len(answer))< 3 or (', , , , , , , , , , ' in answer): 
#                 answer = 'I do not know'
#             else:
#                 source_texts, source_metadata = return_sources(response)
            source_texts, source_metadata = return_sources(response)
        except:
            answer = 'The model failed to provide an answer :('

        for text in source_texts:
            st.session_state.source_texts.append(text)
        for text in source_metadata:
            st.session_state.source_metadata.append(text)    
            
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)

with col1:
    #st.markdown('**length of chat history** = ' + str(len(st.session_state.chat_history)))
    for i, text in enumerate(st.session_state.source_texts):
        #print('in col1',text)
        #print(st.session_state.source_metadata[i]['source'])
        st.markdown('### '+st.session_state.source_metadata[i]['source'].split('\\')[-1])
        st.markdown('_'+text+'_')
        st.text('________________________')
    st.session_state.source_texts = []

with col2:

    # show user input
    st.text_input("Ask a question:", key="user")
    st.button("Send", on_click=send_click)

    # col1.write(response)
    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=100)
            message(st.session_state.responses[i], key=str(i), seed=42)
