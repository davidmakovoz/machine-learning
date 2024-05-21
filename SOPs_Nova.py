#streamlit run SOPs_Nova.py --server.port 8490
import streamlit as st
import os, re, glob
import numpy as np
from streamlit_chat import message
from datetime import datetime
import chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# import base64

# def get_base64_of_image(path):
#     with open(path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode()
    
# img_to_base64 = get_base64_of_image(r'C:\Users\admin1\Documents\Jupyter\images\pexels-henry-&-co-1939485.jpg')    
score_threshold = 0.40
selected_model = 'gpt-4-turbo-preview'
summarization_model = 'gpt-3.5-turbo'
selected_language = 'English'
st.set_page_config(layout="wide")
st.title("SOC Assistant")
internal_answer_option = '''Answer the question from the text of articles after the question with a detailed and nicely formatted answer. 
Make sure the answer matches the question. 
Look for the answer from the beginning of the text provided.
Pay more attention to the first article. But do not ignore other articles, either. 
If the articles do not contain an answer to the question, just say so, do not use your pre-trained knowledge to answer it.
At the end of the response provide the reference to the articles in square brackets. 
For example, if first and second article were used, append [1], [2] at the end of the response.
'''
external_answer_option = '''Answer the question from the text of articles after the question with a detailed and nicely formatted answer. 
Make sure the answer matches the question. 
Look for the answer from the beginning of the text provided.
Pay more attention to the first article. But do not ignore other articles, either. 
If the articles do not contain an answer to the question, use your pretrained knowledge to answer the question.
At the end of the response provide the reference to the articles in square brackets. 
For example, if first and second article were used, append [1], [2] at the end of the response.
'''
if 'number_supporting_documents' not in st.session_state:
    st.session_state.number_supporting_documents = 3
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = 'sops_nova_1000_20' 
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if 'db' not in st.session_state:
    persist_directory = './chromadb_sops_nova/'
    embed_path  = '/embeddings/'
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)    
    if 'db_client' not in st.session_state:
        st.session_state.db_client = chromadb.PersistentClient(path= persist_directory)
    st.session_state.db = Chroma(client=st.session_state.db_client, collection_name = st.session_state.collection_name,  embedding_function=st.session_state.embeddings)
    
# Custom CSS to set the background color
css = """
<style>
    .stApp {
        background-color: #D7BDE2;
    }
</style>
"""
# css = f"""
# <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{img_to_base64}");
#         background-size: cover;
#         background-position: center;
#     }}
# </style>
# """
# Injecting the CSS with st.markdown
st.markdown(css, unsafe_allow_html=True)
col1, col2, col3  = st.columns([1,3, 2])

with col1:
    st.markdown("## Select")

    answer_options = ['contextual',  'contextual and pre-trained']
    selected_option = st.selectbox('Choose the source for the answer:', answer_options)
    if selected_option == 'contextual':
        selected_answer_option = internal_answer_option
    elif selected_option == 'contextual and pre-trained':
        selected_answer_option = external_answer_option
    model_options = ['gpt-4-turbo-preview',  'gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-16k']
    selected_model = st.selectbox('Choose the LLM:', model_options)    
    collection_options =  ['sops_nova_1000_20','sops_nova_10000_200']
    collection_name = st.selectbox('Choose the collection:', collection_options)   
    if st.session_state.collection_name != collection_name:
        st.session_state.collection_name = collection_name
        st.session_state.db = Chroma(client=st.session_state.db_client, collection_name = st.session_state.collection_name,  embedding_function=st.session_state.embeddings)
    
    number_supporting_documents = [int(x) for x in np.linspace(1, 30, 28)]
    st.session_state.number_supporting_documents = st.selectbox('Choose the number of supporting sections:', 
                                                                number_supporting_documents, index = 2)    
    
with col2:
    st.markdown("## Investigate")
    
def answer_question(question):
    print (selected_answer_option)
    try:
        messages=[
                {"role": "system", "content": selected_answer_option},
                {"role": "user", "content": question}
            ]
        documents = st.session_state.db.similarity_search_with_score(question, k = st.session_state.number_supporting_documents)
        number = 0
        for doc, score in documents:
            number += 1
            if score < 0.4:
                with col3:
                    with st.expander("Section " + str(number)):
                        # Put whatever you want inside the expander
                        st.markdown(doc.page_content)
                    

            print (doc.metadata['source'], score)
            messages.append({"role": "user", "content": f"The following is a document for reference: {doc.page_content}"})            
            
        response = st.session_state.client.chat.completions.create(
            model=selected_model,
            messages = messages,
            stream=True,
            temperature=0,
            max_tokens=2024,
        )
        answer =''
        line_buffer =''

        for chunk in response:
            line = chunk.choices[0].delta.content or ""
            #print(line, end="")
#             if '.' in line and  not re.match(r'\d\.', line):
            if '\n' in line:
                line_buffer += line
                with col2:
                    st.markdown(line_buffer)
                line_buffer =''

            else:
                line_buffer += line
            answer += line
        with col2:
            st.markdown(line_buffer)       
    except:
        answer = 'The model failed to provide an answer :('

    return answer

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        #print (prompt)
        #st.session_state.user = ''

        question = prompt# + f'  Answer the question in {selected_language}'
        try:
            answer = answer_question(question)
        except:
            answer = 'The model failed to provide an answer :(:('

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)  


with col2:
    st.text_input("Ask a question:", key="user", label_visibility="collapsed")
    st.button("Ask", on_click=send_click)

#     if st.session_state.prompts:
#         for i in range(len(st.session_state.responses)-1, -1, -1):
#             message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', 
#                      logo = "https://t3.ftcdn.net/jpg/05/13/20/00/360_F_513200016_YTvJGWkVAV53Bl9a5FOFNw3BLE4Pbjdd.jpg")

#             message(st.session_state.responses[i], key=str(i), 
#                     logo = "https://styles.redditmedia.com/t5_3b9u5/styles/communityIcon_d49a7viby3db1.png")