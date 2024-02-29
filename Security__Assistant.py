#streamlit run Security__Assistant.py --server.port 8507
import streamlit as st
import os, re, json
#os.environ["OPENAI_API_KEY"] = ''
import glob
import chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings
import openai
from openai import OpenAI
from streamlit_chat import message
from langchain.vectorstores import Chroma
st.set_page_config(layout="wide")


container1 = st.container()
container2 = st.container()

with container2:
    col21, col22 = st.columns([2,3])
selected_model = 'gpt-4-turbo-preview'

# Add elements to the first container (row)
with container1:
    col1, col2 = st.columns([4,1])
    with col1:
        st.title("Security Assistant")
    with col2:
        answer_options = ['short',  'detailed']
        selected_answer_option = st.selectbox('Choose the answer option:', answer_options)
        if selected_answer_option == 'short':
            selected_answer_option = 'Answer the question from the text after the question in a brief and concise manner, really short answer, no bullet points. You have to answer something. Make sure the answer matches the question. \n '
        if selected_answer_option == 'detailed':
            selected_answer_option = 'Answer the question from the text after the question with a detailed and nicely formatted answer. Make sure the answer matches the question.\n '
        
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'rerun' not in st.session_state:
    st.session_state.rerun = False
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if 'db' not in st.session_state:
    persist_directory = './chromadb/'
    embed_path  = '/embeddings/'
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)
    collection_name = 'dora_1000'

    db_client = chromadb.PersistentClient(path= persist_directory)
    st.session_state.db = Chroma(client=db_client, collection_name = collection_name,  embedding_function=embeddings)
    

def answer_question(question):
    print (selected_answer_option)
    try:
        messages=[
                {"role": "system", "content": selected_answer_option},
                {"role": "user", "content": question}
            ]
        documents = st.session_state.db.similarity_search(question, k = 30)
        for doc in documents:
            messages.append({"role": "system", "content": f"The following is a document for reference: {doc.page_content}"})
        print (question)
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
                with col21:
                    st.markdown(line_buffer)
                line_buffer =''

            else:
                line_buffer += line
            answer += line
        with col21:
            st.markdown(line_buffer)       
    except:
        answer = 'The model failed to provide an answer :('

    return answer


def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        print (prompt)
        #st.session_state.user = ''

        question = prompt# + f'  Answer the question in {selected_language}'
        try:
            answer = answer_question(question)
        except:
            answer = 'The model failed to provide an answer :(:('
#         if selected_language == 'English':
#             st.session_state.chat_history.append((prompt, answer))
#         else:
#             st.session_state.chat_history.append((prompt, ''))                    
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)    
        st.session_state.rerun = True
# Add elements to the second container (row)

with col21:
    st.text_input("Ask a question:", key="user")

    st.button("Ask", on_click=send_click)
#     print (st.session_state.rerun)
#     if st.session_state.rerun:
#         st.rerun() 
#         st.session_state.rerun = False    
with col22:
        # col1.write(response)
    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', 
                     logo = "https://t3.ftcdn.net/jpg/05/13/20/00/360_F_513200016_YTvJGWkVAV53Bl9a5FOFNw3BLE4Pbjdd.jpg")

            message(st.session_state.responses[i], key=str(i), 
                    logo = "https://styles.redditmedia.com/t5_3b9u5/styles/communityIcon_d49a7viby3db1.png")
        print (st.session_state.rerun)
        if st.session_state.rerun:
            print ('rerunning')
            st.session_state.rerun = False 
            st.rerun() 
            
