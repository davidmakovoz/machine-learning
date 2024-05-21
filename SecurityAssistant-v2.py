#streamlit run SecurityAssistant-v2.py --server.port 8507
import streamlit as st
import os, re, json
#os.environ["OPENAI_API_KEY"] = ''
import glob
from openai import OpenAI
import chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings
import openai
import json
from streamlit_chat import message
from langchain.vectorstores import Chroma
st.set_page_config(layout="wide")
container1 = st.container()
st.divider()   
container2 = st.container()
# st.markdown("""
# <style>
# .custom-container {
#     min-height: 100px; /* Set this to your desired value */
# }
# </style>
# """, unsafe_allow_html=True)


# with container1:
#     st.markdown('<div class="custom-container">Your content here</div>', unsafe_allow_html=True)
    
selected_answer_option = '''
Answer the question from the text after the question with a detailed and nicely formatted answer. 
Only use the text provided to you here. 
If you don't know just say so. 
Make sure the answer matches the question.\n 
'''

with container2:
    col21, col22 = st.columns([2,1])
with container1:
    col11, col12 = st.columns([4,1])
with col11:
    st.title("Compliance Adviser")
    
selected_model = 'gpt-4-turbo-preview'
summarization_model = 'gpt-3.5-turbo'
output_filename_map  = {'dora_1000':r'output/dora_1000.json', 'eu_ai_1000':r'output/eu_ai_1000.json'}
if 'current_collection_name' not in st.session_state:
    st.session_state.current_collection_name=''
if 'collection_name' not in st.session_state:
    st.session_state.collection_name='dora_1000'
if 'history' not in st.session_state:
    st.session_state.history = {}
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
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)    
    if 'db_client' not in st.session_state:
        st.session_state.db_client = chromadb.PersistentClient(path= persist_directory)
    st.session_state.db = Chroma(client=st.session_state.db_client, collection_name = st.session_state.collection_name,  embedding_function=st.session_state.embeddings)
# Add elements to the first container (row)

with col12:
#         answer_options = [ 'detailed', 'short']
#         selected_answer_option = st.selectbox('Choose the answer option:', answer_options)
#         if selected_answer_option == 'short':
#             selected_answer_option = 'Answer the question from the text after the question in a brief and concise manner, really short answer, no bullet points. You have to answer something. Make sure the answer matches the question. \n '
#         if selected_answer_option == 'detailed':
#             selected_answer_option = 'Answer the question from the text after the question with a detailed and nicely formatted answer. Make sure the answer matches the question.\n '
    source_options = ['DORA',  'EU AI Act']
    selected_source = st.selectbox('Choose the data source:', source_options)
    if selected_source == 'DORA':
        st.session_state.collection_name = 'dora_1000'
    elif selected_source == 'EU AI Act':
        st.session_state.collection_name = 'eu_ai_1000'  
    if st.session_state.current_collection_name != st.session_state.collection_name:
        if len(st.session_state.prompts) > 0:
        #write the current results and read the new results
            with open(output_filename_map[st.session_state.current_collection_name], 'w') as f:
                json.dump(st.session_state.history, f, indent=4)
        #reset db, history, prompts and responses
        st.session_state.current_collection_name = st.session_state.collection_name   
        st.session_state.db = Chroma(client=st.session_state.db_client, collection_name = st.session_state.collection_name, embedding_function=st.session_state.embeddings)
        st.session_state.prompts = []
        st.session_state.responses = []
        try:
            with open(output_filename_map[st.session_state.current_collection_name], 'r') as f:
                data = json.load(f)
                print ('loaded json from ', output_filename_map[st.session_state.current_collection_name])
                print (len(data))
                for k,v in data.items():
                    print (k, ':', v['short'])
                    st.session_state.prompts.append(k)
                    st.session_state.responses.append(v['short'])
                st.session_state.history = data

        except:
            st.session_state.history = {}


    
def summarize(answer):
    try:
        messages=[
            {"role": "system", "content": "Summarize the following text with a short and brief summary."},
            {"role": "user", "content": answer}
        ]

        response = st.session_state.client.chat.completions.create(
            model=summarization_model,
            messages = messages,
            temperature=0,
            max_tokens=512,
        )
        summary = response.choices[0].message.content
    except:
        summary = 'model failed to summarized the answer :('
    return summary

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
        summary = summarize(answer)
        st.session_state.responses.append(summary)  
        st.session_state.history[prompt] = {'short':summary, 'long':answer}
        st.session_state.rerun = True
# Add elements to the second container (row)
def display_answer(prompt):
    st.session_state.user = prompt
    with col21:
        st.markdown(st.session_state.history[prompt]['long'])
with col21:
    st.text_input("", key="user")

    st.button("Ask", on_click=send_click)
#     print (st.session_state.rerun)
#     if st.session_state.rerun:
#         st.rerun() 
#         st.session_state.rerun = False    
with col22:
        # col1.write(response)
    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            st.button(st.session_state.prompts[i], on_click=lambda item=st.session_state.prompts[i]: display_answer(item))
#             message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', 
#                      logo = "https://t3.ftcdn.net/jpg/05/13/20/00/360_F_513200016_YTvJGWkVAV53Bl9a5FOFNw3BLE4Pbjdd.jpg")

            message(st.session_state.responses[i], key=str(i), 
                    logo = 'https://c8.alamy.com/comp/2EG1B4C/clipboard-summary-icon-black-outline-vector-illustration-flat-design-2EG1B4C.jpg')
#                    logo = "https://styles.redditmedia.com/t5_3b9u5/styles/communityIcon_d49a7viby3db1.png")
#         print (st.session_state.rerun)
#         if st.session_state.rerun:
#             print ('rerunning')
#             st.session_state.rerun = False 
#             st.rerun() 
            
