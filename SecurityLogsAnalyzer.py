#streamlit run SecurityLogsAnalyzer.py --server.port 8530
#What other logs can I use to investigate this attack? 
import streamlit as st
import os, re
import openai

limit = 500000
selected_model = 'gpt-4-turbo-preview'
#selected_model = 'gpt-4o'
st.set_page_config(layout="wide")
st.title("Security Logs Analyzer")
# Custom CSS to set the background color
css = """
<style>
    .stApp {
        background-color: #D7BDE2;
    }
</style>
"""

# Injecting the CSS with st.markdown
st.markdown(css, unsafe_allow_html=True)
col1, col2 = st.columns([1,7])
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False   
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = 1
if 'log_text' not in st.session_state:
    st.session_state.log_text = ''
if 'attacks' not in st.session_state:
    st.session_state.attacks = ''
if 'summary' not in st.session_state:
    st.session_state.summary = ''
if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
    
selected_answer_option = '''Answer the question from the log records after the question with a detailed and nicely formatted answer. 
Make sure the answer matches the question. 
If the question is about additional logs to investigate attacks, do not offer sources of logs already present. 
Only suggest additional logs if you are sure they can help, do not offer something that might help.
Also, asked for additional logs, just give their names, do not give any details about events, unless asked explicitly.
If asked about attacks, use the result of the previous question about idenitified attacks, do not do your own analysis.
If asked for mitigations, only provide the essential, abolutely necessary mitigations that you are sure about, do not offer something that is only potentially useful.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''
attack_system_message = '''You are a cybersecurity expert and also an expert in cyber attack classification. 
Your job is to read the log records provided.
I need a short verdict,  if there is a cyber attack in the records give  me it's name.
I do not need any speculacations. If there are clear IOC just say so, if not just say there is no attack detected.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''
description_system_message = '''You are a cybersecurity expert and also an expert in summarization. 
Your job is to read the log records provided.
I need a short summary in humanly readable formate of the records provided to you.
List all the sources of the logs, e.g. Windows logs, firewall logs, etc. Provide some overall information about the data in those sources.
Do not start with the words Summary of Log Records. 
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''
def read_log(uploaded_file):
    file_content = str(uploaded_file.read())
    return file_content

def produce_output(document_text, system_message):
    response = st.session_state.client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f'The following is the content of the logs:\n {document_text}'}
        ],
        stream=False,
        temperature=0,
        max_tokens=2024,
    )

    answer = response.choices[0].message.content
    return answer  

def answer_question(question, document_text):
    #print (selected_answer_option)
    try:
        messages  = list(st.session_state.chat_history)
        messages.append({"role": "system", "content": selected_answer_option})
        messages.append({"role": "user", "content": question})
 
        messages.append({"role": "user", "content": f'The following is the content of the logs:\n {document_text}'})            
        messages.append({'role':'user', 'content': 'What attacks do you detect in the logs given to you?'})
        messages.append({'role':'assistant', 'content': st.session_state.attacks})
            
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

with col1:
    model_options = ['gpt-4o', 'gpt-4-turbo-preview', 'gpt-4-1106-preview']
    selected_model = st.selectbox('Choose the LLM:', model_options)    
    if st.button("Upload a new File"):
        # Create a file uploader widget
        #print (st.session_state.file_processed, st.session_state.uploaded_file == None)
        st.session_state.uploaded_file = None
        st.session_state.log_text = ''
    if st.session_state.uploaded_file == None:
        st.session_state.uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx"])  
        #print (st.session_state.uploaded_file == None)
        if st.session_state.uploaded_file is not None:
            st.session_state.log_text = read_log(st.session_state.uploaded_file)
            st.session_state.file_processed = False
            #print (len(st.session_state.log_text))
    if st.session_state.uploaded_file is not None and st.session_state.uploaded_file != 1:
        st.write("Filename:", st.session_state.uploaded_file.name)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

with col2:
    if (st.session_state.log_text != '') :
        with st.expander("**Raw Log**"):
            st.markdown(st.session_state.log_text)

        with st.expander("**Attack(s) Detected**"):
            if (not st.session_state.file_processed):
                st.session_state.attacks = produce_output(st.session_state.log_text, attack_system_message)
                st.session_state.chat_history.append({'role':'user', 'content': 'What attacks do you detect in the logs given to you?'})
                st.session_state.chat_history.append({'role':'assistant', 'content': st.session_state.attacks})
            st.markdown(st.session_state.attacks)
        with st.expander("**Log Summary**"):
            if (not st.session_state.file_processed):
                st.session_state.summary = produce_output(st.session_state.log_text, description_system_message)
            st.markdown(st.session_state.summary)
        if (not st.session_state.file_processed):
            st.session_state.file_processed = True
            st.rerun()

        
def send_click():
    
    print ('Received a click request: ', st.session_state.user)
    if st.session_state.user != '':
        prompt = st.session_state.user
        #print (prompt)
        #st.session_state.user = ''

        question = prompt# + f'  Answer the question in {selected_language}'
        try:
            answer = answer_question(question, st.session_state.log_text)
        except:
            answer = 'The model failed to provide an answer :(:('

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)  
        st.session_state.chat_history.append({'role':'user', 'content': prompt})
        st.session_state.chat_history.append({'role':'assistant', 'content': answer})

with col2:
    if (st.session_state.log_text != ''):
        st.markdown("##### Investigate")
        st.text_input("Ask a question:", key="user", label_visibility="collapsed")
        st.button("Ask", on_click=send_click)
        print ('Received a request: ', st.session_state.user)
