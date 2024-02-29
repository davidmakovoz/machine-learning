#streamlit run Playbook_Guide.py --server.port 8541
import streamlit as st
import os, re
from streamlit_chat import message
limit = 500000
selected_model = 'gpt-4-turbo-preview'
st.set_page_config(layout="wide")
st.title("Playbook Guide")
rerun = False

col1, col2 = st.columns([1,4])

if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
if 'text' not in st.session_state:
    with open(r'.\data\Ransomeware_Playbook_Guidance.txt', 'r', encoding='utf-8') as f:
        st.session_state.text = f.read()
if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with col1:
    language_options = ['English',  'Hindi', 'Spanish', 'French', 'German',  'Russian']
    selected_language = st.selectbox('Choose the answer language:', language_options)
    model_options = ['gpt-4-turbo-preview',  'gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-16k']
    selected_model = st.selectbox('Choose the LLM:', model_options)    
    #st.write(f'You selected: {selected_language}')
    answer_options = ['short',  'detailed']
    selected_answer_option = st.selectbox('Choose the answer option:', answer_options)
    if selected_answer_option == 'short':
        selected_answer_option = 'Answer the question from the text after the question in a brief and concise manner, really short answer, no bullet points. You have to answer something. Make sure the answer matches the question. \n '
    if selected_answer_option == 'detailed':
        selected_answer_option = 'Answer the question from the text after the question with a detailed and nicely formatted answer. Make sure the answer matches the question.\n '

def answer_question(question):
    print (selected_answer_option, selected_model)
    try:

        print (question)
        response = st.session_state.client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": selected_answer_option},
                {"role": "user", "content": question},
                {"role": "user", "content": st.session_state.text}
            ],
            stream=True,
            temperature=0,
            max_tokens=2024,
        )
        answer =''
        line_buffer =''

        for chunk in response:
            line = chunk.choices[0].delta.content or ""
            print(line, end="")

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
#         with col2:
#             st.markdown(prompt)
        #st.session_state.user = ''

        question = prompt + f'  Answer the question in {selected_language}'
        try:
            answer = answer_question(question)
        except:
            answer = 'The model failed to provide an answer :(:('
        if selected_language == 'English':
            st.session_state.chat_history.append((prompt, answer))
        else:
            st.session_state.chat_history.append((prompt, ''))                    
        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(answer)    
        #rerun = True
        
with col2:

    # show user input
    st.text_input("Ask a question:", key="user")
   
    st.button("Send", on_click=send_click)
        
    if rerun:
        st.rerun() 
        rerun = False    

