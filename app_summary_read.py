#streamlit run app_summary_read.py --server.port 8530
import streamlit as st
import os, re
#os.environ["OPENAI_API_KEY"] = ''
import asyncio
import openai
import threading
import PyPDF2
from io import BytesIO

limit = 500000
selected_model = 'gpt-4-turbo-preview'
st.set_page_config(layout="wide")
st.title("Cyber Assistant")

col1, col2 = st.columns([1,4])
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False   
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = 1
if 'prompt' not in st.session_state:
    st.session_state.prompt = None
    
def read_pdf_mine(uploaded_file):
    all_text = ""
    with BytesIO(uploaded_file.getvalue()) as opened_pdf_file:
        pdf_reader = PyPDF2.PdfReader(opened_pdf_file) 
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            print (len(text))

    st.session_state.prompt = all_text

def read_pdf(uploaded_file):
    all_text = ""
    with BytesIO(uploaded_file.getvalue()) as opened_pdf_file:
        pdf_reader = PyPDF2.PdfReader(opened_pdf_file) 
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            #print (len(text))
            if len(text) > 1000:
                all_text += text + "\n" 
            if len(all_text) > limit:
                break
    return all_text

with col1:
    language_options = ['English',  'Hindi', 'Spanish', 'French', 'German',  'Russian']
    selected_language = st.selectbox('Choose the answer language:', language_options)
    model_options = ['gpt-4-turbo-preview',  'gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-16k']
    selected_model = st.selectbox('Choose the LLM:', model_options)    
    #st.write(f'You selected: {selected_language}')
    summary_options = ['short',  'detailed']
    selected_summary_option = st.selectbox('Choose the summary option:', summary_options)
    if selected_summary_option == 'short':
        selected_summary_option = 'short no more than 1 bullet point per page'
    if selected_summary_option == 'detailed':
        selected_summary_option = 'detailed nicely formatted at least 1 bullet point per page '
        
    if st.button("Upload a new File"):
        # Create a file uploader widget
        print (st.session_state.file_uploaded, st.session_state.uploaded_file == None)
        st.session_state.uploaded_file = None
        st.session_state.prompt = ''
    if st.session_state.uploaded_file == None:
        st.session_state.uploaded_file = st.file_uploader("Choose a file")  
        print (st.session_state.uploaded_file == None)
        if st.session_state.uploaded_file is not None:
            st.session_state.prompt = read_pdf(st.session_state.uploaded_file)
            print (len(st.session_state.prompt))
    if st.session_state.uploaded_file is not None and st.session_state.uploaded_file != 1:
        st.write("Filename:", st.session_state.uploaded_file.name)
# Define the async function
async def get_chat_response():
    prompt = st.session_state.prompt
    async with openai.AsyncOpenAI() as api:
        async for streamed_response in await api.chat.completions.create( 
           model=selected_model,
            messages=[
                {"role": "system", "content": f" You give a {selected_summary_option} summary of text with a bulletted list in {selected_language}. "},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2048,
            stream=True
        ):
            streaming_data = streamed_response.choices[0]
            try:
                for token in streaming_data.delta.content:
                    yield token  # Yield tokens for Streamlit
            except:
                yield 'the end'
                

# Streamlit app interface
with col2:
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):  # Display a spinner
            async def run_summarization():
                line_buffer = ""  # Initialize a buffer for storing characters to build up lines
                async for line in get_chat_response(): 

                    if line == 'the end':
                        print (line)
                        st.write(line_buffer)
                    elif line != '\n':  
                        line_buffer += line  # Add character to buffer
                        #print (line)
                    else:
                        st.write(line_buffer)
                        line_buffer = ""

            asyncio.run(run_summarization()) 