#streamlit run app_summary.py --server.port 8520
import streamlit as st
import os, re
#os.environ["OPENAI_API_KEY"] = ''
import asyncio
import openai
import threading
from langchain.document_loaders import PyPDFLoader

st.set_page_config(layout="wide")
st.title("Cyber Assistant")

col1, col2 = st.columns([1,4])


with col1:
    language_options = ['English',  'Hindi', 'Spanish', 'French', 'German',  'Russian']
    selected_language = st.selectbox('Choose the answer language:', language_options)
    #st.write(f'You selected: {selected_language}')
    summary_options = ['short',  'detailed']
    selected_summary_options = st.selectbox('Choose the summary option:', summary_options)
    document_options = ['DORA',  'EU AI Act']
    document_to_summarize = st.selectbox('Choose document to summarize:', document_options)
  
                
# Define the async function
async def get_chat_response(prompt):
    async with openai.AsyncOpenAI() as api:
        async for streamed_response in await api.chat.completions.create( 
           model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": f" You give a {selected_summary_options} summary of text with a bulletted list in {selected_language}."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            stream=True
        ):
            streaming_data = streamed_response.choices[0]
            try:
                for token in streaming_data.delta.content:
                    yield token  # Yield tokens for Streamlit
            except:
                yield 'the end'
                
#             line_buffer = ""  # Initialize a buffer for storing characters to build up lines
#             for char in streaming_data.delta.content:
#                 if char != '\n':  
#                     line_buffer += char  # Add character to buffer
#                     print (line_buffer)
#                 else:
#                     yield line_buffer + '\n'  # Yield the complete line when a newline     
if document_to_summarize == 'DORA':
    pdf = r'.\data\DORA.pdf'
    with open(r'.\data\DORA.txt', 'r') as f:
        prompt = f.read()
elif document_to_summarize == 'EU AI Act':
    pdf = r'.\data\EU-AI-Act.pdf'
    with open(r'.\data\EU-AI-Act.txt', 'r') as f:
        prompt = f.read()
# loader = PyPDFLoader(pdf)
# pages = loader.load_and_split()
# if document_to_summarize == 'EU AI Act':
#     pages = pages[10:162]
# page_contents = [x.page_content for x in pages]
# prompt = ' '.join(page_contents)
# print (len(prompt))

# Streamlit app interface
with col2:
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):  # Display a spinner
            async def run_summarization():
                line_buffer = ""  # Initialize a buffer for storing characters to build up lines
                async for line in get_chat_response(prompt): 

                    if line != '\n':  
                        line_buffer += line  # Add character to buffer
                        print (line_buffer)
                    else:
                        st.write(line_buffer)
                        line_buffer = ""

            asyncio.run(run_summarization()) 