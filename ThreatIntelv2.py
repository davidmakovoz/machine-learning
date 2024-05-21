#streamlit run ThreatIntelv2.py --server.port 8502
import streamlit as st
import os, re, glob
from streamlit_chat import message
from datetime import datetime
import chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s?\d{1,2}(?:st|nd|rd|th),\s?\d{4}'


limit = 500000
selected_model = 'gpt-4-turbo-preview'
summarization_model = 'gpt-3.5-turbo'
st.set_page_config(layout="wide")
st.title("Threat Insights")
selected_answer_option = '''Answer the question from the text of articles after the question with a detailed and nicely formatted answer. 
Make sure the answer matches the question. 
Look for the answer from the beginning of the text provided.
Pay more attention to the first article. But do not ignore other articles, either. 
At the end of the response provide the reference to the articles in square brackets. 
For example, if first and second article were used, append [1], [2] at the end of the response.
'''


def ordinal_to_number(day):
    return ''.join(filter(str.isdigit, day))

def sort_chronologically(names, dates):

    combined = []
    for date_str, name in zip(dates, names):
        month, day, year = date_str.rsplit(' ', 2)
        day_numeric = ordinal_to_number(day)
        date_obj = datetime.strptime(f"{month} {day_numeric} {year}", "%B %d %Y")
        combined.append((date_obj, name))

    # Sort the combined list based on the dates
    combined.sort(key=lambda x: x[0], reverse=True)

    # Extract the sorted dates and names
    sorted_dates = [datetime.strftime(date_obj, "%B %dth, %Y") for date_obj, _ in combined]
    sorted_names = [name for _, name in combined]

    return sorted_names, sorted_dates

def get_cached_sources():
    folder = r'./data/therecord/'
    files = glob.glob(folder + '*.txt')
    article_names = []
    article_dates = []
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            article_name = filename.split('\\')[-1]
            article_name = article_name[:-4]
            #print (article_name)
            article_names.append(article_name)  
            text = f.read()
            matches = re.findall(date_pattern, text)
            article_dates.append((matches[0]))
            st.session_state.articles[article_name] = text
            #st.session_state.all_text += text + '\n\n'
    #print (st.session_state.all_text)
    article_names, article_dates = sort_chronologically(article_names, article_dates)
    #print (article_names, article_dates)
    return article_names
                                             
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = 'recorded_future' 
if 'summary' not in st.session_state:
    st.session_state.summary = ''
if 'current_source' not in st.session_state:
    st.session_state.current_source = ''
if 'articles' not in st.session_state:
    st.session_state.articles = {}
if 'sources' not in st.session_state:
    st.session_state.sources = get_cached_sources()
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if 'db' not in st.session_state:
    persist_directory = './chromadb_recorded_future/'
    embed_path  = '/embeddings/'
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name = 'BAAI/bge-large-en', cache_folder=embed_path)    
    if 'db_client' not in st.session_state:
        st.session_state.db_client = chromadb.PersistentClient(path= persist_directory)
    st.session_state.db = Chroma(client=st.session_state.db_client, collection_name = st.session_state.collection_name,  embedding_function=st.session_state.embeddings)
    
col1, col2, col3 = st.columns([3,2,1])


with col1:
    st.markdown("## Investigate")
    
with col2:
    st.markdown("## Summary")
    
with col3:
    st.markdown("## Sources")

    
def answer_question(question):
    print (selected_answer_option)
    try:
        messages=[
                {"role": "system", "content": selected_answer_option},
                {"role": "user", "content": question}
            ]
        documents = st.session_state.db.similarity_search_with_score(question, k = 5)
        for doc, score in documents:
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
                with col1:
                    st.markdown(line_buffer)
                line_buffer =''

            else:
                line_buffer += line
            answer += line
        with col1:
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
       

with col1:
    st.text_input("Ask a question:", key="user", label_visibility="collapsed")
    st.button("Ask", on_click=send_click)


def summarize(article_name):
    try:
        source = st.session_state.articles[article_name] 
        st.session_state.current_source = source
        messages=[
            {"role": "system", "content": "Summarize the following text with a bullet point summary."},
            {"role": "user", "content": source}
        ]

        response = st.session_state.client.chat.completions.create(
            model=summarization_model,
            messages = messages,
            temperature=0,
            max_tokens=512,
        )
        summary = response.choices[0].message.content
        st.session_state.summary = summary
    except:
        summary = 'model failed to summarized the answer :('
#     with col2:
#         st.markdown(summary)
        
with col2:
    st.markdown(st.session_state.summary)
with col3:    
    for source in st.session_state.sources:
            st.button(source, on_click=lambda item=source: summarize(item))
        
        
        