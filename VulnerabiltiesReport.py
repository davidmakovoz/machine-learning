#streamlit run VulnerabiltiesReport.py --server.port 8580
#https://www.cisa.gov/known-exploited-vulnerabilities-catalog
#https://www.fortiguard.com/psirt
import streamlit as st
import os, re
import openai
from langchain.document_loaders import WebBaseLoader
from langchain.document_transformers import Html2TextTransformer
html2text = Html2TextTransformer()

st.set_page_config(layout="wide")
st.title("Vulnerability Report")
# Custom CSS to set the background color
css = """
<style>
    .stApp {
        background-color: #D7BDE2;
    }
</style>
"""
gpt_4turbo = 'gpt-4-turbo-preview'
gpt_4o = 'gpt-4o'

# Injecting the CSS with st.markdown
st.markdown(css, unsafe_allow_html=True)
col1, col2 = st.columns([1,10])

if 'expander_open' not in st.session_state:
    st.session_state.expander_open = True
if 'affected_products' not in st.session_state:
    st.session_state.affected_products  = ''
if 'description' not in st.session_state:
    st.session_state.description  = ''
if 'cve_details' not in st.session_state:
    st.session_state.cve_details  = ''
if 'title' not in st.session_state:
    st.session_state.title  = ''
if 'doc_text' not in st.session_state:
    st.session_state.doc_text  = ''


if 'client' not in st.session_state:
    from openai import OpenAI
    st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cve_pattern = r"CVE-\d{4}-\d+"
def find_cve(text):
    cve_list = re.findall(cve_pattern, text)
    return cve_list

system_message_for_descriptions = '''You are a cybersecurity expert and also an expert in summarization. 
Your job is to read the text provided.
The text contains a description of one or several vulnerabilities.
You are to create a short and crips summary of the text.
The summary should not be longer than 2 sentences.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_impact = '''You are a cybersecurity expert and also an expert in identifying impact of exploited cyber vunerabilities. 
Your job is to read the text provided.
The text contains a description of one or several vulnerabilities.
You are to identify what impact these vulnerabilities will have.
You are to output the description of the impact.
The description should be a short and crisp text containing no more than one sentences.
Do not start with introductions, for example do not say "This vulnerebility will have a great impact" or anything similar.
Do not speculate about some potential impacts that may or may not happen. Only report on something that is fairly certain.
Do not, I repeat, do not describe the vulnerability itself. It was already done. Just report the impact.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_mitigations = '''You are a cybersecurity expert and also an expert in identifying mitigations of exploited cyber vunerabilities. 
Your job is to read the text provided.
The text contains a description of one or several vulnerabilities.
You are to identify what mitigations can be used agaisnt these vulnerabilities.
The mitigations should be based on the text of the article. Do not use your pre-trained knowledge to identify mitigations.
You are to output the description of the mitigations.
If mutliple mitigations are identified, they should be output as a bulleted list. 
The description of each mitigataion should be a short and crisp text containing no more than two sentences.
Do not explain the reasons, do not start with a preamble, just give me a list.
If no mitigations are identified, just output "no mitigations have been identified".
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_tags = '''You are a cybersecurity expert and also an expert in tagging article. 
Your job is to give 10 sharp and crisp tags to the text provided.
Tags shouldn't be longer than four words, unless it is CVE code, in which case you give it all.
Just output them on one line separated by semicolon.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_attack_scenarios = '''You are a cybersecurity expert and also an expert in attack scenarios. 
Your job is to read the text provided.
The text contains a description of one or several vulnerabilities.
You are to identify the attack scenarios in the description.
You are to output all attack scenarios in one sentence.
Each sentence should start with the words "An attacker can exploit this vulnerability(ies)"
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_affected_products = '''You are a cybersecurity expert and also an expert in identifying affected products. 
Your job is to read the text provided.
The text contains a description of one or several vulnerabilities.
You are to identify the products affected by the vulnerabities.
Affected products could be software, firmware, hardware, or any other computer systems.
If any specific versoins of the product are mentioned, you are to identify those versions.
Make sure to differentiate between affected versions and versions that have been patched.
If a version is patched it means it is not affected, do not output that version. 
Most likely all previous versions before the patched one are affected.
However, whenever possible, you really need to work hard to find the earliest version affected and output it.
You are to output all affected products as a list, one product per line including the relevant product versions. 
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''

system_message_for_grouping_affected_products = '''You are an expert in summarizing text. I need to do an excellent job, because my promotion depends on it. 
I'll give you a list of products. If the list contains the same product of different versions you are to convert the list into one list with the name of the product and the versions listed as a comma separated list. 
For example, instead of 
- Google Chrome for Windows versions before 124.0.6367.118/.119
- Google Chrome for Mac versions before 124.0.6367.118/.119
- Google Chrome for Linux versions before 124.0.6367.118
output
Google Chrome versions prior to 124.0.6367.118/.119 for Windows, Mac, and versions prior to 124.0.6367.118 for Linux.

Just return the product name and versions, do not return any explanations, introductions, or conclusions.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''
def group_affected_products(product_list, selected_model):
    response = st.session_state.client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message_for_grouping_affected_products},
            {"role": "user", "content": f'The following is the list of products:\n {product_list}'}
        ],
        stream=False,
        temperature=0,
        max_tokens=2024,
    )

    answer = response.choices[0].message.content
    return answer

system_message_for_tag_verification = '''You are a cybersecurity expert and also an expert in tagging article. 
Your job is to verify that the tags provided to you match the text provided.
Tags shouldn't be longer than four words, unless it is CVE code, in which case you give it all.
Just output the tags that match the text. Do not output the tags that do not match the text. 
Do not output any explanations.
The quality of your answers is very important to me. If they are not good, I'll lose my job, and my kids will go hungry:(
'''
def tag_verification(document_text, tags, selected_model):
    response = st.session_state.client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message_for_tag_verification},
            {"role": "user", "content": f'The following is the list of tags:\n {tags}'},
            {"role": "user", "content": f'The following is the text of the article:\n {document_text}'}
        ],
        stream=False,
        temperature=0,
        max_tokens=2024,
    )

    answer = response.choices[0].message.content
    return answer

def produce_output(document_text, system_message, selected_model):
    response = st.session_state.client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f'The following is the text of the article:\n {document_text}'}
        ],
        stream=False,
        temperature=0,
        max_tokens=2024,
    )

    answer = response.choices[0].message.content
    return answer

def load_url(url):
    loader = WebBaseLoader([url])
    docs = loader.load()
    docs_transformed = html2text.transform_documents(docs)
    st.session_state.doc_text = docs_transformed[0].page_content
    st.session_state.title = docs_transformed[0].metadata['title']
    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

        
def send_click():
        
    print ('Received a click request: ', st.session_state.user)
    if st.session_state.user != '':
        url = st.session_state.user
        load_url(url)
        with col2:
            st.markdown('### ' + st.session_state.title)
            st.markdown(url)
        doc = st.session_state.doc_text
        with col2:
            with st.expander("**CVE Details**", expanded=False):
                st.session_state.cve_details = find_cve(doc)
                st.session_state.cve_details = ', '.join(st.session_state.cve_details)
                st.markdown(st.session_state.cve_details)
        with col2:
            with st.expander("**Affected Products**", expanded=False):
                affected_products = produce_output(doc, system_message_for_affected_products, gpt_4o)
                st.session_state.affected_products = group_affected_products(affected_products, gpt_4turbo)
                st.markdown(st.session_state.affected_products)
        with col2:
            st.session_state.description  = produce_output(doc, system_message_for_descriptions, gpt_4turbo)
            with st.expander("**Description**", expanded=False):
                st.markdown(st.session_state.description)
                
        with col2:
            attack_scenarios  = produce_output(doc, system_message_for_attack_scenarios, gpt_4turbo)
            with st.expander("**Attack Scenario**", expanded=False):
                st.markdown(attack_scenarios)
        impact  = produce_output(doc, system_message_for_impact, gpt_4turbo)
        with col2:
            with st.expander("**Impact**", expanded=False):
                st.markdown(impact)
        with col2:
            mitigation = produce_output(doc, system_message_for_mitigations, gpt_4turbo)
            with st.expander("**Mitigation**", expanded=False):
                st.markdown(mitigation)
        with col2:
            tags = produce_output(doc, system_message_for_tags, gpt_4turbo)
            with st.expander("**Tags**", expanded=False):
                st.markdown(tags)

with col2:
    st.markdown("##### URL")
    st.text_input("Ask a question:", key="user", label_visibility="collapsed")
    st.button("Produce Report ", on_click=send_click)
    print ('Received a request: ', st.session_state.user)

# with col2:
#     with st.expander("**CVE Details**"):
#         st.markdown(st.session_state.cve_details)

#     with st.expander("**Affected Products**"):
#         st.markdown(st.session_state.affected_products)
#     with st.expander("**Description**"):
#         st.markdown(st.session_state.description)
