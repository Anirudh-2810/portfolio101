import streamlit as st
import nltk
import re
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# 1. SETUP FIRST
st.set_page_config(page_title="Nexus Super AI", layout="wide")

# 2. DEFINE TOOLS (Prevents NameError)
def safe_math(expr):
    clean = re.sub(r'[^\d\+\-\*\/\(\)\.]', '', expr)
    try:
        if any(op in clean for op in '+-*/'):
            return f"🔢 **Math**: `{clean} = {eval(clean)}`"
    except:
        return None
    return None

class ResearchEngine:
    def search(self, query):
        try:
            with DDGS() as ddgs:
                # Using .get() prevents the KeyError: 'href'
                raw = list(ddgs.text(query, max_results=3))
                return raw if raw else []
        except:
            return []

# 3. UI LOGIC
st.title("🤖 Nexus Research AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Search or Calculate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Math check
        m_res = safe_math(prompt)
        if m_res:
            st.markdown(m_res)
            st.session_state.messages.append({"role": "assistant", "content": m_res})
        else:
            # Web check
            with st.spinner("Searching..."):
                results = ResearchEngine().search(prompt)
                if results:
                    resp = ""
                    for r in results:
                        # Use .get() to avoid KeyError
                        title = r.get('title', 'No Title')
                        body = r.get('body', 'No Description')
                        link = r.get('href', '#')
                        resp += f"### {title}\n{body}\n[Link]({link})\n\n---\n"
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                else:
                    st.error("Live web unreachable. Try again in 1 minute.")
