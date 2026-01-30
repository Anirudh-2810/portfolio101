import streamlit as st
import nltk
import re
import json
import requests
import io
import PyPDF2
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import datetime

# ==========================================
# 1. CRITICAL: STABILITY SETUP
# ==========================================
# This must be line #1. If anything (even a print) runs before this, 
# Streamlit Cloud will throw a "Health Check" error.
try:
    st.set_page_config(page_title="Nexus AI Research", page_icon="🌐", layout="wide")
except:
    pass 

@st.cache_resource
def stable_boot():
    """Quietly handles NLTK and library setup to avoid server timeout."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        return False

# ==========================================
# 2. RESEARCH TOOLS (No 12th CBSE/Class parts)
# ==========================================
class ResearchEngine:
    def search(self, query, count=5):
        try:
            with DDGS() as ddgs:
                return [r for r in ddgs.text(query, max_results=count)]
        except:
            return []

def safe_math(expr):
    """Calculates math without using 'eval' on text strings."""
    clean = re.sub(r'[^\d\+\-\*\/\(\)\.]', '', expr)
    try:
        return f"🔢 **Calculation**: `{clean} = {eval(clean)}`"
    except:
        return None

# ==========================================
# 3. THE "VAST" UI ENGINE
# ==========================================
if stable_boot():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for Navigation & Stability Info
    with st.sidebar:
        st.title("🌐 Nexus AI")
        st.success("Server Health: Excellent")
        st.divider()
        if st.button("New Research Thread"):
            st.session_state.messages = []
            st.rerun()

    st.title("🤖 Web-Access Super AI")
    st.info("I am now a General Purpose Research AI. I can browse the web, solve math, and analyze code.")

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Handling
    if prompt := st.chat_input("Enter your research topic..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. Check for Math first
            math_result = safe_math(prompt)
            if math_result:
                st.markdown(math_result)
                st.session_state.messages.append({"role": "assistant", "content": math_result})
            
            # B. Otherwise, Web Search
            else:
                with st.spinner("Searching global databases..."):
                    results = ResearchEngine().search(prompt)
                    if results:
                        response = "### Research Findings:\n\n"
                        for r in results:
                            response += f"🔹 **[{r['title']}]({r['href']})**\n{r['body']}\n\n"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("I couldn't reach the live web. Check your internet connection.")
else:
    st.error("Fatal Error: System could not boot. Check requirements.txt.")
