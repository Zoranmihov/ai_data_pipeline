# app/main.py
import streamlit as st
import requests
import json
import pypdfium2 as pdfium

OLLAMA_URL = "http://ollama:11434/api/generate"

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf = pdfium.PdfDocument(file)
    text = ""
    for page in pdf:
        text += page.get_textpage().get_text_range() + "\n\n"
    return text


def query_ollama(prompt):
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json().get("response", "No response from model.")
    else:
        return f"Error: {response.status_code}"

st.title("PDF-based RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", text, height=200)

    user_query = st.text_input("Enter your query:")
    if user_query:
        prompt = f"Context:\n{text}\n\nUser: {user_query}\nAssistant:"
        response = query_ollama(prompt)
        st.write("### Response:")
        st.write(response)
