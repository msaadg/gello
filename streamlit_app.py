import streamlit as st
import os
import tempfile
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Sidebar Navigation
st.sidebar.title("Gello: Chatbot")
page = st.sidebar.selectbox("Choose Functionality", ["Text Chat", "Image Processing", "PDF Reader"])
st.sidebar.subheader("Follow Me")
st.sidebar.link_button("LinkedIn", "https://www.linkedin.com/in/msaad01")
st.sidebar.link_button("GitHub", "https://github.com/msaadg", type="secondary")
st.sidebar.caption("Developed by: Muhammad Saad")

# Image Processing
def process_image(text, image_data):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    message = HumanMessage(content=[{"type": "text", "text": text}, {"type": "media", "mime_type": "image/jpeg", "data": image_data}])
    return llm.invoke([message]).content

def image_page():
    st.title("Image Processing")
    text = st.text_input("Enter Text")
    image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if st.button("Ask Question") and image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = process_image(text, image_data)
        st.subheader("Response")
        st.write(response)
        
# Text Chat
def chat_page():
    st.title("Text Chat")
    text = st.text_input("Enter Text")
    if st.button("Ask Question"):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        response = llm.invoke(text)
        st.subheader("Response")
        st.write(response.content)

# PDF Reader
def load_pdf(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def create_embeddings(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
    vector = FAISS.from_texts(chunks, embedding=embedding)
    vector.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the context provided. If answer not available, answer yes or no.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_question(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
    db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.subheader("Response")
    st.write(response["output_text"])

def pdf_page():
    st.title("PDF Reader")
    with st.sidebar:
        pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Submit File") and pdfs:
            text = load_pdf(pdfs)
            chunks = split_text(text)
            create_embeddings(chunks)
            st.success("PDF Processed")
    
    question = st.text_input("Enter Question")
    if st.button("Ask Question"):
        if question and os.path.exists("faiss_index/index.faiss"):
            process_question(question)
        elif not os.path.exists("faiss_index/index.faiss"):
            st.warning("Please upload and submit a PDF first to process its content.")
        else:
            st.warning("Please enter a question.")

# Page Routing
if page == "Text Chat":
    chat_page()
elif page == "Image Processing":
    image_page()
elif page == "PDF Reader":
    pdf_page()