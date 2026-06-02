import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Tender Assistant", page_icon="🏗️", layout="wide")

CHROMA_PATH = "chroma_db"
INPUT_DIR = "Input"

# --- Models & Prompts ---

@st.cache_resource
def get_embeddings():
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is missing in environment variables.")
        st.stop()
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- UI Setup ---

with st.sidebar:
    st.title("🏗️ Tender Assistant")
    page = st.radio("Navigation", ["📄 Document Upload", "💬 Chat Query", "📊 Tender Analysis"])

if page == "📄 Document Upload":
    st.header("Upload Tender Document")
    st.write("Upload a PDF to embed and analyze its contents.")
    
    uploaded_file = st.file_uploader("Drag & drop your PDF here", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a minute."):
                os.makedirs(INPUT_DIR, exist_ok=True)
                file_path = os.path.join(INPUT_DIR, "uploaded_document.pdf")
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                try:
                    if os.path.exists(CHROMA_PATH):
                        try:
                            old_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
                            old_db.delete_collection()
                        except Exception:
                            pass
                        
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                    chunks = splitter.split_documents(docs)
                    
                    embeddings = get_embeddings()
                    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                    
                    BATCH_SIZE = 100 
                    for i in range(0, len(chunks), BATCH_SIZE):
                        batch_chunks = chunks[i:i + BATCH_SIZE]
                        db.add_documents(batch_chunks)
                        
                    st.success(f"Successfully uploaded and embedded {len(docs)} pages.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

elif page == "💬 Chat Query":
    st.header("Document Assistant")
    st.write("Ask anything about the uploaded tender document.")
    
    if not os.path.exists(CHROMA_PATH):
        st.warning("Database not found. Please upload a document first.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    st.caption(f"Sources: {', '.join(msg['sources'])}")
                    
        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    embeddings = get_embeddings()
                    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                    
                    results = db.similarity_search(prompt, k=3)
                    if not results:
                        response_text = "No relevant context found in the document."
                        unique_sources = []
                        st.markdown(response_text)
                    else:
                        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
                        
                        PROMPT_TEMPLATE = """
                        Answer the question based only on the following context.
                        Please provide a concise and summarized answer rather than quoting large sections of text.

                        {context}

                        ---

                        Answer the question based on the above context: {question}
                        """
                        
                        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                        chain_prompt = prompt_template.format(context=context_text, question=prompt)
                        
                        model = get_llm()
                        response = model.invoke(chain_prompt)
                        
                        sources = [doc.metadata.get("source", None) for doc in results]
                        pages = [doc.metadata.get("page", None) for doc in results]
                        unique_sources = list(set([f"{s} (Page {p})" if p is not None else str(s) for s, p in zip(sources, pages)]))
                        
                        response_text = response.content
                        st.markdown(response_text)
                        if unique_sources:
                            st.caption(f"Sources: {', '.join(unique_sources)}")
                            
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": unique_sources
                    })

elif page == "📊 Tender Analysis":
    st.header("Automated Tender Report")
    st.write("Extracts 13 critical tender clauses automatically.")
    
    if not os.path.exists(CHROMA_PATH):
        st.warning("Database not found. Please upload a document first.")
    else:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Scanning document and generating report... This may take a minute."):
                TENDER_CLAUSES = [
                    "Bid bond", "Performance bond", "Advance payment", "Retention clause",
                    "Interim payment terms", "Final Payment terms", "Insurance clause",
                    "Force majure clause", "Weather clause", "Arbitration/dispute clause",
                    "Defect liability period", "Liquidated damages/Delay Damages", "Progress Milestones"
                ]
                
                PROMPT_TEMPLATE = """
                You are a construction contract expert. Based ONLY on the following context retrieved from a tender document, summarize the requirements or details related to "{clause_name}".
                If the context does not contain any relevant information about this clause, reply exactly with "Not specified in the retrieved context." Keep your summary concise and focused.
                Context:
                {context}
                """
                
                embeddings = get_embeddings()
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                model = get_llm()
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                
                analysis_results = []
                
                for clause in TENDER_CLAUSES:
                    docs = db.similarity_search(clause, k=4)
                    if not docs:
                        analysis_results.append({"Clause": clause, "Summary": "No context found.", "Source Pages": "N/A"})
                        continue
                        
                    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
                    chain_prompt = prompt_template.format(context=context_text, clause_name=clause)
                    response = model.invoke(chain_prompt)
                    
                    pages = [str(doc.metadata.get("page", "Unknown")) for doc in docs if doc.metadata.get("page") is not None]
                    unique_pages = ", ".join(sorted(set(pages))) if pages else "N/A"
                    
                    analysis_results.append({
                        "Clause": clause,
                        "Summary": response.content.strip(),
                        "Source Pages": f"Page {unique_pages}"
                    })
                    
                st.table(analysis_results)
