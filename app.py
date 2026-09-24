import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config( 
    page_title="Knowledge Bot",
    page_icon="",
    layout="wide"
)

# Initialize connections

def init_connections():
    supabase = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SERVICE_KEY'))
    groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    embedding_model = SentenceTransformer('all-MiniLM-l6-v2')
    return supabase, embedding_model, groq_client

supabase, embedding_model, groq_client = init_connections()

# Core function

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1

        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        if next_start == start:
            next_start = start + 1
        start = next_start

    return chunks

def ingest_document(title, content, source):
    chunks = chunk_text(content)
    stored = 0
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        supabase.table('documents').insert({
            "title": title,
            "source": source,
            "content": chunk,
            "page_number": i + 1,
            "embedding": embedding,
        }).execute()
        stored += 1
    return stored


def rag_search(question, top_k=5):
    query_embedding = embedding_model.encode(question).tolist()
    last_error = None

    for function_name in ('match_documents', 'match_document'):
        try:
            results = supabase.rpc(function_name, {
                'query_embedding': query_embedding,
                'match_threshold': 0.0,
                'match_count': top_k
            }).execute()
            if results.data is not None:
                return results.data
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def rag_generate(question, documents):
    context = ""
    for i, doc in enumerate(documents):
        context += "SOURCE: " + doc['title'] + "\n"
        context += doc['content'] + "\n"
        if i < len(documents) - 1:
            context += "\n---\n\n"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Answer questions based ONLY on the provided documents. Cite sources as [Document, title]. If the documents don't contain the answer, say so."},
            {"role": "user", "content": "DOCUMENT:\n" + context + "\n\nQUESTION: " + question}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# Document upload - sidebar
with st.sidebar:
    st.header("Add Documents")
    upload_file = st.file_uploader("Upload a text file", type=['txt'])
    doc_title = st.text_input("Document title")

    if st.button("Ingest Document"):
        if upload_file is not None and doc_title.strip():
            content = upload_file.read().decode('utf-8')
            with st.spinner("Chunking, embedding, and storing..."):
                count = ingest_document(doc_title, content, upload_file.name)
                st.success(f"Stored {count} chunks from {doc_title}.")
        else:
            st.warning("Please provide both a file and a title.")

st.divider()

try:
    count = supabase.table('documents').select('id', count='exact').execute()
    st.caption("Documents in database:" +str(count.count))
except: 
    st.caption("Database connection error")
    
# Main Chat interphase

st.title("Company Knowledge Bot")
st.caption("Ask questions about our policies, manuals and documents.")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

#Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(" -" + s['title'] + "(similarity: " + str(s['similarity']) + ")")

# Chat input
question = st.chat_input("Ask a question...")

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = rag_search(question)
            if not docs:
                answer = "I don't have information about that in my knowledge base."
                sources = []
            else:
                answer = rag_generate(question, docs)
                sources = []
                for doc in docs:
                    sources.append({"title": doc['title'], "similarity": round(doc['similarity'], 3)})

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(" - " + s['title'] + " (similarity: " + str(s['similarity']) + ")")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})



