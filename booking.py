import pandas as pd 
import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load the Embedding Model
@st.cache_resource
def load_embeedings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# Read the CSV file
df = pd.read_csv(r"C:\Users\rishab\Desktop\bookingbee\services_full_corrected (1).csv", encoding="ISO-8859-1", on_bad_lines='skip')



# Make each row as a chunk (Document)
chunks = []
for idx, row in df.iterrows():
    text = "\n".join([
        f"{col}: {row[col]}"
        for col in df.columns
        if pd.notna(row[col])
    ])
    chunks.append(Document(page_content=text, metadata={"row_index": idx}))


# Create embeddings for each row
embeddings = load_embeedings()


# Create Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_store")


# Load the vector store 
@st.cache_resource
def load_vector_store():
    vector_store = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
    return vector_store


# 
# Initialize the LLM 
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8-mupeNy17IfCIX0E9DGm-FsjOSYO9N8"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1
)


def get_rag_answer(user_query: str, top_k: int = 3) -> str:
    """
    1. Convert user query into embeddings
    2. Perform similarity search on chunks
    3. Get top_k chunks as context
    4. Send context + user query to LLM for answer
    """
    # Normalize the user query
    user_query = user_query.lower()

    # Load the vector store
    vector_store = load_vector_store()

    # Perform similarity search
    relevant_docs = vector_store.similarity_search(user_query, k=top_k)

    print("Relevant documents:")
    for doc in relevant_docs:
        print(doc.page_content)  # or whatever field holds the relevant data

    # Join the top chunks into context
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful assistant for a salon.
Use ONLY the information provided in CONTEXT to answer the QUESTION.
If the answer is not present in the context, say you do not know.

CONTEXT:
{context_text}

QUESTION:
{user_query}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content


# Streamlit frontend
st.set_page_config(page_title="Salon RAG Chatbot", page_icon="💬")

st.title("Salon Chatbot 💬")
st.write("Ask things like: **What is the price of Men's Haircut?**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from the RAG function
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_rag_answer(user_input, top_k=3)  # Retrieve top 3 chunks
            st.markdown(answer)

    # Add assistant's message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
