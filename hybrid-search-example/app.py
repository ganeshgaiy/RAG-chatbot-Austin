import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize Pinecone connection once
if 'pinecone_initialized' not in st.session_state:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("hybrid-search-langchain-pinecone")
    
    # Initialize models only once
    embeddings = OpenAIEmbeddings()
    bm25_encoder = BM25Encoder().default()
    
    # Store in session state
    st.session_state.pc = pc
    st.session_state.index = index
    st.session_state.embeddings = embeddings
    st.session_state.bm25_encoder = bm25_encoder
    st.session_state.pinecone_initialized = True

# Initialize LLM and prompt template
if 'llm' not in st.session_state:
    groq_api_key = os.getenv("GROQ_API_KEY")
    st.session_state.llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize retriever only once
if 'retriever' not in st.session_state:
    st.session_state.retriever = PineconeHybridSearchRetriever(
        embeddings=st.session_state.embeddings,
        sparse_encoder=st.session_state.bm25_encoder,
        index=st.session_state.index,
        alpha=0.5
    )

def process_query(query: str, docs):
    """Process the query and generate response"""
    # Create messages for the chat
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant analyzing resumes."
        },
        {
            "role": "user",
            "content": f"""
            Based on the following resume content, please answer this question: {query}
            
            Resume Content:
            {' '.join([doc.page_content for doc in docs])}
            """
        }
    ]
    
    # Get response from LLM
    response = st.session_state.llm.invoke(messages)
    return response.content

# Streamlit UI
st.title("Ganesh's Resume Q&A")

user_prompt = st.text_input("Enter your query about the resumes:", 
                           placeholder="e.g., 'What are the technical skills mentioned in the resumes?'")

if user_prompt:
    with st.spinner("Searching resumes..."):
        try:
            # Get documents using get_relevant_documents
            docs = st.session_state.retriever.invoke(user_prompt)

            if not docs:
                st.warning("No relevant information found in the resumes.")
            else:
                # Generate response
                start = time.process_time()
                response = process_query(user_prompt, docs)
                response_time = time.process_time() - start
                
                # Display results
                st.success(f"Response generated in {response_time:.2f} seconds")
                st.markdown("### Answer")
                st.markdown(f">{response}")
                
                # # Optional: Show source documents
                # if st.checkbox("Show source documents"):
                #     st.markdown("### Source Documents")
                #     for i, doc in enumerate(docs, 1):
                #         with st.expander(f"Document {i}"):
                #             st.write(doc.page_content)
                #             if hasattr(doc, 'metadata'):
                #                 st.write("Metadata:", {k: v for k, v in doc.metadata.items() if k != 'context'})
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                st.error(f"Traceback: {traceback.format_tb(e.__traceback__)}")

st.markdown("---")
st.markdown("""
    This application uses:
    - Groq LLM for answer generation
    - Pinecone for vector storage
    - Hybrid search combining dense (semantic) and sparse (keyword) searching
    - OpenAI embeddings for dense vectors
    - BM25 for sparse vectors
""")