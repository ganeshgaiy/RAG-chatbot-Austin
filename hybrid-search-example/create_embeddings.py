from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from tqdm.auto import tqdm
from typing import List, Dict
import time
import os
import streamlit as st

# Set OpenAI API Key
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Add debug prints for initial setup
print("Starting process...")
print(f"OpenAI API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Pinecone setup
index_name = "hybrid-search-langchain-pinecone"
pinecone_api_key = "pcsk_5RyqYk_Ect86WX6psjCEm6r6Gs3ojWrsyfBPaUtXTmndAmwu4uro2vxnjfLF96CYQZe4R9"
pc = Pinecone(api_key=pinecone_api_key)

# Check Pinecone connection and create index if needed
print("\nChecking Pinecone connection...")
try:
    indexes = pc.list_indexes().names()
    print(f"Available indexes: {indexes}")
    
    if index_name not in indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
            print("Waiting for index to be ready...")
except Exception as e:
    print(f"Error connecting to Pinecone: {str(e)}")

# Initialize index
index = pc.Index(index_name)

# Initialize models and loaders
print("\nInitializing models and loaders...")
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
bm25_encoder = BM25Encoder().default()

# Load and process documents
print("\nLoading documents...")
loader = PyPDFDirectoryLoader("ganesh-resumes")
try:
    docs = loader.load()
    print(f"Number of documents loaded: {len(docs)}")
except Exception as e:
    print(f"Error loading documents: {str(e)}")
    exit()

# Split documents
print("\nSplitting documents...")
final_documents = text_splitter.split_documents(docs)
print(f"Number of chunks after splitting: {len(final_documents)}")

def create_and_store_embeddings(documents: List, embeddings_model, bm25_encoder, pc, index_name: str):
    """
    Create dense and sparse embeddings for documents and store them in Pinecone.
    """
    print(f"\nStarting embeddings creation for {len(documents)} documents...")
    
    index = pc.Index(index_name)
    batch_size = 100
    
    for i in tqdm(range(0, len(documents), batch_size)):
        try:
            batch = documents[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}, size: {len(batch)}")
            
            # Prepare texts and metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Create embeddings
            print("Creating dense embeddings...")
            dense_embeddings = embeddings_model.embed_documents(texts)
            print(f"Dense embeddings created: {len(dense_embeddings)}")
            
            print("Creating sparse embeddings...")
            sparse_embeddings = bm25_encoder.encode_documents(texts)
            print(f"Sparse embeddings created: {len(sparse_embeddings)}")
            
            # Prepare vectors
            vectors = []
            for j, (dense_emb, sparse_emb, metadata) in enumerate(zip(dense_embeddings, sparse_embeddings, metadatas)):
                vector_id = f"vec_{i + j}"
                vector = {
                    "id": vector_id,
                    "values": dense_emb,
                    "sparse_values": sparse_emb,
                    "metadata": {
                        **metadata,
                        "context": texts[j]
                    }
                }
                vectors.append(vector)
            
            # Upsert to Pinecone
            print(f"Upserting {len(vectors)} vectors to Pinecone...")
            index.upsert(vectors=vectors)
            print("Upsert successful")
            
            # Verify after each batch
            stats = index.describe_index_stats()
            print(f"Current total vectors in index: {stats.total_vector_count}")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            raise e

# Main execution
print("\nStarting main embedding process...")
try:
    create_and_store_embeddings(
        documents=final_documents,
        embeddings_model=embeddings,
        bm25_encoder=bm25_encoder,
        pc=pc,
        index_name=index_name
    )
except Exception as e:
    print(f"Error in main process: {str(e)}")

# Final verification
print("\nVerifying final upload...")
index = pc.Index(index_name)
index_stats = index.describe_index_stats()
print(f"Final total vectors in index: {index_stats.total_vector_count}")