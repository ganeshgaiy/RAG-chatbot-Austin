
# RAG Chatbot[aaia rag chat architecture inforgraphic.pdf](https://github.com/user-attachments/files/18761710/aaia.rag.chat.architecture.inforgraphic.pdf)


*A Retrieval-Augmented Generation Chatbot powered by LangChain, Streamlit, and FastAPI.*

---

## *Table of Contents*
1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [How to Run](#how-to-run)
5. [Dependencies](#dependencies)
6. [References](#references)

---

## *Introduction*
The Retrieval-Augmented Generation (RAG) Chatbot integrates large language models with document retrieval capabilities to provide accurate and context-aware responses. This project utilizes:
- *LangChain RetrievalQA pipeline*
- *OpenAI GPT-4 (via ChatOpenAI)*
- *Chroma DB for vector storage*

The chatbot can retrieve information from document-based knowledge sources and generate meaningful responses based on user queries.

---

## *Features*
- User-friendly interface developed in Streamlit
- FastAPI backend for efficient query processing
- Google Drive integration for document retrieval
- Advanced vector embedding storage using Chroma DB

---

## *Setup Instructions*

### Prerequisites:
- *OpenAI API Key | Watsonx API Key:* Obtain from [OpenAI](https://platform.openai.com/) or [WatsonX](https://www.ibm.com/products/watsonx-ai). Add it to .env as:
  plaintext
  OPENAI_API_KEY=your_openai_api_key
  WATSONX_APIKEY=your_watsonx_api_key
  
- *Unstructured API Key:* Obtain from the Unstructured API provider. Add to .env as:
  plaintext
  UNSTRUCTURED_API_KEY=your_unstructured_api_key
  
- *Google Drive API Setup:*
  1. Follow the [Google Drive API Quickstart](https://developers.google.com/drive/api/quickstart/python).
  2. Download and replace service_account.json in the project folder.
  3. Share the Google Drive folder with the client_email specified in service_account.json to allow programmatic access.
- *Python Virtual Environment:*
  bash
  python -m venv venv
  source venv/bin/activate  # For Linux/MacOS
  venv\Scripts\activate   # For Windows
  pip install -r requirements.txt
  

---

## *How to Run*

### Steps:
1. *Process Documents:*
   Run document_processor.py to ingest documents and generate embeddings.
   bash
   python document_processor.py
   
2. *Start Backend:*
   start the FastAPI backend in dev mode.
   bash
   fastapi dev main.py
   
3. *Run Frontend:*
   Launch the Streamlit app.
   bash
   streamlit run streamlit_app.py
   

The chatbot interface will be accessible at http://localhost:8501.

---

## *Dependencies*
The following packages are required to run the project:

- *Streamlit* (1.40.2): Frontend framework
- *FastAPI* (0.115.5): Backend API framework
- *LangChain* (0.3.9): Core library for retrieval and generation pipelines
- *Chroma DB*: Vector storage solution
- *Google API Client*: Integration with Google Drive
- *python-dotenv*: Environment variable management

### File: requirements.txt

streamlit==1.40.2
fastapi==0.115.5
langchain==0.3.9
google-auth==2.36.0
google-api-python-client==2.154.0
langchain-unstructured==0.1.6
python-dotenv==1.0.1
langchain-chroma==0.1.4
langchain-openai==0.2.10


---

## *References*
1. [LangChain Documentation](https://python.langchain.com/docs/introduction/)
2. [OpenAI API Documentation](https://platform.openai.com/docs/)
3. [Chroma DB Documentation](https://www.trychroma.com/)
4. [Streamlit Documentation](https://docs.streamlit.io/)
5. [FastAPI Documentation](https://fastapi.tiangolo.com/)
6. [Google Drive API Quickstart](https://developers.google.com/drive/api/quickstart/python)
