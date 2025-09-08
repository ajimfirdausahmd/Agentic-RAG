### Agentic Retrieval-Augmented Generation (RAG) with Supabase & LangChain

This project implements a Retrieval-Augmented Generation (RAG) pipeline using:

* LangChain for orchestration

* Supabase as the vector store

* OpenAI embeddings & LLMs

* Ragas for evaluation of the RAG system

The pipeline retrieves relevant documents from Supabase, generates structured answers with citations, and evaluates the performance using multiple metrics.

### Features

Integration with Supabase vector store for document retrieval

✅ OpenAI GPT-4o model for generation

✅ Structured output with citations of retrieved sources

✅ Custom retrieval tool using LangChain’s @tool decorator

✅ Automated evaluation pipeline with Ragas
 (faithfulness, correctness, precision, recall)

### Project Structure    

├── Agentic_rag.py             # Core Agentic RAG pipeline with evaluation
├── Agentic_rag_streamlit.py   # Streamlit app for interactive Q&A
├── ingest.py                  # Script to ingest documents into Supabase
├── requirements.txt           # Python dependencies
├── README.md   


### Tech Stack

LangChain → Agent orchestration
Supabase → Vector database
OpenAI → GPT-4o + embeddings
Ragas → Evaluation framework
Streamlit → Interactive UI
