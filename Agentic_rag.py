import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness,answer_correctness,context_precision,context_recall)

from supabase.client import Client, create_client
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from typing import List

#load_environment variables
load_dotenv()

#supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url,supabase_key)

#embedding models
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

#vectorstore
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client = supabase,
    table_name="documents",
    query_name="match_documents"
)

#large Language Model
llm = ChatOpenAI(
    model = 'gpt-4o',
    temperature=0.6
)

#prompt 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided documents to answer questions. "
               "Always cite which Source IDs support your answer."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

#Output Schema
class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer:str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer"
    )

#Wrap LLM with structured output
structured_llm = llm.with_structured_output(CitedAnswer)

#create the tools
@tool(response_format="content_and_artifact")
def retrieve(query:str):
    """Retrieve information related to a query from the Supabase vector store."""
    retrieved_docs = vector_store.similarity_search(query,k=3)
    serialized = "\n\n".join(
        f"Source ID: {i}\nMetadata: {doc.metadata}\nContent: {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )
    return serialized, retrieved_docs

#tools and LLM
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

#agent_executor
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

#invoke the agent
response =  agent_executor.invoke({"input":"what is evaluation target?"})

cited_result = structured_llm.invoke(response["output"])

#result
print("\n=== Final Answer ===")
print(cited_result.answer)

print("\n=== Citations (Source IDs) ===")
print(cited_result.citations)

#queries with ground thruth

qa_pairs = [
    {
        "question": "What is the main advantage of Retrieval-Augmented Generation (RAG) over standalone generative models?",
        "ground_truth": "By retrieving relevant information from external sources, RAG significantly reduces the incidence of hallucinations or factually incorrect outputs, thereby improving the content’s reliability."
    },
    {
        "question": "What are the two primary components of a RAG system?",
        "ground_truth": "The RAG system comprises two primary components: Retrieval and Generation."
    },
    {
        "question": "What does the retrieval component of RAG involve?",
        "ground_truth": "The retrieval component aims to extract relevant information from external knowledge sources. It involves two phases: indexing, which organizes documents, and searching, which fetches relevant documents, often incorporating rerankers."
    },
    {
        "question": "What is the purpose of A Unified Evaluation Process of RAG (Auepora)?",
        "ground_truth": "We introduce A Unified Evaluation Process of RAG (Auepora), which focuses on three key questions of benchmarks: What to Evaluate? How to Evaluate? How to Measure? correlated to Target, Dataset, and Metric respectively."
    }
]

# build pipeline rows
pipeline_rows = []
for qa in qa_pairs:
    out = agent_executor.invoke({"input": qa["question"]})
    ans = out["output"]

    # run retriever separately to capture actual contexts
    retrieved_docs = vector_store.similarity_search(qa["question"], k=3)
    ctx = [doc.page_content for doc in retrieved_docs]

    pipeline_rows.append({
        "question": qa["question"],
        "answer": ans,
        "contexts": ctx,
        "ground_truth": qa["ground_truth"],
    })

# dataset for ragas
dataset = Dataset.from_list(pipeline_rows)
print(dataset[:2])

# Evaluation ragas
non_llm_results = evaluate(
    dataset,
    metrics=[faithfulness, answer_correctness, context_precision, context_recall],
    llm=None,
    embeddings=embeddings,
)
print("Non-LLM results:", non_llm_results)
