import os
from dotenv import load_dotenv

import streamlit as st

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

from supabase.client import Client, create_client
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

#streamlit
st.set_page_config(page_title="Agentic RAG Chatbot")
st.title("Agentic RAG Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input chat
user_question = st.chat_input("Ask me something...")

if user_question:
    # Add user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(content=user_question))
    
    # Run agent
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})

    # Force structured output
    cited = structured_llm.invoke(result["output"])
    if cited.citations:  # only if not empty
        final_text = f"{cited.answer}\n\nSources: {cited.citations}"
    else:
        final_text = cited.answer


    # Add assistant reply
    with st.chat_message("assistant"):
        st.markdown(final_text)
    st.session_state.messages.append(AIMessage(content=final_text))
