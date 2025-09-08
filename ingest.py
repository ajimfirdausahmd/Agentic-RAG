
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

from supabase.client import Client, create_client

#load environment variables
load_dotenv()

#supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

if not supabase_url or not supabase_key:
    raise ValueError('Supabase credentials are missing. Check your .env file')
else:
    print('supabase credentials done')

#embedding models
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

#load pdf docs
loader = PyPDFDirectoryLoader('knowledge_base')
documents = loader.load()

#split the documents in multiple chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
text = text_splitter.split_documents(documents)

vector_store = SupabaseVectorStore.from_documents(
    text,
    embeddings,
    client = supabase,
    table_name = "documents",
    query_name = 'match_documents',
    chunk_size=1000,
)
