import openai
import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Import environment variables from .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set logging flag to debug to view requests made to LLM
openai.log = "info" # 'info' = less verbose, 'debug' = more verbose

# Collect the documents in the 'data' folder , which in this case
# is just a single .txt file containing the text of Paul Graham's
# essay "What I Worked On".
documents = SimpleDirectoryReader('data').load_data()

# Build an index from the documents in the data folder.
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index to query the documents.
query_engine = index.as_query_engine()

# Send query to the query engine and print the response.
response = query_engine.query('What did the author do growing up?')

print('Query: What did the author do growing up?')
print(response)

# By default, data is stored in-memory. To persist the index to disk
# at `./storage`, we can use `storage_context`:
index.storage_context.persist()

# Then, we can reload the index from disk by:

# 1. Rebuilding the storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# 2. Loading the index from the storage context
index = load_index_from_storage(storage_context)