from dotenv import load_dotenv
import os
from src.helper import load_pdf_file,text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
import time
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired


if index_name not in pc.list_indexes().names():
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # must match embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until index is ready
    while True:
        status = pc.describe_index(index_name)
        if status.status['ready']:
            break
        print("Waiting for index to be ready...")
        time.sleep(2)

print("Index ready.")

# ✅ Connect to index
index = pc.Index(index_name)

# ✅ Upload documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Indexing complete.")