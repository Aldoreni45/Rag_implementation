from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import google.generativeai as genai  # Fixed import
from langchain_groq import ChatGroq

app = Flask(__name__)

load_dotenv()

# LangChain tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "my-rag-project")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Check required API keys
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found in .env file")
    exit(1)
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file") 
    exit(1)
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file") 
    exit(1)

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Configure Google AI for embeddings
genai.configure(api_key=GEMINI_API_KEY)

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone vector store
index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize Groq chat model
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",  # Updated model name
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(f"User input: {input}")
        
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print(f"Response: {answer}")
        
        return str(answer)
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I encountered an error processing your request."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)