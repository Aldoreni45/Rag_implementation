FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install pip dependencies with staged approach
RUN pip install --upgrade pip

# Install core dependencies first
RUN pip install --timeout=1000 --retries=5 --no-cache-dir \
    flask>=3.1.1 \
    sentence-transformers>=4.1.0 \
    pypdf>=5.6.1 \
    python-dotenv>=1.1.0 \
    pydantic>=2.0 \
    requests>=2.31.0

# Install langchain packages
RUN pip install --timeout=1000 --retries=5 --no-cache-dir \
    langchain>=0.3.0,<0.4.0 \
    langchain-pinecone>=0.2.0 \
    langchain-community>=0.3.0 \
    langchain-huggingface>=0.1.0 \
    langchain-cohere>=0.1.0

# Install Google packages with specific compatible versions
RUN pip install --timeout=1000 --retries=5 --no-cache-dir \
    google-generativeai==0.7.2 \
    langchain-google-genai>=2.1.0

# Install PyTorch CPU version
RUN pip install --timeout=1000 --retries=5 --no-cache-dir \
    torch>=1.11.0 --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . /app

EXPOSE 8080

CMD ["python3", "app.py"]