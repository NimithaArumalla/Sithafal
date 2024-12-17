import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Website Content Extraction
def extract_text_from_url(url):
    """Extracts and cleans text from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove scripts and styles
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Extract visible text
    text = soup.get_text(separator='\n')
    return text

# Step 2: Text Chunking
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Chunks the text into smaller pieces for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Step 3: Convert Text Chunks to Embeddings
def create_embeddings(text_chunks):
    """Generates embeddings for the text chunks and stores them in a FAISS vector database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a free SentenceTransformer model
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_chunks

# Step 4: Query Handling and Response Generation
def query_website_system(index, text_chunks, query):
    """Handles user queries and retrieves answers using the vector database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)  # Retrieve top 5 matches
    results = [text_chunks[i] for i in indices[0]]
    return "\n".join(results)

# Example Workflow
def main():
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/",
    ]

    # Step 1: Extract text from websites
    print("Extracting text from websites...")
    all_text = ""
    for url in urls:
        print(f"Processing URL: {url}")
        all_text += extract_text_from_url(url) + "\n"

    # Step 2: Chunk the extracted text
    print("Chunking text...")
    text_chunks = chunk_text(all_text)

    # Step 3: Create embeddings and store in a FAISS vector database
    print("Creating embeddings...")
    index, chunks = create_embeddings(text_chunks)

    # Step 4: Handle a sample query
    print("Handling user query...")
    user_query = input("enter query")  # Replace with user query
    response = query_website_system(index, chunks, user_query)

    print("Response:", response)

if __name__ == "__main__":
    main()
