## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import time
import tracemalloc
import csv

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Define chunking strategies (Chunk Size, Overlap)
CHUNKING_STRATEGIES = [
    (200, 0),
    (200, 50),
    (200, 100),
    (500, 0),
    (500, 50),
    (500, 100),
    (1000, 0),
    (1000, 50),
    (1000, 100),
]

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using specified model
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def preprocess_text(text: str) -> str:
    # Remove non-ASCII characters
    text = ''.join([char for char in text if ord(char) < 128])

    # Remove extra whitespace by splitting and joining words
    text = ' '.join(text.split())

    return text


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    # Preprocess the text before splitting into chunks
    text = preprocess_text(text)

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir, chunk_size, overlap, csv_filename):
    start_time = time.time()
    tracemalloc.start()

    total_chunks = 0
    total_files = 0
    all_chunks = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                # Preprocess the text before chunking
                text = preprocess_text(text)

                chunks = split_text_into_chunks(text, chunk_size, overlap)
                total_chunks += len(chunks)

                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),  # Storing full chunk instead of index
                        embedding=embedding,
                    )

                # Collect all chunks for the query later
                all_chunks.extend(chunks)

            print(f" -----> Processed {file_name}")  # Log progress
            total_files += 1

    # After processing all documents for this chunking strategy, query Redis once
    answer = query_redis("What is a binary search tree?")  # Query once for the combined text

    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Store the result for this chunking strategy, including speed and memory
    store_query_result(chunk_size, overlap, elapsed_time, peak_memory / 1e6, total_chunks, answer, csv_filename)

    return chunk_size, overlap, elapsed_time, peak_memory / 1e6, total_chunks


def store_query_result(chunk_size, overlap, speed, peak_memory, total_chunks, answer, csv_filename):
    """
    Store the query result for a given chunking strategy in a CSV file.
    """
    # Ensure answer only contains ASCII characters
    answer = ''.join([char for char in answer if ord(char) < 128])

    # Check if the file exists to add the header row only once
    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if the file doesn't exist
        if not file_exists:
            writer.writerow(["Chunk Size", "Overlap", "Speed (s)", "Memory (MB)", "Total Chunks", "Resulting Documents"])


        # Write the result row
        writer.writerow([chunk_size, overlap, speed, peak_memory, total_chunks, answer])


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    result = "No results found"
    if res.docs:
        result = "\n".join([f"{doc.id} - {doc.vector_distance}" for doc in res.docs])
        print(result)
    return result


def main():
    clear_redis_store()
    create_hnsw_index()

    results = []
    csv_filename = "chunking_results.csv"

    for chunk_size, overlap in CHUNKING_STRATEGIES:
        chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename)
        results.append([chunk_size, overlap, time_taken, memory_used, num_chunks])

    print("\n---Done processing PDFs---\n")
    #query_redis("What do we have in our arsenal?")


if __name__ == "__main__":
    main()