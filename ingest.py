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
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import chromadb
import json

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

# Initialize PyMongo connection
mongo_client = MongoClient(
    "mongodb://jocelyn:abc123@localhost:27018/"
)
db = mongo_client["ds4300-mongodb"]
mongo_collection = db["mongoCollection"]

# Initialize Chroma connection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="chromaCollection")

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

# Used to clear the redis vector store
def clear_store(store_type="redis"):
    # Clear redis store
    if store_type == "redis":
        print("Clearing existing Redis store...")
        redis_client.flushdb()
        print("Redis store cleared.")
    # Delete chroma collection
    elif store_type == "chroma":
        print("Clearing existing Chroma store...")
        try:
            chroma_client.delete_collection("chromaCollection")
        except Exception as e:
            print(f"Error deleting Chroma store: {e}")
        
        # Recreate an empty collection
        chroma_client.create_collection(name="chromaCollection")
        print("Chroma store cleared.")

    # drop mongo database
    elif store_type == "mongo":
        print("Clearing existing Mongo store...")
        mongo_collection.delete_many({})
        print("Mongo store cleared.")

    else:
        print(f"Invalid store type: {store_type}. Must be one of: redis, chroma, mongo")


# # Be able to access chroma from a different file
# def get_chroma_collection():
#     print("check: ", chroma_client.get_collection(name="chromaCollection").count())
#     collection = chroma_client.get_collection(name="chromaCollection")
#     return collection

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
def get_embedding(text: str, model: str="nomic-embed-text") -> list:

    if model=="nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        
    else:
        response = SentenceTransformer(model)

    # return response.encode(text)
    return response["embedding"]


# Store the embedding in given collection
def store_embedding(file: str, page: str, chunk: str, embedding: list, collection: str="redis"):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"

    if collection == "redis":
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
        print(f"Stored embedding for: {chunk} in {collection}")

    elif collection == "mongo":
        # Organize the file
        doc = {
            "key":key,
            "file":file,
            "page":page,
            "chunk":chunk,
            "embedding":embedding
        }

        # Insert it into the collection
        mongo_collection.insert_one(doc)
        print(f"Stored embedding for: {chunk} in {collection}")
    
    elif collection == "chroma":
        # Get or create the Chroma collection
        if "chromaCollection" not in chroma_client.list_collections():
            print(f"Collection: chromaCollection does not exist")
            chroma_collection = chroma_client.create_collection(name="chromaCollection")
        else:
            chroma_collection = chroma_client.get_collection(name="chromaCollection")
        
        # add to chroma collection
        chroma_collection.add(
            documents=[chunk], 
            ids=[key],
            metadatas=[{
                "file":file,
                "page":page,
                "chunk":chunk
            }],
            embeddings=[embedding]
        )

        # Create to save in JSON
        doc = {
            "embedding": embedding,
            "file": file,
            "page": page,
            "chunk": chunk,
            "id":key
        }
        print(f"Stored embedding for: {chunk} in {collection}")
        return doc
    else:
        return(f"Invalid collection: {collection}. Please use one of: redis, mongo, chroma.")


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
def process_pdfs(data_dir, chunk_size, overlap, csv_filename, collection):
    start_time = time.time()
    tracemalloc.start()

    total_chunks = 0
    total_files = 0
    all_chunks = []
    all_docs = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            # initialize a list of docs to save to JSON for chromaCollection
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                # Preprocess the text before chunking
                text = preprocess_text(text)

                chunks = split_text_into_chunks(text, chunk_size, overlap)
                total_chunks += len(chunks)

                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    if collection == "redis":
                        store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),  # Storing full chunk instead of index
                            embedding=embedding,
                            collection="redis"
                        )
                    elif collection == "chroma":
                        emb_doc = store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk),  # Storing full chunk instead of index
                            embedding=embedding,
                            collection="chroma"
                        )
                        all_docs.append(emb_doc)

                # Collect all chunks for the query later
                all_chunks.extend(chunks)

            print(f" -----> Processed {file_name}")  # Log progress
            total_files += 1

    # Save chroma collection to JSON
    if collection == "chroma":
        with open("chromaCollection.json", "w") as f:
            json.dump(all_docs, f, indent=4)
        print("Chroma Collection saved to chromaCollection.json")

    # After processing all documents for this chunking strategy, query Redis once
    if collection == "redis":
        answer = query("What is a binary search tree?", "redis")  # Query once for the combined text
    elif collection == "chroma":
        answer = query("What is a binary search tree?", "chroma")
    else:
        answer = query("What is a binary search tree?", "mongo")
    elapsed_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Store the result for this chunking strategy, including speed and memory
    store_query_result(chunk_size, overlap, elapsed_time, peak_memory / 1e6, total_chunks, answer, csv_filename)

    return chunk_size, overlap, elapsed_time, peak_memory / 1e6, total_chunks

def mongo_coll(db_name, coll):
    """
    Mongo client connection
    """
    mongo_client = MongoClient(
    "mongodb://jocelyn:abc123@localhost:27018/"
    )
    db = mongo_client[db_name]
    mongo_collection = db[coll]

    return mongo_collection

def store_query_result(chunk_size, overlap, speed, peak_memory, total_chunks, answer, csv_filename):
    """
    Store the query result for a given chunking strategy in a CSV file.
    """
    # Ensure answer only contains ASCII characters
    if answer:
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


def query(query_text: str, query_type = "redis"):
    if query_type == "redis":
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
    elif query_type == "chroma":
        embedding = get_embedding(query_text)
        results = chroma_collection.query(
                query_embeddings=embedding,
            )
        return results
    else:
        pass
        


def main():
    # Set up
    results = []
    csv_filename = "chunking_results.csv"

    use_collection = "mongo"
    clear_store(use_collection)

    if use_collection == "redis":
        create_hnsw_index()
        for chunk_size, overlap in CHUNKING_STRATEGIES:
            chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename, collection="redis")
            results.append([chunk_size, overlap, time_taken, memory_used, num_chunks])

    elif use_collection == "chroma":
        # chroma_collection = chroma_client.create_collection(name="chromaCollection")
        for chunk_size, overlap in CHUNKING_STRATEGIES:
            chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename, collection="chroma")
            results.append([chunk_size, overlap, time_taken, memory_used, num_chunks])
        
    else:
        for chunk_size, overlap in CHUNKING_STRATEGIES:
            chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename, collection="mongo")
            results.append([chunk_size, overlap, time_taken, memory_used, num_chunks])

    # results = []
    # csv_filename = "chunking_results.csv"

    # for chunk_size, overlap in CHUNKING_STRATEGIES:
    #     chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename, collection=use_collection)
    #     results.append([chunk_size, overlap, time_taken, memory_used, num_chunks])

    print("\n---Done processing PDFs---\n")
    #query_redis("What do we have in our arsenal?")


if __name__ == "__main__":
    main()