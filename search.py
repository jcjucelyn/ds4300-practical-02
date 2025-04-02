"""
Sophie Sawyers and Jocelyn Ju
DS4300 || Practical 2

search.py : A Python file to systematically vary the chunking strategies, embedding models, various prompt tweaks,
choice of Vector DB, and choice of LLM used to search documents
"""
# Import necessary packages
import chromadb
import csv
import gc
import ollama
import numpy as np
import redis
import time
import json
import tracemalloc
from pymongo import MongoClient
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# Set up redis and chroma client connections
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)
chroma_client = chromadb.Client()

# Initialize PyMongo connection through limited role
user = "ds4300_staff"
pwd = "staffStaff4300"
CONNECTION_STR = f"mongodb+srv://{user}:{pwd}@cluster0.dhzls.mongodb.net/"
mongo_client = MongoClient(
    CONNECTION_STR
)
db = mongo_client["4300-pracB"]
mongo_collection = db["mongoCollection"]

# Define global variables
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# Define sample system prompts for comparison
SYSTEM_PROMPT_VARIATIONS = [
    "You are a helpful AI assistant. Use the following context to answer the query as accurately as possible. If the context is not relevant to the query, say 'I don't know'.",
    "You are an expert in technical writing and software engineering.",
    "You are a professor explaining concepts to a student.",
    "You are a creative storyteller.",
    "You are a concise and direct AI, providing brief answers."
]

# Define function to check validity of a user's input
def check_validity(user_input, acceptable_list, system=False):
    """ Check the validity of a user's input. Supports both single selections and multiple comma-separated selections.
    Handles cases where input is already a list."""
    # If user_input is already a list, just validate each item in the list
    if isinstance(user_input, list):
        invalid_selections = [s for s in user_input if s not in acceptable_list]
    else:
        # Split input if multiple values are provided
        selections = [s.strip() for s in user_input.split(",")]
        invalid_selections = [s for s in selections if s not in acceptable_list]

    if invalid_selections:
        if not system:
            print(f"Invalid entries: {', '.join(invalid_selections)}")
            new_input = input(f"Please choose from: {acceptable_list}: ")
            return check_validity(new_input, acceptable_list) # Recursively prompt again
        else:
            print(f"Invalid entries: {', '.join(invalid_selections)}")
            for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                print(f"{i + 1}. {prompt}")
            prompt_index = input(
                f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
            new_input = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]
            
            return check_validity(new_input, acceptable_list, True) # Recursively prompt again

    return user_input if isinstance(user_input, list) else selections[0] # Return list if multiple, else single value
    
# Define function to generate an embedding using nomic-embed-text, all-MiniLM-L6-v2, or all-mpnet-base-v2
def get_embedding(text: str, model: str="nomic-embed-text") -> list:
    # Access through ollama (nomic-embed) or SentenceTransformer(other)
    if model=="nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        
    else:
        response = SentenceTransformer(model)

    return response["embedding"]

# Define function to search embeddings
def search_embeddings(query, emb_type="nomic-embed-text", collection="redis", top_k=3, chroma_coll=None):
    # Get the embedding of the query
    query_embedding = get_embedding(text=query, model=emb_type)

    if collection == "redis":    
        # Convert embedding to bytes for Redis search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Construct the vector similarity search query
            # Use a more standard RediSearch vector search syntax
            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "file", "page", "chunk", "vector_distance")
                .dialect(2)
            )

            # Perform the search
            results = redis_client.ft(INDEX_NAME).search(
                q, query_params={"vec": query_vector}
            )

            # Handle case where no results are found
            if not results.docs:
                return []

            # Transform results into the expected format
            top_results = [
                {
                    "file": result.file,
                    "page": result.page,
                    "chunk": result.chunk,
                    "similarity": result.vector_distance,
                }
                for result in results.docs
            ][:top_k]

            return top_results

        except Exception as e:
            print(f"Search error: {e}")
            return []
        
    elif collection =="chroma":
        try:
            # Extract results
            results = chroma_coll.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            # Handle case where no results are found
            if not results["documents"]:
                return []
            elif results["documents"] == [[]]:
                return []
            
            # Transform results into the expected format
            else:
                # Initialize storage
                top_results = []
                
                # Iterate through the results
                for i in range(len(results["documents"][0])):
                    document = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Metadata
                    file = metadata["file"]
                    page = metadata["page"]
                    chunk = metadata["chunk"]
                   
                    # Append to the storage list
                    top_results.append({
                        "file":file,
                        "page":page,
                        "chunk":chunk,
                        "similarity":distance
                    })

                return top_results[:top_k]

        except Exception as e:
            print(f"Search error: {e}")
            return []
    else: # collection == mongo
        try:
            candidates = mongo_collection.count_documents({})

            # Extract the results
            results = db.mongoCollection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "pracB_searchindex",
                        "limit": top_k,
                        "numCandidates": candidates,
                        "path": "embedding",
                        "queryVector": query_embedding
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "file": 1,
                        "page":1,
                        "chunk":1,
                        "similarity": {"$meta":"vectorSearchScore"}
                        
                    }
                }])
                        
            # Handle case where no results are found
            if not results:
                return []
            elif results == [[]]:
                return []   
            else: 
                # Transform results into the expected format
                top_results = [
                    {
                        "file": result["file"],
                        "page": result["page"],
                        "chunk": result["chunk"],
                        "similarity": result["similarity"],
                    }
                    for result in results
                ][:top_k]

                return top_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

# Define function to generate rag response
def generate_rag_response(query, context_results, model_name="mistral:latest", system_prompt=SYSTEM_PROMPT_VARIATIONS[0]):
    
    # Handle case where no relevant context is found
    if not context_results:
        return "I couldn't find relevant information. Try rephrasing your query.", 0, 0

    # Prepare context string
    context_str = ""
    for result in context_results:
        file = result.get('file', 'Unknown file')
        page = result.get('page', 'Unknown page')
        chunk = result.get('chunk', 'Unknown chunk')
        similarity = result.get('similarity', 0) # Default to 0 if similarity is missing

        context_str += f"From {file} (page {page}, chunk {chunk}) with similarity {float(similarity):.2f}\n"

    # Construct prompt with context
    prompt = f"""{system_prompt}

Context:
{context_str}

Query: {query}

Answer:"""

    # Force garbage collection before tracking (forces Python to clear unused memory before tracking)
    gc.collect()

    # Track performance metrics
    tracemalloc.start()
    start_time = time.time()

    # Generate response using Ollama
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])["message"]["content"]
    except Exception as e:
        response = f"Error generating response: {e}"

    end_time = time.time()
    execution_time = end_time - start_time

    # Record peak memory usage
    peak_memory = tracemalloc.get_traced_memory()[1]  # in bytes
    tracemalloc.stop()

    # Convert memory usage to MB
    peak_memory_mb = peak_memory / (1024 * 1024)

    # Remove non-ASCII characters from the response using encode and decode
    response = response.encode("ascii", "ignore").decode("ascii")

    return response, execution_time, peak_memory_mb

# Define function to get user preferences for benchmarking/search
def get_user_preferences():
    """ Ask the user whether and what to compare."""
    compare_vdbs = input("\nDo you want to compare multiple vector databases? (yes/no): ").strip().lower() == "yes"
    if not compare_vdbs:
        print("Benchmarking with default vector database ONLY ('redis').")
    compare_embeddings = input("\nDo you want to compare multiple embedding types? (yes/no): ").strip().lower() == "yes"
    if not compare_embeddings:
        print("Benchmarking with default embedding type ONLY ('nomic-embed-text').")
    compare_models = input("\nDo you want to compare multiple LLMs? (yes/no): ").strip().lower() == "yes"
    if not compare_models:
        print("Benchmarking with default LLM model ONLY ('mistral:latest').")
    compare_prompts = input("\nDo you want to compare multiple system prompts? (yes/no): ").strip().lower() == "yes"
    if not compare_prompts:
        print("Benchmarking with default system prompt ONLY ('You are a helpful AI assistant. Use the following context to answer the query as accurately as possible. If the context is not relevant to the query, say 'I don't know'.').")

    return compare_models, compare_prompts, compare_vdbs, compare_embeddings

# Define function to compare multiple variables and save results
def compare_all(query, model_names, compare_models, vdb_names, compare_vdbs, embedding_names, compare_embeddings, compare_prompts, output_file="query_results.csv"):
    """ Compare multiple LLMs, prompts, vector databases, and embedding types and save results to a CSV file."""
    # Determine variations based on user choices, inputted separated by / or all (enter)
    if compare_models:
        mods_to_test = input(
            f"Which models of: {model_names} would you like to compare? Input separated by /, or enter for all. "
            ).split("/")
        models_to_test = mods_to_test if mods_to_test != [''] else model_names
        # Ensure the entry is valid
        models_to_test = check_validity(models_to_test, model_names)

    else:
        models_to_test = [model_names[0]]

    if compare_embeddings:
        embs_to_test = input(
            f"Which embedding types of: {embedding_names} would you like to compare? Input separated by /, or enter for all. "
        ).split("/")
        embeddings_to_test = embs_to_test if embs_to_test != [''] else embedding_names

        # Ensure the entry is valid
        embeddings_to_test = check_validity(embeddings_to_test, embedding_names)
    else:
        embeddings_to_test = [embedding_names[0]]

    if compare_vdbs:
        vdbs_to_test = input(
            f"Which vector database types of: {vdb_names} would you like to compare? Input separated by /, or enter for all. "
        ).split("/")
        vectordbs_to_test = vdbs_to_test if vdbs_to_test != [''] else vdb_names
        
        # Ensure the entry is valid
        vectordbs_to_test = check_validity(vdbs_to_test, vdb_names)
    else:
        vectordbs_to_test = [vdb_names[0]]

    if compare_prompts:
        sys_prompts = input(
            f"Which system prompt of: {SYSTEM_PROMPT_VARIATIONS} would you like to compare? Input separated by / or enter for all. "
        ).split("/") or SYSTEM_PROMPT_VARIATIONS
        system_prompts = sys_prompts if sys_prompts != [''] else SYSTEM_PROMPT_VARIATIONS
        # Ensure the entry is valid
        system_prompts = check_validity(system_prompts, SYSTEM_PROMPT_VARIATIONS)
    else:
        system_prompts = [SYSTEM_PROMPT_VARIATIONS[0]]

    results = []

    # Initialize Chroma if it is in the test list
    if "chroma" in vectordbs_to_test:
        # Access Chroma Collection from file (run ingest.py first)
        try:
            with open("chromaCollection.json", "r") as f:
                chroma_data = json.load(f)
        except Exception as e:
            chroma_data = {}
            print(f"Error reading JSON: {e}")

        # Turn Chroma JSON into a collection
        chroma_collection = chroma_client.create_collection(name="chromaCollection")
        ids = list(item["id"] for item in chroma_data)
        embs = [item["embedding"] for item in chroma_data]
        metadata = [{"file": item["file"],
                    "page": item["page"],
                    "chunk": item["chunk"]} for item in chroma_data]

        # Add file to chroma_collection
        chroma_collection.add(
            ids=ids,
            documents=[item["chunk"] for item in chroma_data],
            metadatas=metadata,
            embeddings=embs
        )

    print('\nLoading...\n')

    # Iterate through choices
    for model_name in models_to_test:
        for system_prompt in system_prompts:
            for vdb_name in vectordbs_to_test:
                # Generate context results
                if vdb_name == "chroma":
                    context_results = search_embeddings(query, collection=vdb_name, chroma_coll=chroma_collection)
                else:
                    context_results = search_embeddings(query, collection=vdb_name)

                for embedding_type in embeddings_to_test:
                    response, response_time, memory_used = generate_rag_response(
                    query, context_results, model_name, system_prompt
                    )

                    results.append({
                        "LLM": model_name,
                        "Vector DB": vdb_name,
                        "Embedding Type": embedding_type,
                        "System Prompt": system_prompt,
                        "Speed (s)": round(response_time, 3),
                        "Memory (MB)": round(memory_used, 3),
                        "Response": response
                        })

    # Check if the output file ends with ".csv"
    if not output_file.endswith(".csv"):
        print("Warning: The provided filename doesn't end with '.csv'. Appending '.csv'.")
        output_file += ".csv"  # append ".csv" if not already present

    # Save results to CSV
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["LLM", "Vector DB", "Embedding Type", "System Prompt", "Speed (s)", "Memory (MB)", "Response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerows(results)

    print(f"\nCompared models: {models_to_test} \nCompared prompts: {system_prompts}\nCompared embeddings: {embeddings_to_test} \nCompared vector databases: {vectordbs_to_test}")
    print(f"\nResults saved to {output_file}")
