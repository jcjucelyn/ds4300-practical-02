import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import time
import tracemalloc
import gc
import csv

from torch.backends.cudnn import benchmark

# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Sample system prompts for comparison
SYSTEM_PROMPT_VARIATIONS = [
    "You are a helpful AI assistant. Use the following context to answer the query as accurately as possible. If the context is not relevant to the query, say 'I don't know'.",
    "You are an expert in technical writing and software engineering.",
    "You are a professor explaining concepts to a student.",
    "You are a creative storyteller.",
    "You are a concise and direct AI, providing brief answers."
]


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

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

        # Print results for debugging
        # for result in top_results:
        #     print(
        #         f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
        #     )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, model_name="mistral:latest", system_prompt=SYSTEM_PROMPT_VARIATIONS[0]):
    # Handle case where no relevant context is found
    if not context_results:
        return "I couldn't find relevant information. Try rephrasing your query."

    # Prepare context string
    context_str = ""
    for result in context_results:
        file = result.get('file', 'Unknown file')
        page = result.get('page', 'Unknown page')
        chunk = result.get('chunk', 'Unknown chunk')
        similarity = result.get('similarity', 0)  # Default to 0 if similarity is missing

        context_str += f"From {file} (page {page}, chunk {chunk}) with similarity {float(similarity):.2f}\n"

    print(f"\nUsing model: {model_name}")
    print(f"context_str: {context_str}")

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

    print(f"Execution time: {execution_time:.2f} sec")
    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")

    return response, execution_time, peak_memory_mb


def get_user_preferences():
    """ Ask the user whether to compare models, system prompts, both, or neither."""
    compare_models = input("\nDo you want to compare multiple LLMs? (yes/no): ").strip().lower() == "yes"
    if not compare_models:
        print("Benchmarking with mistral:latest model ONLY.")
    compare_prompts = input("\nDo you want to compare multiple system prompts? (yes/no): ").strip().lower() == "yes"
    if not compare_prompts:
        print("Benchmarking with default system prompt ONLY ('You are a helpful AI assistant. Use the following context to answer the query as accurately as possible. If the context is not relevant to the query, say 'I don't know'.').")
    return compare_models, compare_prompts


def compare_llms_and_prompts(query, context_results, model_names, compare_models, compare_prompts, output_file="query_results.csv"):
    """ Compare multiple LLMs and save results to a CSV file."""

    # Determine variations based on user choices
    models_to_test = model_names if compare_models else [model_names[0]]
    system_prompts = SYSTEM_PROMPT_VARIATIONS if compare_prompts else [SYSTEM_PROMPT_VARIATIONS[0]]

    results = []

    for model_name in models_to_test:
        for system_prompt in system_prompts:
            response, response_time, memory_used = generate_rag_response(
                query, context_results, model_name, system_prompt
            )

            results.append({
                "LLM": model_name,
                "System Prompt": system_prompt,
                "Speed (s)": round(response_time, 3),
                "Memory (MB)": round(memory_used, 3),
                "Response": response
            })

            print(f"Tested {model_name} with system prompt '{system_prompt}' "
              f"in {response_time:.3f}s, using {memory_used:.3f}MB memory.")

    # Check if the output file ends with ".csv"
    if not output_file.endswith(".csv"):
        print("Warning: The provided filename doesn't end with '.csv'. Appending '.csv'.")
        output_file += ".csv"  # append ".csv" if not already present

    # Save results to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["LLM", "System Prompt", "Speed (s)", "Memory (MB)", "Response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        benchmark_choice = input(
            "\nDo you want to compare multiple LLMs (mistral:latest, gemma3:1b, and llama3.2) and/or system prompts? (yes/no): ").strip().lower()

        if benchmark_choice == "yes":
            compare_models, compare_prompts = get_user_preferences()

            # Check if both compare_models and compare_prompts are 'no'
            if not compare_models and not compare_prompts:
                print(
                    "\nYou did not select the proper criteria for comparing multiple LLMs and/or system prompts. One LLM and one prompt will be used instead.")
                # Execute the else code: select the default LLM and prompt without benchmarking
                model_name = input(
                    "\nEnter the LLM to use (mistral:latest, gemma3:1b, or llama3.2): ").strip() or "mistral:latest"
                print("\nAvailable system prompts:")
                for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                    print(f"{i + 1}. {prompt}")
                prompt_index = input(
                    f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
                system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                    prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]
            else:
                filename = input(
                    "\nBenchmarking results will be stored in a .csv file for easy comparison. What would you like to name the file? ")
                # Define models to benchmark
                models_to_test = ["mistral:latest", "gemma3:1b", "llama3.2"]  # Add more models as needed
                compare_llms_and_prompts(query, context_results, models_to_test, compare_models, compare_prompts, output_file=filename)

                # Ask user which model they want for the final response
                model_name = input(
                    "Which model do you want to use for the final response? (mistral:latest, gemma3:1b, or llama3.2) ").strip() or "mistral:latest"

                # Let user choose a system prompt if they are benchmarking prompts
                print("\nAvailable system prompts:")
                for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                    print(f"{i + 1}. {prompt}")
                prompt_index = input(
                    f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
                system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                    prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]
        else:
            model_name = input(
                "\nEnter the LLM to use (mistral:latest, gemma3:1b, or llama3.2): ").strip() or "mistral:latest"
            print("\nAvailable system prompts:")
            for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                print(f"{i + 1}. {prompt}")
            prompt_index = input(
                f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
            system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]

            # Generate the final response with the chosen model
        final_response, _, _ = generate_rag_response(query, context_results, model_name=model_name,
                                                     system_prompt=system_prompt)

        print("\n--- Final Response ---")
        print(final_response)


# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()