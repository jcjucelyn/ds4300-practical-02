"""
Sophie Sawyers and Jocelyn Ju
DS4300 || Practical 2

driver.py : A driver Python script to execute various versions of the indexing pipeline and to collect
important data about the process (memory, time, etc.)
"""
# Import necessary packages
import json
from search import search_embeddings, chroma_client, get_user_preferences, compare_all, generate_rag_response, \
    SYSTEM_PROMPT_VARIATIONS, check_validity

# Define function for interactive search interface
def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' when prompted for a search query to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        # Identify whether the user would like to benchmark or not
        print("\nNOTE: The default pipeline is as follows:\n*LLM: mistral:latest\n"
              "*System prompt: You are a helpful AI assistant. Use the following context to answer the query as accurately "
              "as possible. If the context is not relevant to the query, say 'I do not know'.\n*Vector database: Redis\n"
              "*Embedding type: nomic-embed-text\n\nIf you choose to benchmark pipeline variations, selecting 'no' to"
              " comparing a certain variable will cause the program to select the default option for comparison.\nIf you "
              "would like to measure the performance of a certain model while benchmarking (even if you are only interested"
              " in that one model for the respective variable \n(ex: chroma for vector DB)), select 'yes' to comparing the "
              "variable and type in your model selection to be included in the benchmarking results.")
        print("\nONLY IF you would like to use ONE model for each variable should you answer 'no' to the following question.")
        bench = input(
            "\nWould you like to compare one or more of the following: LLMs, system prompts, vector databases, or embedding types? (yes/no) ")

        # Initialize asking for a query
        query_eval = True
                
        benchmark_choice = bench.strip().lower() == "yes"

        # Define options to benchmark
        model_options = ["mistral:latest", "gemma3:1b", "llama3.2"]  # Add more models as needed
        embedding_options = ["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        vectordb_options = ["redis", "chroma", "mongo"]

        # If user would like to compare 1+ of the categories, figure out what to compare
        if benchmark_choice:
            # Identify whether the user would like to run the query to return a value in addition to benchmarking
            query_eval = input(
                "\nWould you like to generate a response to your query in addition to benchmarking? (yes/no) ").strip().lower() == "yes"

            # Determine what to compare
            compare_models, compare_prompts, compare_vdbs, compare_embeddings = get_user_preferences()

            # Check if all compare_models and compare_prompts and compare_vdbs and compare_embeddings are 'no'
            if not compare_models and not compare_prompts and not compare_vdbs and not compare_embeddings:
                print(
                    "\nYou did not select the proper criteria for comparing multiple LLMs, vector DBs, embedding types, and/or system prompts.")
                print("Restarting the search process...")
                continue  # Restart the loop from the beginning (asking for the search query)

            # If 1+ of them is true, ask user for filename to save to, then compare all variations selected
            else:
                filename = input(
                    "\nBenchmarking results will be stored in a .csv file for easy comparison. What would you like to name the file? ")

                compare_all(query, model_options, compare_models, vectordb_options, compare_vdbs,
                            embedding_options, compare_embeddings, compare_prompts, output_file=filename)
        # else:
        if query_eval:
            print("\n----------\nThe following questions are for generating the answer to your query (select ONE option only):\n----------\n")
            # Search for relevant embeddings given choice of vector db
            collection_choice = input(
                "What vector database would you like to use for answer generation? (redis/chroma/mongo): ")
            embedding_choice = input(
                "\nWhat embedding type would you like to use for answer generation? (nomic-embed-text, all-MiniLM-L6-v1, all-mpnet-base-v2): ")

            if collection_choice == "redis" or collection_choice == "mongo":
                context_results = search_embeddings(query, emb_type=embedding_choice, collection=collection_choice)

            elif collection_choice == "chroma":
                # Access Chroma Collection from file (run ingest.py first)
                try:
                    with open("chromaCollection.json", "r") as f:
                        chroma_data = json.load(f)
                except Exception as e:
                    chroma_data = {}
                    print(f"Error reading JSON: {e}")

                # Turn Chroma JSON into a collection if it doesn't already exist
                try:
                    chroma_collection = chroma_client.get_collection(name="chromaCollection")
                    print("Chroma collection already exists. Using existing collection.")
                except Exception:
                    print("Creating new Chroma collection...")
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

                context_results = search_embeddings(query, emb_type=embedding_choice, collection=collection_choice,
                                                    chroma_coll=chroma_collection)
            else:
                print(f"Invalid collection choice: {collection_choice}. Please select one of: redis/chroma/mongo")

            llm_name = input(
                "\nWhat LLM would you like to use for answer generation? (mistral:latest, gemma3:1b, or llama3.2): ").strip() or "mistral:latest"

            # Certify model name
            model_name = check_validity(llm_name, model_options)

            # Ask the user to select a prompt
            print("\nAvailable system prompts:")
            for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                print(f"{i + 1}. {prompt}")
            prompt_index = input(
                f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
            system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]

            print("\nLoading...\n")

            # Generate the final response with the chosen model
            final_response, _, _ = generate_rag_response(query, context_results, model_name=model_name,
                                                         system_prompt=system_prompt)

            print("\n--- Final Response ---")
            print(final_response)


def main():
    # Run interactive search
    interactive_search()


if __name__ == "__main__":
    main()