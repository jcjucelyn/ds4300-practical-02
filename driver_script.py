"""
Sophie Sawyers and Jocelyn Ju
DS4300 || Practical B

driver_script.py : a python file to run and collect data on various pipelines
"""

import time
import csv
import json
from search_trial import search_embeddings, chroma_client, get_user_preferences, compare_all, generate_rag_response, SYSTEM_PROMPT_VARIATIONS, check_validity

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        bench = input("\nWould you like to compare one or more of the following: LLMs, system prompts, vector databases, or embedding types? (yes/no) ")

        benchmark_choice = bench.strip().lower() == "yes"

        # Define models to benchmark
        model_options = ["mistral:latest", "gemma3:1b", "llama3.2"]  # Add more models as needed
        embedding_options = ["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        vectordb_options = ["redis", "chroma", "mongo"]

        # If user would like to compare 1+ of the categories, figure out what to compare
        if benchmark_choice:

            # Determine what to compare
            compare_models, compare_prompts, compare_vdbs, compare_embeddings = get_user_preferences()

            # Check if all compare_models and compare_prompts and compare_vdbs and compare_embeddings are 'no'
            if not compare_models and not compare_prompts and not compare_vdbs and not compare_embeddings:
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

            # If 1+ of them is true, ask user    
            else:
                pass
                # Search for relevant embeddings given choice of vector db
                collection_choice = input("\nWhat vector database would you like to use? (redis/chroma/mongo): ").strip().lower() or "redis"

                if collection_choice == "redis":
                    context_results = search_embeddings(query, collection=collection_choice)
                elif collection_choice == "chroma":
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

                    context_results = search_embeddings(query, collection=collection_choice, chroma_coll=chroma_collection)
                elif collection_choice == "mongo":
                    context_results = search_embeddings(query, collection=collection_choice)
                else:
                    print(f"Invalid collection choice: {collection_choice}. Please select one of: redis/chroma/mongo")

                
                filename = input(
                    "\nBenchmarking results will be stored in a .csv file for easy comparison. What would you like to name the file? ")
                
                # # Define models to benchmark
                # model_options = ["mistral:latest", "gemma3:1b", "llama3.2"]  # Add more models as needed
                # embedding_options = ["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
                # vectordb_options = ["redis", "chroma", "mongo"]

                # if compare_models:
                #     models_to_test = list(input(
                #         f"Which models of: {model_options} would you like to compare? Input as a list of strings, or enter for all. "
                #     )) or model_options
                # else:
                #     models_to_test = ["mistral:latest"]

                # if compare_embeddings:
                #     embeddings_to_test = list(input(
                #         f"Which embedding types of: {embedding_options} would you like to compare? Input as a list of strings, or enter for all. "
                #     )) or embedding_options
                # else:
                #     embeddings_to_test = ["nomic-embed-text"]

                # if compare_vdbs:
                #     vectordbs_to_test = list(input(
                #         f"Which vector database types of: {vectordb_options} would you like to compare? Input as a list of strings, or enter for all. "
                #     )) or vectordb_options
                # else:
                #     vectordbs_to_test = ["redis"]

                compare_all(query, context_results, model_options, compare_models, vectordb_options, compare_vdbs, embedding_options, compare_embeddings, compare_prompts, output_file=filename)

                # # Ask user which model they want for the final response
                # model_name = input(
                #     "Which model do you want to use for the final response? (mistral:latest, gemma3:1b, or llama3.2) ").strip() or "mistral:latest"

                # Let user choose a system prompt if they are benchmarking prompts
                # print("\nAvailable system prompts:")
                # for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
                #     print(f"{i + 1}. {prompt}")
                # prompt_index = input(
                #     f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
                # system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                #     prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]
        else:

            # Search for relevant embeddings given choice of vector db
            collection_choice = input("\nWhat vector database would you like to use? (redis/chroma/mongo): ").strip().lower()
            embedding_choice = input("\nWhat embedding type would you like to use? (nomic-embed-text, all-MiniLM-L6-v1, all-mpnet-base-v2): ")

            model_name = check_validity(llm_name, model_options)
            if collection_choice == "redis":
                context_results = search_embeddings(query, emb_type=embedding_choice, collection=collection_choice)
            elif collection_choice == "chroma":
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

                context_results = search_embeddings(query, collection=collection_choice, chroma_coll=chroma_collection)
            elif collection_choice == "mongo":
                context_results = search_embeddings(query, collection=collection_choice)
            else:
                print(f"Invalid collection choice: {collection_choice}. Please select one of: redis/chroma/mongo")
            
            # Ask user which model they want for the final response
            # model_name = input(
            #         "Which model do you want to use for the final response? (mistral:latest, gemma3:1b, or llama3.2) ").strip() or "mistral:latest"

            # # Let user choose a system prompt if they are benchmarking prompts
            # print("\nAvailable system prompts:")
            # for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
            #     print(f"{i + 1}. {prompt}")
            # prompt_index = input(
            #     f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
            # system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
            #     prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]
            
            llm_name = input(
                "\nEnter the LLM to use (mistral:latest, gemma3:1b, or llama3.2): ").strip() or "mistral:latest"
            print("\nAvailable system prompts:")

            model_name = check_validity(llm_name, model_options)


            # for i, prompt in enumerate(SYSTEM_PROMPT_VARIATIONS):
            #     print(f"{i + 1}. {prompt}")
            prompt_index = input(
                f"Select a system prompt (1-{len(SYSTEM_PROMPT_VARIATIONS)}) or press Enter to use the first: ").strip()
            system_prompt = SYSTEM_PROMPT_VARIATIONS[int(prompt_index) - 1] if prompt_index.isdigit() and 1 <= int(
                prompt_index) <= len(SYSTEM_PROMPT_VARIATIONS) else SYSTEM_PROMPT_VARIATIONS[0]

            # Generate the final response with the chosen model
            final_response, _, _ = generate_rag_response(query, context_results, model_name=model_name,
                                                     system_prompt=system_prompt)

            print("\n--- Final Response ---")
            print(final_response)

if __name__ == "__main__":
    interactive_search()