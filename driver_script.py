"""
Sophie Sawyers and Jocelyn Ju
DS4300 || Practical B

driver_script.py : a python file to run and collect data on various pipelines
"""

import time
import csv
from ingest import process_pdfs, CHUNKING_STRATEGIES

# File save names
CHUNK_CSV = "chunk_stats.csv"
GENERAL_CSV = "general_stats.csv"

# 

def process_all_interface():
    # Initializations
    all_embs = ["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    all_vdbs = ["redis", "chroma", "mongo"]
    chunk_csv = []

    """Interactive search interface."""
    print("Processing Interface")
    print("Type 'exit' to quit")

    while True:
        run_all = input("\nRun all options? (yes/no): ")
        if run_all.lower() == "exit":
            break

        if run_all == "yes":
            pass

        else:


            # if emb_mod == "all":
            #     for emb in all_embs:
            #         if vec_db == "all":
            #             for chunk_size, overlap in CHUNKING_STRATEGIES:
            #                 for vdb in all_vdbs:
            #                     chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename="chunking_results.csv", coll=vec_db, emb=emb)

            #                     # Store in list
            #                     chunk_csv.append([vec_db, emb, chunk_size, overlap, time_taken, memory_used, num_chunks])
            #         else:
            #             chunk_size, overlap, time_taken, memory_used, num_chunks = process_pdfs("../Files/", chunk_size, overlap, csv_filename="chunking_results.csv", coll=vec_db, emb=emb)

            # # Selected embedding type
            # else:
            #     pass


    print("Processing completed, and results saved!")



if __name__ == "__main__":
    process_all_interface()