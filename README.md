# DS4300 - Spring 2025 - Practical #2 - Vector DBs and LLMs

Welcome to team jo-fi's repository for Practical #2 - Vector DBs and LLMs of DS4300 Spring 2025! This guide will walk you through 
the steps to successfully run our program on your own machine. Let's get started!

## Step 0: Setting Up Your Environment
Before we get into the specifics of the project code, begin by downloading all of the files from team jo-fi's project
repository onto your machine. Open the entire project folder in an IDE of your choice (we suggest using PyCharm, but use
the IDE you feel most comfortable with!). Be sure that all packages in the `requirements.txt` file in the project repository are installed on your machine if they are not
already by running the command: 
```bash
pip install -r requirements.txt
```
This ensures the additional packages listed in the `requirements.txt` file are installed on your machine for use in running the project scripts. 

**Important note: You will also need to ensure that both Ollama and Redis are running in order to run our program.** 
* Ollama is listed in the `requirements.txt ` file, so it should have already been installed as an application on your machine; open the application for use in the project code.
* We will use Docker Desktop to ensure Redis is running. If you have not already, create a container for Redis/Redis-Stack using the directions given in class and start it up in Docker Desktop before executing any code in the repository. This is important to ensure you do not get an error! **Take note of the port number you assign to the container as it will be needed later.**

With both Ollama and Redis running, we are now ready to begin running the code!

## Step 1: Ingesting the Documents
With all the necessary packages installed and applications running, we can begin ingesting documents. To do so, open the `ingest.py` file in the project repository. The main purpose of
this file is to evaluate different chunking strategies to analyze their impact on retrieval performance. Before running the script, ensure Redis, MongoDB, and ChromaDB are running:
* Redis: Start a Redis server on port 6380
* MongoDB: Ensure your MongoDB Atlas or local instance is configured correctly (You will need to replace the "user" and "pwd" variables in lines 26 and 27 respectively, as well as adjusting the CONNECTION_STR in line 29 to match your Atlas cluster).
* ChromaDB: No additional setup is required

To run the script, either click the 'run' button in your IDE or type the following in the terminal: 
```bash
python ingest.py
```
**Important note: `ingest.py` must be run BEFORE running any other scripts in the repository (`search.py`, `driver.py`).**

The script will clear existing data in Redis, MongoDB, and ChromaDB and create an HNSW index in Redis for vector search. It will then process all PDF files in the `./Files/` directory
and generate text embeddings using nomic-embed-text or SentenceTransformers. The script will store embeddings in the selected vector database and evaluate and log chunking strategies 
in `chunking_results.csv`. 

**Note: The script assumes that the documents to be ingested are in a folder named `Files` within the project directory. If this is not the case, the 'data_dir' variable name will need
to be updated when running the process_pdfs function in 'main'. Furthermore, if you would like the results to be stored under a different file name, modify the 'csv_filename' variable in 'main'.**

This script uses the example chunk sizes and chunk overlap sizes given in the Practical 2 instructions. To modify the chunking strategies or embedding models, update the 'CHUNKING_STRATEGIES' and 'EMBEDDING_TYPE' global variables in the script. 
You can also change the target database (mongo, redis, or chroma) by adjusting the coll parameter in the process_pdfs function in 'main'.

#### Output
After running the `ingest.py` script, processed embeddings will be stored in the specified vector database, and performance metrics (processing time, memory usage, total chunks) are recorded in `chunking_results.csv` (unless named otherwise).

## Step 2: Querying the RAG Search Interface
Now that all documents have been ingested and indexed using an embedding type and vector database, we can interact with our RAG search interface through queries. This interaction will be
facilitated by `driver.py` which, as the name suggests, is a driver Python script to execute various versions of the indexing pipeline and to collect important data about the process (memory, time, etc).
We will ultimately use this data to make a recommendation for which pipeline works best as we compare various pipeline performances by modifying different variables. 

To initiate the RAG search interface, open the `driver.py` file in the project repository. Once again, before running the script, ensure Redis, MongoDB, and ChromaDB are running. Redis and ChromaDB do not need additional adjustments, but MongoDB is using Read-Only permissions for the creator's Atlas Cluster, and thus will need to be changed in `search.py`, lines 27-29, if you have ingested the files and want to use your own.

To run the script, either click the 'run' button in your IDE or type the following in the terminal: 
```bash
python driver.py
```
This will launch an interactive search interface where you can enter queries and retrieve relevant documents from stored embeddings.

To perform a search, enter the search query when prompted. The system will then ask you if you would like to compare one or more of the following: LLMs, system prompts, vector databases, or embedding types.
If you enter 'yes', you will be prompted to decide whether you would like to return a response to your query in addition to the comparison file. You will then note which variables you would like to compare and enter a filename for the .csv results file that will be generated in the project repository folder. After the results file is created, if you answered 'yes' to the response, you will be prompted to enter the LLM, system prompt, vector database, and embedding type you would like to use for your search. The search is then executed using the selected variables and the final response is returned.
If you enters 'no', you will be prompted to enter the LLM, system prompt, vector database, and embedding type you would like to use for your search. The search is then executed using the selected variables
and the final response is returned.

If at any time you would like to leave the RAG search interface, you can simply type 'exit' and the system will be terminated. You can ask the system as many questions as you would like while in the RAG search interface system.

You can modify the available LLMs, embedding types, vector databases, and system prompts in the script as needed (found in the 'model_options', 'embedding_options', 'vectordb_options', and 'SYSTEM_PROMPT_VARIATIONS' variables, respectively).

As mentioned above, it was important to run the `ingest.py` script first as this gives the interface relevant context to inform its responses.

#### Output
The `driver.py` script returns ranked search results with relevant document embeddings and generates responses using a selected LLM, embedding type, vector database, and system prompt. If benchmarking is enabled, retrieval performance metrics are saved to a user-specified .csv file for further analysis.

**Note: You may have noticed a third Python file in the project repository titled `search.py`. While we don't need to explicitly run this file, it is important because it contains many functions that will allow us to systematically vary the embedding models, prompt tweaks, choice of vector database, and choice of LLM used to search documents.**

After completing this step, we have successfully built a local RAG system that achieves the following:
* Ingests a collection of documents that represent material, such as course notes, we have collected throughout the semester. 
* Indexes those documents using embedding and a vector database.
* Accepts a query from the user. 
* Retrieves relevant context based on the user’s query.
* Packages the relevant context up into a prompt that is passed to a locally-running LLM to generate a response.

Feel free to ask the RAG system any questions you may have about DS4300 content! 

## Other Project Files
You may have noticed some folders/files were seemingly not used or mentioned in this guide. This includes the `.idea` folder, the `__pycache__` folder, and the `.DS_Store` file. While not explicitly mentioned, all of these files may be important in ensuring the project files run successfully behind the scenes. Therefore, be sure they are also properly loaded when loading in the project repository and do not delete them!

## Thank You
Thank you for reaching the end of this guide on how to execute the files within team jo-fi's project repository for
Practical #2 - Vector DBs and LLMs! Feel free to reach out to us with any questions not answered in this guide if you need assistance
running the project files. Happy searching!
