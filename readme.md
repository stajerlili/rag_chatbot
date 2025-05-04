# RAG based chatbot with LangGraph

## Setup 

1. I created a conda environment (I suppose it should work with regular venv as well, but I prefer conda to set the python version):
```shell
conda create -p ./rag_env python=3.10
```

2. Activated the env

3. Installed dependencies by running
```shell
pip install -r requirements.txt
```

4. You can run the program by:
```
python rag_advanced.py
```

## Documentation

I didn't use a jupyter notebook for development, I hope that is not a problem, I am just more comfortable with regular py files. 
Also at first I accidentally completely disregarded that the framework to be used is LangGraph, therefore I have a simpler solution in the form of rag.py.

### Architecture
The architecture I used is pretty standard based on my knowledge of RAG augmented LLMs. 
#### Document processing
This part handles the loading, processing and storing od documents. In this implementations I used a couple of wikipedia pages as an example. They are then split into chunks of 250 character length with 50 characters overlapping with the RecursiveCharacterTextSplitter. This particular method supposedly tries to keep text belonging together as one part as much as possible which is good for semantic understanding ([link](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/))
#### Embedding creation
In this section the code converts the split documents into vectors for efficient similarity searches. 
1. I used a sentence-trasformes model called paraphrase-MiniLM-L3-v2 . This is a very small model and could fit my machine's COU as well, but it is interchangable which is good for scalability. It maps chunks to a 384 dimensional vector space.
2. The main reason I used FAISS is because I have already worked with it previously and it has a gpu optimized version. The from_documents method is the one used for indexing.
#### Agent

This part coordinates the workflow and decision-making of the agent by using a state graph.
The system follows the following workflow:

1. Information retrieval: receives query and retrieves documents from the vector database.
2. Context integration: documents are formatted into a context for the LLM.
3. Response generation: the LLM generates a response based on the query and retrieved context.
4. Final answer is returned to the user.

#### Integrated LLM
Again opted for a very small model for GPU reasons.

### Hyperparameters
Tunable for better performance(but I did not perform this):
- Chunk size (250 characters)
- Chunk Overlap (50 characters)
- Retrieval Count (k=3)

Models selected for GPU constraint:
- TinyLlama-1.1B as LLM
- paraphrase-MiniLM-L3-v2 as sentence tarnsformer

### Testing
I added some test queries to see how the system performs:
```
test_queries = [
    "What is the longest English word?",  
    "What is an inherently funny word?",  
    "Where does the fan death theory originate from?",  
    "What is the capital of France?",  
    "Explain what Retrieval-Augmented Generation is" 
]
```
What I want to test with these:

The first question shouldn't work as the longest English word is not within the given chunk size --> indeed not, it usually returned "strenth" as the answer for me, altough due to temperature sometimes different answer.

Second --> RAG based good answer expected

Third question --> I commented this from knowledge base, so only hallucinating answer is expected and for me that is the case.

Fourth --> Good answer from base model probably


### Possible improvements, bottlenecks, performance measures

Current limitations were:
- the model sizes, I couldn't use anything bigger
- the fixed number of retrieved docs
- autonomy of agent is not really high

Improvement ideas:
- analyze the query before deciding retrieval number or strategy (wheter it is even needed)
- more complex queries might need subtask breakdown
- scaling for more document input (altough not exactly sure how)

Performance measures:
- time of answering or even time of subtasks such as embedding time, retrieval, etc.
- did it retrieve the right documents? --> adding the ground truth necessary documents for each test query
- how accurately did it answer? 