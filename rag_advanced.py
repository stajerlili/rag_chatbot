from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langgraph.graph import StateGraph, END#, START

import torch


class Agent:
    def __init__(self, vector_db, llm, raw_pipeline):
        # self.vector_db = vector_db
        self.llm = llm
        self.raw_pipeline = raw_pipeline
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    def retrieve_info(self, state):
        docs = self.retriever.get_relevant_documents(state["query"]) # depricated method but didn't have time to change
        search_results = "\n\n".join([doc.page_content for doc in docs])
        
        return {**state, "search_results": search_results}
    
    def generate(self, state):
        prompt = f"""
        <human>: Use this information to answer the question.      
        Information: {state["search_results"]}        
        Question: {state["query"]}
        <assistant>:
        """
        ans = (self.raw_pipeline(prompt)[0]['generated_text']).split("<assistant>:", 1)[1].strip()
        return {**state, "final_answer": ans}
    


def load_docs():
    #loading
    urls = [
        "https://en.wikipedia.org/wiki/Inherently_funny_word",
        "https://en.wikipedia.org/wiki/Longest_word_in_English",
        # "https://en.wikipedia.org/wiki/Fan_death",
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    ]
    documents = (WebBaseLoader(urls)).load()
    
    
    # splitting
    text_split = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_split.split_documents(documents)   
    return chunks

def create_vectordb(chunks):
    # i am running on cpu because i have a small gpu only
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",model_kwargs={'device': 'cpu'})
    return FAISS.from_documents(documents=chunks, embedding=embeddings)

def load_llm():
    # local llm 
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.8,
        repetition_penalty=1.1,
        do_sample=True
    )

    # LangChain wrapper around the local pipeline
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm, pipe


def build_graph(agent_functions):
    # workflow with langgraph based on this: https://medium.com/ai-agents/langgraph-for-beginners-part-4-stategraph-794004555369
    # graph with dict state
    state_keys = {
        "query": str,
        "search_results": str,
        "final_answer": str,
        "approach": str,
    }
    workflow = StateGraph(state_keys)
    
    # nodes
    workflow.add_node("retrieve_info", agent_functions.retrieve_info)
    workflow.add_node("generate", agent_functions.generate)
    
    # edges
    # workflow.add_edge(START, "retrieve_info") --> this didn't work for me
    workflow.add_edge("retrieve_info", "generate")
    workflow.add_edge("generate", END)
    
    # entry point
    workflow.set_entry_point("retrieve_info") 
    return workflow.compile()

def test_agent(graph, query):
    # test w query
    print(f"\n query: {query}")
    
    # state with query
    state = {
        "query": query,
        "search_results": "",
        "final_answer": "",
        "approach": "RAG" 
    }

    # run the agent
    final_state = graph.invoke(state)
    
    print(f"\n Answer: {final_state['final_answer']}")
    
    return final_state

def main():
    print("Loading docs")
    chunks = load_docs()
    vector_db = create_vectordb(chunks)
    
    print("Loading llm")
    llm, pipe = load_llm()
    
    print("Building agent ")
    agent_graph = build_graph(Agent(vector_db, llm, pipe))
    
    # Test the agent
    test_queries = [
        "What is the longest English word?", # shouldn't work because of tokeization
        "What is an inherently funny word?", # should work
        "Where does the fan death theory originate from?", # shouldn't work as I commented out, should hallucinate stg
        "What is the capital of France?",  # should work frombase llm knowledge
        "Explain what Retrieval-Augmented Generation is"
    ]
    
    for query in test_queries:
        test_agent(agent_graph, query)
    

if __name__ == "__main__":
    main()
