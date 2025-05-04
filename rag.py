from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


#based on this https://www.geeksforgeeks.org/build-rag-pipeline-using-open-source-large-language-models/

# load and process whatever web content, can be substitued with wikipedia
urls = [
    "https://en.wikipedia.org/wiki/Inherently_funny_word",
    "https://en.wikipedia.org/wiki/Longest_word_in_English", # --> this does not work because of tokenization i guess
    "https://en.wikipedia.org/wiki/Fan_death"
]

documents = WebBaseLoader(urls).load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# create embeddings using a local model 
# smaller model that can run on a CPU (my machine only has MX250 GpU)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",model_kwargs={'device': 'cpu'})# all-MiniLM-L6-v2",

# vector store
# i only ever used faiss for vector database and then clustering
# i read that chroma is also good, but never worked with it
faiss_vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = faiss_vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# small local open-source LLM
# my machine's GPU cannot host much bigger models, but can be switched
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# text gen pipline (https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# I didn't play around a lot with the parameters
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

# create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# # run a query
# query = "What is the best anime of 2024?"
# print(f"the query was: {query}")
# result = qa.invoke(query)
# print(f"the answer: {result['result']}")

def test_rag(query):
    print(f"\nQuery: {query}")
    
    # 1. with rag
    print("With RAG:")
    rag_res = qa.invoke(query)
    print(f"Answer: {rag_res['result']}")
    print()
    print("Sources:")
    for i, doc in enumerate(rag_res['source_documents']):
        print(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content prev: {doc.page_content[:150]}")
    
    # 2. just the llm 
    print("\nWithout RAG:")
    # query formatting for langchain
    prompt = f"<human>: {query}\n<assistant>:"
    # just the response, everything after prompt
    res = (pipe(prompt)[0]['generated_text']).split("<assistant>:", 1)[1].strip()
    print(f"Answer: {res}")

# questions that should be in the documents
test_rag("What is a the longest english word?")
test_rag("What is an inheretly funny word?")
test_rag("Where does the fan death theory originate from?")

# question unlikely to be in the documents
test_rag("What is the capital of France?")
test_rag("What is the best anime of 2024?")
# couldn't load anime pipeline