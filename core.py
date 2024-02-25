from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

import os
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# INDEX_NAME = "langchain-dox-index"


def run_llm(prompt: str, VectorStore: FAISS,chat_history: List[Dict[str, Any]] = []):
    docs = VectorStore.similarity_search(query=prompt, k=3)
    llm=ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=prompt)
        print(response)
        print(cb)
    
  
   
    
    return response