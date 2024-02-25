from typing import Set

from core import run_llm
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# Sidebar contents
from langchain_community.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

with st.sidebar:
    st.title("LangChain🦜🔗 Udemy Course- Helper Bot")
    st.markdown("### Enter your message here...")
    add_vertical_space(5)
    st.write("Made with 🦜 by Tewo")


if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    


def main():
    st.write("Chat with PDF ")
    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write(chunks)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}/index.faiss"):
            with open(f"{store_name}/index.faiss", "rb") as f:
               
                VectorStore=FAISS.load_local(store_name, embeddings=OpenAIEmbeddings())            
            # st.write(VectorStore)
            st.write("Store loaded from Disk")
        else:
            st.write("Store created")
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(store_name)
        
        # query = st.text_input("Ask your question here please...")
        prompt = st.text_input("Prompt", placeholder="Enter your message here...") or st.button("Submit")
        if prompt:
            with st.spinner("Generating response..."):
             generated_response = run_llm(prompt=prompt, chat_history=st.session_state["chat_history"],VectorStore=VectorStore)
             st.write(generated_response)

            # sources = set(
            #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
            # )
            # formatted_response = (
            #     f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
            # )

            # st.session_state.chat_history.append((prompt, generated_response["answer"]))
            # st.session_state.user_prompt_history.append(prompt)
            # st.session_state.chat_answers_history.append(formatted_response)
            
           

if __name__ == "__main__":
    main()
