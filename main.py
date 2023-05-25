from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def run():
    load_dotenv()
    st.set_page_config(page_title="与PDF对话")
    st.header("快来问我！")

    # 上传文件
    pdf = st.file_uploader("上传你的PDF", type="pdf")

    # 提取文本
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # 分块
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge = FAISS.from_texts(chunks, embeddings)

        # 提问
        question = st.text_input("向你的PDF提问吧：")
        if question:
            docs = knowledge.similarity_search(question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question)
            st.write(response)


if __name__ == '__main__':
    run()
