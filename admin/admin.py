import streamlit as st
import uuid
import boto3
import os

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.vectorstores import FAISS

s3_client = boto3.client('s3')
BUKCET_NAME = "awscloudwatch-test-bucket"

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def split_text(pages, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = text_splitter.split_documents(pages)
    return docs
   
def create_vector_store(text_chunks, req_id):
    vectorstore_faiss = FAISS.from_documents(
        text_chunks,
        bedrock_embeddings
    )

    file_name = f"{req_id}.bin"

    folder_path = "/tmp/"

    vectorstore_faiss.save_local(folder_path=folder_path, index_name=file_name)

    s3_client.upload_file(
        Filename=folder_path+"/"+file_name+".faiss",
        Bucket=BUKCET_NAME,
        Key="my_faiss.faiss"
    )

    s3_client.upload_file(
        Filename=folder_path+"/"+file_name+".pkl",
        Bucket=BUKCET_NAME,
        Key="my_faiss.pkl"
    )

    return True



def get_uuid():
    return str(uuid.uuid4())

def main():
    st.write("This is Admin Site for PDF chat bot application")
    upload_file = st.file_uploader("Upload a PDF file", type="pdf")

    if upload_file is not None:
        req_id = get_uuid()
        st.write(f"File Request ID: {req_id}")
        save_file = f"{req_id}.pdf"
        with open(save_file, "wb") as f:
            f.write(upload_file.getvalue())

        loader = PyPDFLoader(save_file)
        pages = loader.load_and_split()

        st.write(f"Total no. of pages : {len(pages)}")

        split_document = split_text(pages, 1000, 200)
        st.write(f"Total no. of chunks : {len(split_document)}")

        st.write("Embedding the document chunks")
        response = create_vector_store(split_document, req_id)

        if response:
            st.write("Vector store created successfully")
        else:
            st.write("Vector store creation failed")

if __name__ == "__main__":
    main()
