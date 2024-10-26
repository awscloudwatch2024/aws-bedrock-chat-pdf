import streamlit as st
import boto3
import os
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

s3_client = boto3.client('s3')
BUCKET_NAME = "awscloudwatch-test-bucket"

folder_path = "/tmp/"

def get_response(llm, faiss_index, question):
    prompt_tempalte = """

    Human: Use the following pieces of context to provide a concise answer to the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Assistant:
    """

    PROMPT = PromptTemplate(template=prompt_tempalte, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer =  qa({"query":question})
    return answer['result']

def get_llm():
    llm = Bedrock(
        model_id="anthropic.claude-v2:1", 
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 512}
    )
    return llm

def load_index():
    
    s3_client.download_file(
        Filename=f"{folder_path}my_faiss.faiss",
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss"
    )

    s3_client.download_file(
        Filename=f"{folder_path}my_faiss.pkl",
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl"
    )

def main():
    st.header("This is client side PDF Chat bot")

    load_index()

    dir_list = os.listdir("/tmp/")
    st.write("Files in /tmp/ directory")
    st.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("Vector store loaded successfully")
    question = st.text_input("Enter your question")
    if st.button("Get Response"):
        with st.spinner("Getting response..."):

            llm = get_llm()

            st.write(get_response(llm, faiss_index, question))
            st.success("Done!")

if __name__ == "__main__":
    main()