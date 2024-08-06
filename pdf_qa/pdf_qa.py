import os
import json
import getpass
import argparse
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import requests
import tempfile

CONFIG_FILE = 'api_keys.json'

def load_api_keys():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_api_keys(api_keys):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(api_keys, f)

def get_api_key(llm_choice):
    api_keys = load_api_keys()
    key_name = f"{llm_choice.upper()}_API_KEY"
    api_key = api_keys.get(key_name)
    if not api_key:
        api_key = getpass.getpass(f"Enter your {llm_choice} API key: ")
        api_keys[key_name] = api_key
        save_api_keys(api_keys)
    return api_key

def select_llm(llm_choice):
    if llm_choice == "openai":
        api_key = get_api_key("openai")
        return ChatOpenAI(api_key=api_key)
    elif llm_choice == "anthropic":
        api_key = get_api_key("anthropic")
        return ChatAnthropic(api_key=api_key)
    elif llm_choice == "google":
        api_key = get_api_key("google")
        return ChatGoogleGenerativeAI(api_key=api_key)
    else:
        print("Invalid LLM choice. Exiting.")
        exit()

def load_pdf(path_or_url):
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            os.unlink(temp_file_path)
            return docs
        else:
            loader = PyPDFLoader(path_or_url)
            return loader.load()
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return None

def pdf_qa(pdf_path, llm_choice="openai", question=None):
    llm = select_llm(llm_choice)
    docs = load_pdf(pdf_path)
    
    if not docs:
        print("Failed to load the PDF. Please check the file or URL.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    if not splits:
        print("No text was extracted from the PDF. Please check the file content.")
        return

    openai_api_key = get_api_key("openai")
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

    retriever = vectorstore.as_retriever()

    system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, say that you 
    don't know. Use three sentences maximum and keep the 
    answer concise.

    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("PDF loaded successfully. You can now ask questions.")
    if question:
        response = rag_chain.invoke({"input": question})
        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
        print(f"Answer: {answer}")
    else:
        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = rag_chain.invoke({"input": question})
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            print(f"Answer: {answer}\n")

def main():
    parser = argparse.ArgumentParser(description="PDF Question Answering CLI")
    parser.add_argument("pdf_path", help="Path or URL to the PDF file")
    parser.add_argument("-l", "--llm", choices=["openai", "anthropic", "google"], default="openai", help="Choose the LLM to use")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    args = parser.parse_args()

    llm = select_llm(args.llm)
    docs = load_pdf(args.pdf_path)
    
    if not docs:
        print("Failed to load the PDF. Please check the file or URL.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    if not splits:
        print("No text was extracted from the PDF. Please check the file content.")
        return

    openai_api_key = get_api_key("openai")
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

    retriever = vectorstore.as_retriever()

    system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, say that you 
    don't know. Use three sentences maximum and keep the 
    answer concise.

    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("PDF loaded successfully. You can now ask questions.")
    
    if args.question:
        response = rag_chain.invoke({"input": args.question})
        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
        print(f"Answer: {answer}")
    else:
        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = rag_chain.invoke({"input": question})
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()