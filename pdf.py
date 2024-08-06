import os
import json
import getpass
import argparse
import pickle
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests
import tempfile

CONFIG_FILE = 'api_keys.json'
HISTORY_DIR = 'chat_histories'

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
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            os.unlink(temp_file_path)
            return docs
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
            return None
    else:
        loader = PyPDFLoader(path_or_url)
        return loader.load()

def load_chat_history(session_id: str) -> ChatMessageHistory:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.pkl")
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            return pickle.load(f)
    return ChatMessageHistory()

def save_chat_history(session_id: str, history: ChatMessageHistory):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.pkl")
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)

def display_chat_history(history: ChatMessageHistory):
    messages = history.messages
    if messages:
        print("Previous conversation:")
        displayed_messages = []
        for msg in messages:
            role = "Human" if msg.type == "human" else "AI"
            message = f"{role}: {msg.content}"
            if message not in displayed_messages:
                print(message)
                displayed_messages.append(message)
        print("\n")
    else:
        print("No previous conversation found.\n")

def main(args):
    llm = select_llm(args.llm)

    docs = load_pdf(args.pdf_path)

    if not docs:
        print("Failed to load the PDF. Please check the file or URL.")
        exit()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    if not splits:
        print("No text was extracted from the PDF. Please check the file content.")
        exit()

    openai_api_key = get_api_key("openai")
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If asked about previous questions or conversation history, refer to the chat history provided. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    pdf_name = os.path.basename(args.pdf_path)
    session_id = f"session_{pdf_name}"

    chat_history = load_chat_history(session_id)

    print("PDF loaded successfully. You can now ask questions.")
    display_chat_history(chat_history)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        nonlocal chat_history
        return chat_history

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    if args.question:
        response = conversational_rag_chain.invoke(
            {"input": args.question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
        print(f"Answer: {answer}")
        chat_history.add_user_message(args.question)
        chat_history.add_ai_message(answer)
        save_chat_history(session_id, chat_history)
    else:
        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            response = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            print(f"Answer: {answer}\n")
            chat_history.add_user_message(question)
            chat_history.add_ai_message(answer)
            save_chat_history(session_id, chat_history)

    print(f"Chat history saved to {os.path.join(HISTORY_DIR, f'{session_id}.pkl')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Question Answering CLI")
    parser.add_argument("pdf_path", help="Path or URL to the PDF file")
    parser.add_argument("-l", "--llm", choices=["openai", "anthropic", "google"], default="openai", help="Choose the LLM to use")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    args = parser.parse_args()
    main(args)