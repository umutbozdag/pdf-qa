# Gerekli modülleri içe aktarın
import os
import getpass
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# OpenAI API anahtarını ayarlayın
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def load_file(file_path):
    """Load an Excel or CSV file."""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv file.")

# Dosyayı yükleyin
file_path = "./industry.csv"  # veya "path/to/your/file.csv"
df = load_file(file_path)

print("Available columns:", df.columns.tolist())

# Tüm sütunları birleştirerek yeni bir içerik sütunu oluşturun
df['combined_content'] = df.apply(lambda row: ' | '.join(row.astype(str)), axis=1)

# DataFrame'i belgelere dönüştürün
loader = DataFrameLoader(df, page_content_column='combined_content')
docs = loader.load()

# Belgeleri parçalara ayırın
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Number of documents:", splits)

# Vektör deposu oluşturun
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Dil modelini ayarlayın
llm = ChatOpenAI(model="gpt-4o-mini")

# Prompt şablonunu oluşturun
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

# Soru-cevap zinciri oluşturun
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Soru sormak için fonksiyon
def ask_question(question):
    results = rag_chain.invoke({"input": question})
    return results["answer"]

# Örnek kullanım
question = "What is this CSV file about?"
answer = ask_question(question)
print(f"Soru: {question}")
print(f"Cevap: {answer}")