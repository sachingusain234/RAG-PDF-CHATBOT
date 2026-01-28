from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
# -------------------------------
# 1. Load PDF
# -------------------------------
pdf_path = "C:/Users/sachi/Downloads/Sachin Gusain Resume January.pdf"   # <-- your PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# -------------------------------
# 2. Split text into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# -------------------------------
# 3. Create embeddings + vector store
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# 4. Groq LLM (Inference)
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    groq_api_key=groq_api_key
)

# -------------------------------
# 5. Prompt
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)

# -------------------------------
# 6. RAG Chain
# -------------------------------
def rag_qa(question: str):
    docs = retriever.invoke(question)   # ✅ FIX
    context = "\n\n".join(d.page_content for d in docs)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question
    })



# -------------------------------
# 7. Ask questions
# -------------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag_qa(query)
        print("\nAnswer:\n", answer)
