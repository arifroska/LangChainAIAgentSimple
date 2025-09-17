from config import DB_CONN, OPENAI_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

#data yang akan dijadikan vector sebagai referensi untuk RAG model
texts = [
    "LangChain adalah framework untuk membangun aplikasi berbasis LLM.",
    "PostgreSQL mendukung penyimpanan vektor melalui ekstensi pgvector."
]

#proses pemilihan model embedding
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#proses mengubah data menjadi vector
vectorstore = PGVector.from_texts(
    texts,
    embedding=embeddings,
    collection_name="documents",
    connection_string=DB_CONN
)

print("✅ Data berhasil dimasukkan ke database (Gemini Embeddings)!")
