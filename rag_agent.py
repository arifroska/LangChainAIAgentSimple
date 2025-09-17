from config import DB_CONN, GEMINI_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

#pemilihan model embedding
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#pengubahan data menjadi vector database
vectorstore = PGVector(
    connection_string=DB_CONN,
    embedding_function=embeddings,
    collection_name="documents"
)

#membuat retriever untuk melakukan embedding, mencari dokumen vector yang mirip dan mengembalikan untuk diproses LLM
retriever = vectorstore.as_retriever()

#siapkan llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

#menggabungkan retriever dengan LLM 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=(
                "Kamu adalah asisten AI yang hanya menjawab berdasarkan konteks berikut.\n\n"
                "Konteks:\n{context}\n\n"
                "Instruksi:\n"
                "Pertanyaan: {question}"
                "1. Jika konteks berisi informasi relevan, jawab hanya berdasarkan konteks.\n"
                "2. Jika konteks kosong atau tidak relevan, awali jawaban dengan kalimat:\n"
                "\"Tidak ada dokumen ditemukan. Dijawab dengan pengetahuan umum.\"\n"
                "dan kemudian jawab berdasarkan pengetahuan umummu.\n\n"
            ),
            input_variables=["context", "question"]
        )
    }
)

#input pertanyaan user
while True:
    query = input("\n🧠 Pertanyaan: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Keluar dari agent...")
        break

    #cek dokumen apakah ada atau tidak
    # docs = retriever.get_relevant_documents(query)
    # if not docs:
    #     print("⚠️ Tidak ada dokumen relevan. Menjawab dengan knowledge umum...")
    #     answer = ChatGoogleGenerativeAI(...).invoke(query)
    # else:
    #     answer = qa.run(query)
    answer = qa.run(query)
    print(f"🤖 Jawaban: {answer}")
