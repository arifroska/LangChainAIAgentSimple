import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ragdb")

DB_CONN = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # kalau masih mau pakai OpenAI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")