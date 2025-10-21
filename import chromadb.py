import chromadb
from chromadb.config import Settings

db = chromadb.Client(Settings(persist_directory="/Users/sanduandrei/Desktop/Betting_RAG/Index/chroma"))
col = db.get_collection("football_top5")

print("Collection name:", col.name)
print("Total items:", col.count())