from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
# data loading
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))

# text splitting
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
text_chunks = split_text(documents=documents)
#print("Length of text chunks: ", len(text_chunks)) 

# embedding
def embed_text():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = embed_text()

# vector store
vec_Path = "vector_store/faiss_database"
db = FAISS.from_documents(text_chunks, embeddings)
# Save the vector store
db.save_local(vec_Path)

