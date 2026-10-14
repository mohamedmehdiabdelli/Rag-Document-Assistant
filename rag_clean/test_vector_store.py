from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# Load document
text = DocumentLoader.load_pdf(
    "data/sample.pdf"
)


# Split
pages = DocumentLoader.load_pdf("data/sample.pdf")

chunks = TextSplitter.split_pages(
    pages,
    "data/sample.pdf"
)

# Create embeddings
embedder = EmbeddingModel()

embeddings = embedder.encode_chunks(chunks)

# Create FAISS store
store = VectorStore()

store.create(
    embeddings,
    chunks
)


# Test question
question = "what is the CONTROL NUMBER of Policy and Procedures?"


question_vector = embedder.encode_query(question)


results = store.search(
    question_vector,
    k=3
)


print("\nRelevant chunks:\n")

for r in results:
    print("\n----------------")
    print("Document:", r["document"])
    print("Page:", r["page"])
    print("Score:", r["distance"])
    print("Text:", r["text"])