from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm import LocalLLM


# Load PDF
pages = DocumentLoader.load_pdf(
    "data/sample.pdf"
)

# Split
chunks = TextSplitter.split_pages(
    pages,
    "data/sample.pdf"
)

# Embeddings
embedder = EmbeddingModel()

embeddings = embedder.encode_chunks(chunks)

# FAISS
store = VectorStore()

store.create(
    embeddings,
    chunks
)

# LLM
llm = LocalLLM()

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    question_embedding = embedder.encode_query(question)

    retrieved_chunks = store.search(
        question_embedding,
        k=3
    )

    context = "\n\n".join(
        r["text"] for r in retrieved_chunks
    )

    answer = llm.generate(
        context,
        question
    )

    print("\nAnswer:")
    print(answer)