from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm import LocalLLM


class RAGPipeline:

    def __init__(self):

        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.llm = LocalLLM()

        self.chunks = []
        self.embeddings = None

    # STEP 1: Load + build index
    def load_document(self, file_path):

        pages = DocumentLoader.load_pdf(file_path)

        self.chunks = TextSplitter.split_pages(
            pages,
            file_path
        )

        self.embeddings = self.embedder.encode_chunks(self.chunks)

        self.vector_store.create(
            self.embeddings,
            self.chunks
        )

    # STEP 2: Ask question
    def ask(self, question, k=3):

        # embed query
        query_vector = self.embedder.encode_query(question)

        # retrieve
        retrieved = self.vector_store.search(query_vector, k=k)

        # build context
        context = "\n\n".join([r["text"] for r in retrieved])

        # generate answer
        answer = self.llm.generate(context, question)

        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "context": context
        }