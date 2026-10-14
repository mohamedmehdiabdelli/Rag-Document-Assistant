import faiss
import numpy as np


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []

    def create(self, embeddings, chunks):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(np.array(embeddings))

        self.chunks = chunks

    def search(self, query_embedding, k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]),
            k
        )

        expanded = {}

        # Store original retrieved chunks with their scores
        for rank, idx in enumerate(indices[0]):

            if idx == -1:
                continue

            expanded[idx] = float(distances[0][rank])

            # Add neighbors (no score available)
            for neighbor in [idx - 1, idx + 1]:

                if 0 <= neighbor < len(self.chunks):
                    if neighbor not in expanded:
                        expanded[neighbor] = None

        results = []

        for idx in sorted(expanded.keys()):

            chunk = self.chunks[idx]

            results.append({
                "chunk_id": idx,
                "text": chunk["text"],
                "page": chunk["page"],
                "document": chunk["document"],
                "distance": expanded[idx]
            })

        return results