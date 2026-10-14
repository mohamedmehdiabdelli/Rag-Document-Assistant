from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


class TextSplitter:

    @staticmethod
    def split_pages(pages, file_path):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )

        chunks = []

        document_name = os.path.basename(file_path)

        for page in pages:

            page_chunks = splitter.split_text(page["text"])

            for chunk in page_chunks:

                chunks.append({
                    "text": chunk,
                    "page": page["page"],
                    "document": document_name
                })

        return chunks