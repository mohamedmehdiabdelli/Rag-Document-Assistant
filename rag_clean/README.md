# Document Question Answering with Retrieval-Augmented Generation

A locally hosted question-answering application for PDF documents. Users upload a PDF and ask questions in natural language; answers are generated from retrieved passages of the document, with page-level source attribution. All processing runs on the local machine, so no document content is sent to external services and no API keys are required.

## Features

* PDF text extraction with page tracking
* Overlapping text chunking that preserves source page and document name
* Dense semantic retrieval using sentence embeddings and a FAISS index
* Context expansion that includes the chunks adjacent to each retrieved match
* Answer generation by a local LLM, constrained to the retrieved context
* Streamlit interface with an optional view of retrieved chunks, pages and similarity scores

## Architecture

```
PDF --> Loader --> Text Splitter --> Embedding Model --> FAISS Index
                                                              |
Question --> Embedding Model --> Top-k Retrieval + Neighbours
                                                              |
                                          Prompt + Context --> Local LLM --> Answer
```

|Stage|Implementation|
|-|-|
|PDF parsing|`pypdf`|
|Chunking|`RecursiveCharacterTextSplitter` (500 characters, 150 overlap)|
|Embeddings|`sentence-transformers/all-MiniLM-L6-v2` (normalized vectors)|
|Vector search|FAISS `IndexFlatIP` (inner product, equivalent to cosine similarity on normalized vectors)|
|Generation|`qwen2.5:3b` served by Ollama (temperature 0.1, fixed seed)|
|Interface|Streamlit|

The system prompt instructs the model to answer only from the supplied context and to state explicitly when the answer is not present in the document.

## Project Structure

```
.
├── app.py                  Streamlit application
├── requirements.txt        Python dependencies
├── data/
│   └── sample.pdf          Sample document for testing
├── src/
│   ├── document\\\_loader.py  PDF text extraction
│   ├── text\\\_splitter.py    Chunking with page metadata
│   ├── embeddings.py       Embedding model wrapper
│   ├── vector\\\_store.py     FAISS index and retrieval
│   ├── llm.py              Ollama client and prompt template
│   └── rag\\\_pipeline.py     End-to-end pipeline
└── test\\\_\\\*.py               Manual test scripts for individual components
```

## Requirements

* Python 3.10 or later
* [Ollama](https://ollama.com) installed and running locally

## Installation

```bash
git clone https://github.com/mohamedmehdiabdelli/Rag-Document-Assistant.git
cd <repository>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\\\Scripts\\\\activate
pip install -r requirements.txt

ollama pull qwen2.5:3b
```

The embedding model is downloaded automatically from Hugging Face on first run.

## Usage

Start the web application:

```bash
streamlit run app.py
```

Then upload a PDF, wait for the confirmation message, and enter a question. Enable "Show retrieval details" to inspect the chunks used to produce the answer.

To use the command-line interface with the sample document:

```bash
python test\\\_pipeline.py
```

## Configuration

Settings are defined directly in the source and can be adjusted as needed.

|Setting|Location|Default|
|-|-|-|
|LLM model, temperature, seed|`src/llm.py`|`qwen2.5:3b`, 0.1, 42|
|Ollama endpoint|`src/llm.py`|`http://localhost:11434`|
|Chunk size and overlap|`src/text\\\_splitter.py`|500, 150|
|Embedding model|`src/embeddings.py`|`all-MiniLM-L6-v2`|
|Number of retrieved chunks (k)|`src/rag\\\_pipeline.py`|3|

## Limitations

* One document is indexed at a time; uploading a new file replaces the previous index.
* The index is held in memory and is not persisted between sessions.
* Scanned or image-only PDFs are not supported, as no OCR step is included.
* Answer quality depends on the local model; larger models generally produce better results.

## Roadmap

* Persist the vector index to disk
* Support multiple documents and cross-document queries
* Add automated tests and a retrieval evaluation set
* Add optional OCR for scanned documents
* Externalize configuration to a settings file or environment variables

## Sample Data

`data/sample.pdf` is Ford Motor Company's publicly available 2008 business plan submission to the U.S. Senate Banking Committee, included for demonstration purposes only.

