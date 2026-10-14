from src.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

pipeline.load_document("data/sample.pdf")

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    result = pipeline.ask(question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\n--- SOURCES ---")

    for r in result["retrieved_chunks"]:
        print(f"\nDoc: {r['document']}")
        print(f"Page: {r['page']}")
        print(f"Score: {r['distance']}")