import requests


class LocalLLM:

    def __init__(self, model="qwen2.5:3b", temperature=0.1, seed=42):
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def generate(self, context, question):

        prompt = f"""
You are an AI assistant answering questions about a document.

Rules:
- Use ONLY the information found in the provided context.
- Do not use outside knowledge.
- If the answer cannot be found, respond with:
  "The answer is not found in the uploaded document."

When the answer exists:
- Write a complete and well-structured response.
- Combine information from multiple context chunks if necessary.
- Preserve all important numbers, dates, and names exactly.
- Do not invent information.
- Use 1–3 paragraphs depending on the amount of information available.

Context:
{context}

Question:
{question}

Answer:
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed
                }
            }
        )

        return response.json()["response"]