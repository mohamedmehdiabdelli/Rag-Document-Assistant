from src.llm import LocalLLM

llm = LocalLLM()

context = """
Employees are entitled to 25 days of annual leave.

Leave requests must be submitted through the HR portal.
"""

question = "How do I request annual leave?"

answer = llm.generate(context, question)

print(answer)