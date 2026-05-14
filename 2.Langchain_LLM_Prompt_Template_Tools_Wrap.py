import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

prompt = PromptTemplate.from_template(
    "Answer clearly: {question}"
)

chain = prompt | llm

qa_tool = Tool(
    name="Simple QA",
    func=lambda q: chain.invoke({"question": q}).content,
    description="A basic LLM chain that answers clear questions"
)

query = "What is LangGraph in LangChain?"
answer = qa_tool.run(query)

output_text = f"User Question: {query}\n\nTool Answer:\n{answer}"

print(output_text)

with open("output2.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\nSaved to output2.txt")
