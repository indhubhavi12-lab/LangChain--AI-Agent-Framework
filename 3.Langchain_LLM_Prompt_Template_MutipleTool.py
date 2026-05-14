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

# Tool 1: Simple QA Tool
qa_prompt = PromptTemplate.from_template("Answer clearly: {question}")
qa_chain = qa_prompt | llm
qa_tool = Tool(
    name="Simple QA",
    func=lambda q: qa_chain.invoke({"question": q}).content,
    description="Answer factual questions clearly"
)

# Tool 2: Summarizer Tool
summary_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{text}")
summary_chain = summary_prompt | llm
summary_tool = Tool(
    name="Summarizer",
    func=lambda text: summary_chain.invoke({"text": text}).content,
    description="Summarizes input text"
)

qa_query = "What is LangGraph in LangChain?"
summary_text = """
LangGraph is a library built on top of LangChain that helps developers create stateful, multi-step agents
as graphs. Each node represents a step like calling an LLM or a tool. It's ideal for advanced AI workflows.
"""

qa_answer = qa_tool.run(qa_query)
summary_answer = summary_tool.run(summary_text)

output_text = (
    f"Simple QA Tool\n{'='*40}\n"
    f"Question: {qa_query}\n\n"
    f"Answer:\n{qa_answer}\n\n"
    f"Summarizer Tool\n{'='*40}\n"
    f"Input Text:{summary_text}\n"
    f"Summary:\n{summary_answer}\n"
)

print(output_text)

with open("output3.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Saved to output3.txt")
