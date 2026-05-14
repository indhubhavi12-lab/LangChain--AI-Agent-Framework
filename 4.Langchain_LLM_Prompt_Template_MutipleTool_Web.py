import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ----------------------------------------
# Load API Keys
# ----------------------------------------

load_dotenv()

google_api_key = os.getenv("AIzaSyBxTLxI2HC27PDbKAe4Kbfk-CJD8Fn_WQI")
tavily_api_key = os.getenv("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")

# ----------------------------------------
# Initialize Gemini LLM
# ----------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0
)

# ----------------------------------------
# Tool 1: Simple QA
# ----------------------------------------

qa_prompt = PromptTemplate.from_template(
    "Answer clearly: {question}"
)

qa_chain = qa_prompt | llm

qa_tool = Tool(
    name="Simple QA",
    func=lambda q: qa_chain.invoke({"question": q}).content,
    description="Answer factual questions clearly"
)

# ----------------------------------------
# Tool 2: Summarizer
# ----------------------------------------

summary_prompt = PromptTemplate.from_template(
    "Summarize the following text:\n\n{text}"
)

summary_chain = summary_prompt | llm

summary_tool = Tool(
    name="Summarizer",
    func=lambda text: summary_chain.invoke({"text": text}).content,
    description="Summarizes input text"
)

# ----------------------------------------
# Tool 3: Tavily Web Search
# ----------------------------------------

search = TavilySearchResults(max_results=3)

search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Search the internet for current information"
)

# ----------------------------------------
# Example Inputs
# ----------------------------------------

qa_query = "What is LangGraph in LangChain?"

summary_text = """
LangGraph is a framework for building stateful multi-step agents using LangChain.
It uses graph-based design to model agent workflows and memory.
"""

search_query = "Latest updates on GPT-4o by OpenAI"

# ----------------------------------------
# Run Tools
# ----------------------------------------

qa_result = qa_tool.run(qa_query)

summary_result = summary_tool.run(summary_text)

search_result = search_tool.run(search_query)

# ----------------------------------------
# Print Outputs
# ----------------------------------------

print("\n🧠 Simple QA Tool Output:\n")
print(qa_result)

print("\n📝 Summarizer Tool Output:\n")
print(summary_result)

print("\n🌐 Web Search Tool Output:\n")
print(search_result)

# ----------------------------------------
# Save Outputs to output4.txt
# ----------------------------------------

with open("output4.txt", "w", encoding="utf-8") as file:

    file.write("🧠 Simple QA Tool Output:\n")
    file.write(str(qa_result))
    file.write("\n\n")

    file.write("📝 Summarizer Tool Output:\n")
    file.write(str(summary_result))
    file.write("\n\n")

    file.write("🌐 Web Search Tool Output:\n")
    file.write(str(search_result))

print("\n✅ Results saved to output4.txt")
