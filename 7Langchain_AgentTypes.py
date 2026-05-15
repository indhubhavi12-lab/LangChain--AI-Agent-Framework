# ================================
# Gemini + LangChain Agent Example
# ================================

from dotenv import load_dotenv
import os

# LangChain Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.agents import Tool, initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

# Tavily Search
from langchain_community.tools.tavily_search import TavilySearchResults

# ==================================
# 🔐 Load API Keys
# ==================================
env_path = os.path.join(os.path.dirname(__file__), ".env1")
load_dotenv(env_path)

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY is missing. Add it to .env1.")

if not tavily_api_key:
    raise RuntimeError("TAVILY_API_KEY is missing. Add it to .env1.")

# ==================================
# 🤖 Gemini LLM
# ==================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ==================================
# 🧠 Simple QA Tool
# ==================================
qa_prompt = PromptTemplate.from_template(
    "Answer clearly: {question}"
)

qa_chain = LLMChain(
    llm=llm,
    prompt=qa_prompt
)

qa_tool = Tool(
    name="Simple_QA",
    func=qa_chain.run,
    description="Answer questions clearly without hallucination"
)

# ==================================
# 🌐 Web Search Tool
# ==================================
search_tool = Tool(
    name="Web_Search",
    func=TavilySearchResults(max_results=3).run,
    description="Search the internet for current information"
)

tools = [qa_tool, search_tool]

# ==================================
# 💾 Memory
# ==================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ==================================
# 1️⃣ Zero Shot ReAct Agent
# ==================================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Summarize LangChain in 2 lines, then tell me who created it."
)

print("\n✅ Zero Shot Result:\n", result)

# ==================================
# 2️⃣ Conversational ReAct Agent
# ==================================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Summarize LangChain in 2 lines, then tell me who created it."
)

print("\n✅ Conversational Result:\n", result)

# ==================================
# 3️⃣ Chat Conversational ReAct
# ==================================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Plan a 3-step study path for LangChain."
)

print("\n✅ Study Plan:\n", result)

# ==================================
# 4️⃣ Structured Tool Example
# ==================================

# Input Schema
class TitleInput(BaseModel):
    topic: str
    tone: str = "concise"

# Tool Function
def title_tool_fn(topic: str, tone: str = "concise") -> str:
    return f"{tone.title()} Title: {topic} in Practice"

# Structured Tool
title_tool = StructuredTool.from_function(
    name="TitleMaker",
    func=title_tool_fn,
    description="Generate a title using topic and tone",
    args_schema=TitleInput
)

# Structured Chat Agent
agent = initialize_agent(
    tools=[title_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Make a friendly title about LangGraph tutorials."
)

print("\n✅ Structured Tool Result:\n", result)

# ==================================
# 5️⃣ OpenAI Functions Equivalent
# ==================================
# Gemini works best with STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Search the web for LangGraph docs and give me 3 bullets."
)

print("\n✅ Web Search Result:\n", result)

# ==================================
# 6️⃣ Self Ask With Search
# ==================================

search_tool_selfask = Tool(
    name="Intermediate Answer",
    func=TavilySearchResults(max_results=1).run,
    description="Search for missing facts"
)

agent = initialize_agent(
    tools=[search_tool_selfask],
    llm=llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
    handle_parsing_errors=True
)

result = agent.run(
    "When was the first LangChain release and who founded it?"
)

print("\n✅ Self Ask Result:\n", result)

# Save output to output7.txt

result = agent.run(
    "Plan a 3-step study path for LangChain."
)

# Print output
print("\n✅ Agent Output:\n")
print(result)

# Save output into text file
with open("output7.txt", "w", encoding="utf-8") as file:
    file.write("===== LangChain Agent Output =====\n\n")
    file.write(result)

print("\n✅ Output successfully saved to output7.txt")
