import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import initialize_agent, AgentType

# ----------------------------------------
# Load API Keys
# ----------------------------------------

load_dotenv()

google_api_key = os.getenv("Enter your API Key")
tavily_api_key = os.getenv("Enter your API Key")

# ----------------------------------------
# Initialize Gemini LLM
# ----------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0
)

# ----------------------------------------
# Tool 1: Simple QA Tool
# ----------------------------------------

qa_prompt = PromptTemplate.from_template(
    "Answer clearly: {question}"
)

qa_chain = qa_prompt | llm

qa_tool = Tool(
    name="Simple QA",
    func=lambda q: qa_chain.invoke({"question": q}).content,
    description="Answers factual questions clearly"
)

# ----------------------------------------
# Tool 2: Summarizer Tool
# ----------------------------------------

summary_prompt = PromptTemplate.from_template(
    "Summarize this text:\n\n{text}"
)

summary_chain = summary_prompt | llm

summary_tool = Tool(
    name="Summarizer",
    func=lambda text: summary_chain.invoke({"text": text}).content,
    description="Summarizes long paragraphs or text content"
)

# ----------------------------------------
# Tool 3: Web Search Tool
# ----------------------------------------

search = TavilySearchResults(max_results=3)

search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Search the internet for current and live information"
)

# ----------------------------------------
# Combine Tools
# ----------------------------------------

tools = [
    qa_tool,
    summary_tool,
    search_tool
]

# ----------------------------------------
# Initialize Agent
# ----------------------------------------

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ----------------------------------------
# User Queries
# ----------------------------------------

queries = [
    "What is LangGraph in LangChain?",
    "Summarize this: LangChain is a framework to build LLM apps using prompts, memory, tools, and agents.",
    "Latest news about OpenAI GPT-4o"
]

# ----------------------------------------
# Run Queries + Save Output
# ----------------------------------------

with open("output5.txt", "w", encoding="utf-8") as file:

    for query in queries:

        print("\n🧑‍💻 User Query:", query)

        response = agent_executor.run(query)

        print("\n🤖 Agent Response:", response)

        # Save to file
        file.write("🧑‍💻 User Query:\n")
        file.write(query)
        file.write("\n\n")

        file.write("🤖 Agent Response:\n")
        file.write(str(response))
        file.write("\n\n")
        file.write("=" * 60)
        file.write("\n\n")

print("\n✅ All outputs saved to output5.txt")
