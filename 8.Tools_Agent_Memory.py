import os
import sys
import warnings
from contextlib import redirect_stdout

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory,
)

warnings.filterwarnings("ignore")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def simple_qa(question):
    question_lower = question.lower()

    if "created" in question_lower or "founder" in question_lower:
        return "Harrison Chase created LangChain."

    if "simple" in question_lower or "again" in question_lower:
        return "LangChain helps developers connect LLMs with prompts, tools, data, and memory."

    return (
        "LangChain is a framework for building applications with large language "
        "models, tools, retrieval, agents, and memory."
    )


def build_agent(llm, memory):
    qa_tool = Tool(
        name="Simple QA",
        func=simple_qa,
        description="Answer factual questions about LangChain clearly.",
    )

    return initialize_agent(
        tools=[qa_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
    )


def print_messages(memory):
    for msg in memory.chat_memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")


def run_demo():
    responses = [
        "Thought: Do I need to use a tool? No\nAI: LangChain is a framework for building applications powered by large language models.",
        "Thought: Do I need to use a tool? No\nAI: LangChain was created by Harrison Chase.",
        "Thought: Do I need to use a tool? No\nAI: In simple terms, LangChain helps an AI model use prompts, tools, data, and memory.",
        "Thought: Do I need to use a tool? No\nAI: LangChain helps developers build LLM apps with chains, agents, tools, and retrieval.",
        "Thought: Do I need to use a tool? No\nAI: Harrison Chase created LangChain.",
        "Thought: Do I need to use a tool? No\nAI: Simply, LangChain connects an LLM to useful context and actions.",
        "Thought: Do I need to use a tool? No\nAI: LangChain is an LLM application framework.",
        "The user asked what LangChain is, and the assistant explained that it is an LLM application framework.",
        "Thought: Do I need to use a tool? No\nAI: It was created by Harrison Chase.",
        "The conversation says LangChain is an LLM application framework created by Harrison Chase.",
        "Thought: Do I need to use a tool? No\nAI: Simply, it helps AI apps remember context and use tools.",
        "The conversation says LangChain is an LLM framework created by Harrison Chase that helps AI apps use memory and tools.",
        "Thought: Do I need to use a tool? No\nAI: LangChain is a framework for connecting LLMs with tools, data, and workflows.",
        "Thought: Do I need to use a tool? No\nAI: Harrison Chase is the founder of LangChain.",
    ]

    llm = FakeListChatModel(responses=responses)
    qa_tool = Tool(
        name="Simple QA",
        func=simple_qa,
        description="Answer factual questions about LangChain clearly.",
    )

    print("===== LangChain Tools, Agent, and Memory Output =====\n")

    print("Tool check:")
    print(qa_tool.run("What is LangChain?"))

    print("\n1. ConversationBufferMemory\n")
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    agent = build_agent(llm, buffer_memory)
    print("Answer:", agent.run("What is LangChain?"))
    print("Answer:", agent.run("Who created it?"))
    print("Answer:", agent.run("Explain it simply again."))
    print("\nStored Messages:")
    print_messages(buffer_memory)

    print("\n2. ConversationBufferWindowMemory\n")
    window_memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
    )
    agent = build_agent(llm, window_memory)
    print("Answer:", agent.run("What is LangChain?"))
    print("Answer:", agent.run("Who created it?"))
    print("Answer:", agent.run("Explain it simply again."))
    print("\nStored Messages:")
    print_messages(window_memory)

    print("\n3. ConversationSummaryMemory\n")
    summary_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=False,
    )
    agent = build_agent(llm, summary_memory)
    print("Answer:", agent.run("What is LangChain?"))
    print("Answer:", agent.run("Who created it?"))
    print("Answer:", agent.run("Explain it simply again."))
    print("\nRunning Summary:")
    print(summary_memory.load_memory_variables({})["chat_history"])

    print("\n4. VectorStoreRetrieverMemory\n")
    embeddings = FakeEmbeddings(size=8)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_texts(["initial memory: LangChain connects LLMs with tools and data."])
    retriever = vectorstore.as_retriever()
    vector_memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="chat_history",
    )
    agent = build_agent(llm, vector_memory)
    print("Answer:", agent.run("What is LangChain?"))
    print("Answer:", agent.run("Who created it?"))

    vector_memory.save_context(
        {"input": "Who is the founder of LangChain?"},
        {"output": "Harrison Chase is the founder of LangChain."},
    )

    query = "LangChain founder"
    print(f"\nRetrieval for: {query}")
    for index, doc in enumerate(retriever.invoke(query), start=1):
        print(f"{index}. {doc.page_content}")

    print("\nAll Indexed Texts:")
    for index, doc in enumerate(vectorstore.store.values(), start=1):
        print(f"{index}. {doc['text']}")


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output8.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
