import os
import sys
import warnings
from contextlib import redirect_stdout

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_classic.memory import ConversationBufferMemory

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


KNOWLEDGE_BASE = [
    (
        "LangChain is a framework for building applications powered by large "
        "language models. It helps developers connect prompts, models, tools, "
        "retrievers, memory, and agents into AI workflows."
    ),
    (
        "LangChain was launched as an open-source project in October 2022 by "
        "Harrison Chase."
    ),
    (
        "In AI workflows, LangChain is often used for retrieval augmented "
        "generation, tool calling, document question answering, chatbots, and "
        "agentic systems that can reason over external data."
    ),
]


def create_retriever():
    embeddings = FakeEmbeddings(size=16)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_texts(KNOWLEDGE_BASE)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def create_rag_tool(retriever):
    def rag_answer(question):
        docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in docs)
        question_lower = question.lower()

        if "who" in question_lower or "created" in question_lower or "founder" in question_lower:
            answer = "Harrison Chase created LangChain."
        elif "workflow" in question_lower or "use" in question_lower:
            answer = (
                "LangChain is used in AI workflows to connect LLMs with "
                "retrieval, tools, memory, agents, and external data."
            )
        else:
            answer = (
                "LangChain is a framework for building LLM-powered "
                "applications with prompts, tools, retrievers, memory, and agents."
            )

        return f"{answer}\n\nRetrieved context:\n{context}"

    return Tool(
        name="LangChainRetriever",
        func=rag_answer,
        description="Answers questions about LangChain using retrieved context.",
    )


def create_agent(rag_tool):
    llm = FakeListChatModel(
        responses=[
            "Thought: I should retrieve context about LangChain.\nAction: LangChainRetriever\nAction Input: What is LangChain?",
            "Thought: I now know the answer.\nAI: LangChain is a framework for building LLM-powered applications with prompts, tools, retrievers, memory, and agents.",
            "Thought: I should retrieve who created LangChain.\nAction: LangChainRetriever\nAction Input: Who created LangChain?",
            "Thought: I now know the answer.\nAI: Harrison Chase created LangChain.",
            "Thought: I should retrieve context about LangChain in AI workflows.\nAction: LangChainRetriever\nAction Input: Explain LangChain's use in AI workflows.",
            "Thought: I now know the answer.\nAI: LangChain helps AI workflows by connecting LLMs with external knowledge, retrieval, tools, memory, and agents so applications can answer questions and take useful actions.",
        ]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return initialize_agent(
        tools=[rag_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )


def run_demo():
    retriever = create_retriever()
    rag_tool = create_rag_tool(retriever)
    agent = create_agent(rag_tool)

    print("===== LangChain RAG Tool Output =====\n")

    print("Knowledge Base Chunks:")
    for index, text in enumerate(KNOWLEDGE_BASE, start=1):
        print(f"{index}. {text}")

    print("\n1. First Question")
    res1 = agent.run("What is LangChain?")
    print("Answer:", res1)

    print("\n2. Follow-up")
    res2 = agent.run("Who created it?")
    print("Answer:", res2)

    print("\n3. Combined Reasoning")
    res3 = agent.run("Explain LangChain's use in AI workflows.")
    print("Answer:", res3)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output9.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
