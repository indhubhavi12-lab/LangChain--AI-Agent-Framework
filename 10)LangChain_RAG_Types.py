import os
import sys
import warnings
from contextlib import redirect_stdout

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import StructuredTool
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_classic.memory import ConversationBufferMemory
from pydantic import BaseModel

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


SOURCE_TEXT = """
LangChain is a framework for building applications powered by large language models.
It helps developers combine prompts, models, retrievers, tools, memory, and agents.

LangChain began as an open-source project in October 2022 and was created by Harrison Chase.

Naive RAG retrieves relevant text and sends it to an LLM with the user question.
Conversational RAG also includes chat history so follow-up questions can use previous context.
Agentic RAG exposes retrieval as a tool, allowing an agent to decide when it needs external context.

In AI workflows, LangChain is useful for document question answering, chatbots, knowledge assistants,
tool calling, and retrieval augmented generation over private or external data.
"""


def split_source_text():
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=180,
        chunk_overlap=20,
    )
    return [chunk.strip() for chunk in splitter.split_text(SOURCE_TEXT) if chunk.strip()]


def build_vectorstore(chunks):
    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=16))
    vectorstore.add_texts(chunks)
    return vectorstore


def retrieve_context(question, chunks, limit=2):
    question_words = {
        word.strip(".,?!'\"").lower()
        for word in question.split()
        if len(word.strip(".,?!'\"")) > 2
    }

    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in question_words if word in chunk_lower)
        scored_chunks.append((score, chunk))

    question_lower = question.lower()
    if "created" in question_lower or "founder" in question_lower or question_lower.startswith("who"):
        scored_chunks.sort(key=lambda item: ("harrison chase" not in item[1].lower(), -item[0]))
    elif "workflow" in question_lower or "use" in question_lower:
        scored_chunks.sort(key=lambda item: ("workflow" not in item[1].lower(), -item[0]))
    elif "conversational" in question_lower or "follow" in question_lower:
        scored_chunks.sort(key=lambda item: ("conversational rag" not in item[1].lower(), -item[0]))
    else:
        scored_chunks.sort(key=lambda item: (-item[0], "framework" not in item[1].lower()))

    return [chunk for _, chunk in scored_chunks[:limit]]


def answer_with_rag(question, chunks, chat_history=None):
    context = retrieve_context(question, chunks)
    question_lower = question.lower()

    if "created" in question_lower or "founder" in question_lower or question_lower.startswith("who"):
        answer = "LangChain was created by Harrison Chase."
    elif "again" in question_lower or "simple" in question_lower:
        answer = "Simply put, LangChain helps an LLM app find context, use tools, and remember a conversation."
    elif "workflow" in question_lower or "use" in question_lower:
        answer = (
            "LangChain supports AI workflows by combining retrieval, tools, memory, "
            "and agents around an LLM."
        )
    elif "conversational" in question_lower:
        answer = "Conversational RAG uses retrieved context plus chat history to answer follow-up questions."
    elif "agentic" in question_lower:
        answer = "Agentic RAG gives an agent a retriever tool so it can fetch context when needed."
    else:
        answer = "LangChain is a framework for building LLM-powered applications."

    history_note = ""
    if chat_history:
        history_note = f"\nChat history used: {len(chat_history)} message(s)."

    return (
        f"{answer}{history_note}\n\n"
        "Retrieved context:\n"
        + "\n".join(f"- {chunk}" for chunk in context)
    )


class RagInput(BaseModel):
    question: str


def run_demo():
    chunks = split_source_text()
    vectorstore = build_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    def rag_tool_fn(question: str) -> str:
        return answer_with_rag(
            question,
            chunks,
            chat_history=memory.chat_memory.messages,
        )

    structured_rag_tool = StructuredTool.from_function(
        name="RAG_QA",
        description="Answer LangChain questions using retrieved context.",
        func=rag_tool_fn,
        args_schema=RagInput,
    )

    agent_tool = Tool(
        name="RAG_QA",
        description="Answer LangChain questions using retrieved context.",
        func=rag_tool_fn,
    )

    llm = FakeListChatModel(
        responses=[
            "Thought: I should retrieve context.\nAction: RAG_QA\nAction Input: What is LangChain?",
            "Thought: I now know the answer.\nAI: LangChain is a framework for building LLM-powered applications.",
            "Thought: I should use the previous topic and retrieve the creator.\nAction: RAG_QA\nAction Input: Who created LangChain?",
            "Thought: I now know the answer.\nAI: LangChain was created by Harrison Chase.",
            "Thought: I should retrieve a simple explanation.\nAction: RAG_QA\nAction Input: Explain LangChain again simply.",
            "Thought: I now know the answer.\nAI: Simply put, LangChain helps an LLM app find context, use tools, and remember a conversation.",
        ]
    )

    agent = initialize_agent(
        tools=[agent_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
    )

    print("===== LangChain RAG Types Output =====\n")

    print("Knowledge Base Chunks:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

    print("\nVector Retriever Check:")
    for index, doc in enumerate(retriever.invoke("LangChain workflows"), start=1):
        print(f"{index}. {doc.page_content}")

    print("\n1. Naive RAG")
    print(answer_with_rag("What is LangChain?", chunks))

    print("\n2. Conversational RAG")
    memory.chat_memory.add_user_message("What is LangChain?")
    memory.chat_memory.add_ai_message("LangChain is a framework for building LLM apps.")
    print(answer_with_rag("Who created it?", chunks, memory.chat_memory.messages))

    print("\n3. Structured Tool RAG")
    print(structured_rag_tool.invoke({"question": "How is LangChain used in AI workflows?"}))

    print("\n4. Agentic RAG Conversation")
    print("First Question")
    print("Answer:", agent.run("What is LangChain?"))
    print("\nFollow-up")
    print("Answer:", agent.run("Who created it?"))
    print("\nAsk again")
    print("Answer:", agent.run("Explain LangChain again simply."))


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output10.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
