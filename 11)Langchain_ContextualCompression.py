import os
import re
import sys
import warnings
from contextlib import redirect_stdout

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.documents import Document
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


SOURCE_DOCUMENTS = [
    Document(
        page_content=(
            "LangChain is a framework for building applications powered by large "
            "language models. It connects prompts, models, retrievers, memory, "
            "tools, and agents into reusable AI workflows."
        )
    ),
    Document(
        page_content=(
            "LangChain was launched as an open-source project in October 2022. "
            "It was created by Harrison Chase while he was working at Robust "
            "Intelligence."
        )
    ),
    Document(
        page_content=(
            "Contextual compression retrieves documents first and then keeps only "
            "the parts that are relevant to the user's question. This reduces "
            "noise before the context is sent to the model."
        )
    ),
    Document(
        page_content=(
            "Multi-query retrieval creates several versions of a user question, "
            "retrieves documents for each version, and merges the results to "
            "improve coverage."
        )
    ),
    Document(
        page_content=(
            "LangChain is useful for document question answering, chatbots, "
            "retrieval augmented generation, tool calling, and agentic systems "
            "that reason over external data."
        )
    ),
]


def tokenize(text):
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def expand_query_terms(question):
    terms = tokenize(question)
    question_lower = question.lower()

    if "created" in question_lower or "creator" in question_lower or "founder" in question_lower:
        terms.update({"harrison", "chase", "created", "launched"})

    if "feature" in question_lower or "workflow" in question_lower or "use" in question_lower:
        terms.update({"tools", "agents", "memory", "retrieval", "workflows"})

    if "compression" in question_lower:
        terms.update({"compression", "relevant", "noise", "context"})

    return terms


def retrieve_documents(question, limit=3):
    terms = expand_query_terms(question)
    scored = []

    for doc in SOURCE_DOCUMENTS:
        score = len(terms.intersection(tokenize(doc.page_content)))
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:limit] if score > 0]


def compress_documents(question, documents):
    terms = expand_query_terms(question)
    compressed = []

    for doc in documents:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        selected = [
            sentence
            for sentence in sentences
            if terms.intersection(tokenize(sentence))
        ]

        if selected:
            compressed.append(Document(page_content=" ".join(selected)))

    return compressed


def answer_from_context(question, compressed_docs, chat_history=None):
    question_lower = question.lower()

    if "created" in question_lower or "creator" in question_lower or "founder" in question_lower:
        answer = "LangChain was created by Harrison Chase."
    elif "compression" in question_lower:
        answer = (
            "Contextual compression retrieves documents and keeps only the "
            "question-relevant parts, which reduces noisy context."
        )
    elif "feature" in question_lower or "workflow" in question_lower or "use" in question_lower:
        answer = (
            "LangChain supports AI workflows with retrieval, memory, tools, "
            "agents, prompts, and model integrations."
        )
    else:
        answer = "LangChain is a framework for building LLM-powered applications."

    history_note = ""
    if chat_history:
        history_note = f"\nChat history used: {len(chat_history)} message(s)."

    context = "\n".join(f"- {doc.page_content}" for doc in compressed_docs)
    return f"{answer}{history_note}\n\nCompressed context:\n{context}"


def contextual_rag(question, memory=None):
    retrieved = retrieve_documents(question)
    compressed = compress_documents(question, retrieved)
    chat_history = memory.chat_memory.messages if memory else []
    answer = answer_from_context(question, compressed, chat_history)

    if memory:
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer.split("\n\n", 1)[0])

    return answer


def multi_query_variants(question):
    return [
        question,
        "Who created LangChain and when was it launched?",
        "What are the main LangChain features for AI applications?",
        "How does LangChain support retrieval and tools?",
    ]


def multi_query_rag(question):
    unique_docs = []
    seen = set()

    for variant in multi_query_variants(question):
        for doc in retrieve_documents(variant, limit=2):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

    compressed = compress_documents(question, unique_docs)
    return answer_from_context(question, compressed), compressed


def run_demo():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    def rag_tool_fn(question):
        return contextual_rag(question, memory)

    rag_tool = Tool(
        name="RAG_Tool",
        description="Answer LangChain questions with compressed retrieved context.",
        func=rag_tool_fn,
    )

    llm = FakeListChatModel(
        responses=[
            "Thought: I should use the RAG tool.\nAction: RAG_Tool\nAction Input: What is LangChain?",
            "Thought: I now know the answer.\nAI: LangChain is a framework for building LLM-powered applications.",
            "Thought: I should use the RAG tool and conversation context.\nAction: RAG_Tool\nAction Input: Who created LangChain?",
            "Thought: I now know the answer.\nAI: LangChain was created by Harrison Chase.",
        ]
    )

    agent = initialize_agent(
        tools=[rag_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )

    print("===== LangChain Contextual Compression Output =====\n")

    print("Source Documents:")
    for index, doc in enumerate(SOURCE_DOCUMENTS, start=1):
        print(f"{index}. {doc.page_content}")

    print("\n1. Contextual Compression Results")
    compressed_docs = compress_documents(
        "Who created LangChain?",
        retrieve_documents("Who created LangChain?"),
    )
    for doc in compressed_docs:
        print("-", doc.page_content)

    print("\n2. Conversational RAG")
    print("First Question")
    print(contextual_rag("What is LangChain?", memory))
    print("\nFollow-up")
    print(contextual_rag("Who created it?", memory))

    print("\n3. Agent Conversation")
    print("Answer:", agent.run("What is LangChain?"))
    print("Answer:", agent.run("Who created it?"))

    print("\n4. MultiQuery Retriever")
    question = "Tell me about LangChain creator and features."
    print("Query variants:")
    for variant in multi_query_variants(question):
        print("-", variant)

    multi_answer, sources = multi_query_rag(question)
    print("\nAnswer:")
    print(multi_answer)

    print("\nSources:")
    for doc in sources:
        print("-", doc.page_content)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output11.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
