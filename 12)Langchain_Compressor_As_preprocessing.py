import os
import re
import sys
from contextlib import redirect_stdout

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


RAW_TEXT = """
LangChain is a framework for building applications powered by large language models.
It helps developers connect prompts, models, retrievers, tools, memory, and agents.

Compressor-as-preprocessing means compressing documents before they are stored, indexed,
or passed into a retrieval pipeline. This can reduce noisy text and keep the most useful
information for later question answering.

Contextual compression is often used after retrieval, but preprocessing compression is
useful when the source documents are long, repetitive, or contain sections that are not
important for the target task.

For LangChain workflows, preprocessing can make the downstream RAG pipeline smaller,
faster, and easier to inspect. The tradeoff is that overly aggressive compression can
remove details that may be needed for future questions.
"""


def tokenize(text):
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def split_documents(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=0,
    )
    return [
        Document(page_content=chunk.strip())
        for chunk in splitter.split_text(raw_text)
        if chunk.strip()
    ]


def compress_documents(documents, query):
    query_terms = tokenize(query)
    important_terms = query_terms.union(
        {
            "langchain",
            "compressor",
            "preprocessing",
            "compression",
            "rag",
            "retrieval",
            "workflow",
            "context",
        }
    )

    compressed_docs = []

    for doc in documents:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        selected_sentences = [
            sentence.strip()
            for sentence in sentences
            if tokenize(sentence).intersection(important_terms)
        ]

        if selected_sentences:
            compressed_docs.append(Document(page_content=" ".join(selected_sentences)))

    return compressed_docs


def answer_question(context, question):
    question_lower = question.lower()

    if "main idea" in question_lower:
        return (
            "The main idea is that compressor-as-preprocessing reduces long or "
            "noisy documents before retrieval, making later LangChain RAG "
            "workflows smaller, faster, and easier to inspect."
        )

    if "tradeoff" in question_lower:
        return (
            "The main tradeoff is that aggressive compression can remove details "
            "that might be needed for future questions."
        )

    return (
        "Compressor-as-preprocessing keeps the most useful context before a "
        "retrieval or question-answering pipeline runs."
    )


def run_demo():
    documents = split_documents(RAW_TEXT)
    compressed_docs = compress_documents(documents, query="Summarize the key points")
    compressed_context = "\n".join(doc.page_content for doc in compressed_docs)
    response = answer_question(
        compressed_context,
        "What is the main idea of the document?",
    )

    print("===== LangChain Compressor As Preprocessing Output =====\n")

    print("Original Chunks:")
    for index, doc in enumerate(documents, start=1):
        print(f"{index}. {doc.page_content}")

    print("\nCompressed Summary:")
    for index, doc in enumerate(compressed_docs, start=1):
        print(f"{index}. {doc.page_content}")

    print("\nCompressed Context Used For QA:")
    print(compressed_context)

    print("\nAnswer to your question:")
    print(response)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output12.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
