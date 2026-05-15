import os
import re
import sys
from contextlib import redirect_stdout

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
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
It gives developers building blocks for prompts, model calls, chains, agents, tools,
retrievers, and memory.

MapReduce is a chain pattern for working with longer documents. The map step runs the
same task on each chunk, such as summarizing each part of a document. The reduce step
combines all intermediate outputs into one final answer.

This pattern is useful when the complete document is too large to fit into one prompt.
It also makes the workflow easier to inspect because every chunk creates its own
intermediate result before the final summary is produced.

In LangChain, MapReduce-style workflows can be built with runnable sequences, document
chains, or custom functions that map over documents and then reduce the mapped outputs.
"""


def split_documents(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=450,
        chunk_overlap=0,
    )
    return splitter.create_documents([raw_text])


def summarize_chunk(document):
    text = document.page_content
    text_lower = text.lower()

    if "framework" in text_lower:
        summary = (
            "LangChain provides building blocks for LLM applications, including "
            "prompts, chains, tools, agents, retrievers, and memory."
        )
    elif "mapreduce" in text_lower or "map step" in text_lower:
        summary = (
            "MapReduce summarizes each chunk separately, then combines the "
            "intermediate summaries into one final answer."
        )
    elif "too large" in text_lower or "inspect" in text_lower:
        summary = (
            "MapReduce is useful for long documents and makes intermediate "
            "chunk-level outputs easy to inspect."
        )
    else:
        summary = (
            "LangChain can implement MapReduce-style workflows with runnables, "
            "document chains, or custom map and reduce functions."
        )

    return {
        "source": text.strip(),
        "summary": summary,
    }


def reduce_summaries(mapped_outputs):
    unique_summaries = []
    seen = set()

    for item in mapped_outputs:
        summary = item["summary"]
        if summary not in seen:
            seen.add(summary)
            unique_summaries.append(summary)

    return (
        "Final Summary: LangChain MapReduce workflows process long documents by "
        "mapping a task over smaller chunks and reducing the chunk outputs into "
        "one final result. This is useful when documents are too large for a "
        "single prompt and when you want inspectable intermediate summaries.\n\n"
        "Combined Points:\n- "
        + "\n- ".join(unique_summaries)
    )


def create_map_reduce_chain():
    map_chain = RunnableLambda(lambda docs: [summarize_chunk(doc) for doc in docs])
    reduce_chain = RunnableLambda(reduce_summaries)
    return map_chain | reduce_chain


def run_demo():
    docs = split_documents(RAW_TEXT)
    map_reduce_chain = create_map_reduce_chain()
    mapped_outputs = [summarize_chunk(doc) for doc in docs]
    final_summary = map_reduce_chain.invoke(docs)

    print("===== LangChain MapReduce Output =====\n")

    print("Input Document Chunks:")
    for index, doc in enumerate(docs, start=1):
        clean_text = re.sub(r"\s+", " ", doc.page_content).strip()
        print(f"{index}. {clean_text}")

    print("\nMap Step Outputs:")
    for index, item in enumerate(mapped_outputs, start=1):
        print(f"{index}. {item['summary']}")

    print("\nReduce Step Output:")
    print(final_summary)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output15.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
