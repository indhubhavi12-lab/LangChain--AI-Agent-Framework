import os
import re
import sys
from contextlib import redirect_stdout


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


DOCUMENTS = [
    (
        "LangChain is a framework for building applications powered by large "
        "language models. It provides reusable pieces for prompts, models, "
        "retrievers, memory, tools, agents, and chains."
    ),
    (
        "A chain connects multiple steps together. One step might format a "
        "prompt, another might call a model, and another might parse or refine "
        "the output."
    ),
    (
        "Stuff document chains combine documents into one context block before "
        "answering. Refine chains build an answer from one document and improve "
        "it as more documents are read."
    ),
    (
        "Map-reduce chains summarize or answer each document separately, then "
        "combine the intermediate outputs into a final result."
    ),
    (
        "Retrieval QA chains first retrieve relevant documents and then answer "
        "the user's question using only that retrieved context."
    ),
]


def tokenize(text):
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def simple_llm(prompt):
    prompt_lower = prompt.lower()

    if "summary" in prompt_lower or "summarize" in prompt_lower:
        return (
            "LangChain chains connect prompts, model calls, retrieved context, "
            "and output processing into repeatable workflows."
        )

    if "document about" in prompt_lower:
        return (
            "The document explains LangChain chain patterns such as stuff, "
            "refine, map-reduce, and retrieval QA."
        )

    if "keywords" in prompt_lower:
        return "chains, retrieval, documents, prompts, workflows"

    return (
        "LangChain helps developers build LLM applications by composing "
        "reusable steps."
    )


def llm_chain(question):
    prompt = f"Answer clearly: {question}"
    return simple_llm(prompt)


def sequential_chain(topic):
    summary = simple_llm(f"Write a summary about {topic}.")
    keywords = simple_llm(f"Extract keywords from this summary: {summary}")
    return summary, keywords


def stuff_documents_chain(question, documents):
    context = "\n".join(documents)
    prompt = f"Use this context to answer: {context}\n\nQuestion: {question}"
    return simple_llm(prompt)


def refine_documents_chain(documents):
    summary = f"Initial summary: {documents[0]}"

    for document in documents[1:]:
        if "chain" in document.lower() or "retrieval" in document.lower():
            summary += " " + document

    return (
        "Refined summary: LangChain uses chains to connect prompts, models, "
        "documents, retrieval, refinement, and final answers."
    )


def map_reduce_chain(documents):
    mapped = []
    for index, document in enumerate(documents, start=1):
        if "stuff" in document.lower():
            mapped.append(f"Doc {index}: Stuff chains combine documents into one context.")
        elif "refine" in document.lower():
            mapped.append(f"Doc {index}: Refine chains improve an answer step by step.")
        elif "map-reduce" in document.lower():
            mapped.append(f"Doc {index}: Map-reduce chains process chunks separately and merge results.")
        elif "retrieval" in document.lower():
            mapped.append(f"Doc {index}: Retrieval QA answers from relevant documents.")
        else:
            mapped.append(f"Doc {index}: LangChain provides chain building blocks.")

    reduced = (
        "Combined answer: chain types organize LLM workflows by stuffing context, "
        "refining across documents, mapping and reducing chunk outputs, or "
        "retrieving relevant context before answering."
    )
    return mapped, reduced


def retrieval_qa_chain(question, documents):
    query_terms = tokenize(question)
    ranked = sorted(
        documents,
        key=lambda document: len(query_terms.intersection(tokenize(document))),
        reverse=True,
    )
    retrieved = ranked[:2]

    if "retrieval" in question.lower():
        answer = (
            "Retrieval QA chains search for relevant documents first, then use "
            "that context to answer the question."
        )
    else:
        answer = stuff_documents_chain(question, retrieved)

    return retrieved, answer


def run_demo():
    print("===== LangChain Types of Chains Output =====\n")

    print("Source Documents:")
    for index, document in enumerate(DOCUMENTS, start=1):
        print(f"{index}. {document}")

    print("\n1. LLMChain Style")
    print("Question: What is LangChain?")
    print("Answer:", llm_chain("What is LangChain?"))

    print("\n2. SequentialChain Style")
    summary, keywords = sequential_chain("LangChain chains")
    print("Step 1 Summary:", summary)
    print("Step 2 Keywords:", keywords)

    print("\n3. StuffDocumentsChain Style")
    print(
        "Answer:",
        stuff_documents_chain("What is this document about?", DOCUMENTS[:3]),
    )

    print("\n4. RefineDocumentsChain Style")
    print(refine_documents_chain(DOCUMENTS))

    print("\n5. MapReduceDocumentsChain Style")
    mapped, reduced = map_reduce_chain(DOCUMENTS)
    print("Mapped Outputs:")
    for item in mapped:
        print("-", item)
    print("Reduced Output:", reduced)

    print("\n6. RetrievalQA Chain Style")
    retrieved, answer = retrieval_qa_chain(
        "How does retrieval QA work?",
        DOCUMENTS,
    )
    print("Retrieved Documents:")
    for document in retrieved:
        print("-", document)
    print("Answer:", answer)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output14.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
