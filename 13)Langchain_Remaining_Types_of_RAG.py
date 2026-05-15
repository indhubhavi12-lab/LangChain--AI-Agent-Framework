import math
import os
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def tokenize(text):
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    ]


def keyword_score(query, text):
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    return len(query_terms.intersection(text_terms))


def retrieve_by_keywords(query, documents, limit=2):
    ranked = sorted(
        documents,
        key=lambda doc: keyword_score(query, doc.page_content),
        reverse=True,
    )
    return [doc for doc in ranked[:limit] if keyword_score(query, doc.page_content) > 0]


def parent_document_retriever(query):
    parent_docs = [
        Document(
            page_content=(
                "LangChain helps build LLM-powered apps with memory, retrieval, "
                "tools, and agents. Agents in LangChain can decide which tools "
                "to call in order to answer a question."
            ),
            metadata={"id": "parent-1"},
        ),
        Document(
            page_content=(
                "Retrievers find useful context from documents before an answer "
                "is generated. Parent document retrieval stores small child chunks "
                "but returns the larger parent document."
            ),
            metadata={"id": "parent-2"},
        ),
    ]

    child_chunks = []
    for parent in parent_docs:
        sentences = re.split(r"(?<=[.!?])\s+", parent.page_content)
        for sentence in sentences:
            if sentence:
                child_chunks.append(Document(sentence, {"parent_id": parent.metadata["id"]}))

    matched_children = retrieve_by_keywords(query, child_chunks, limit=2)
    matched_parent_ids = {child.metadata["parent_id"] for child in matched_children}

    return [
        parent
        for parent in parent_docs
        if parent.metadata["id"] in matched_parent_ids
    ]


def bm25_retriever(query, documents, limit=3):
    tokenized_docs = [tokenize(doc.page_content) for doc in documents]
    query_terms = tokenize(query)
    total_docs = len(documents)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / total_docs

    scores = []
    for doc, doc_tokens in zip(documents, tokenized_docs):
        score = 0.0
        doc_len = len(doc_tokens)

        for term in query_terms:
            term_frequency = doc_tokens.count(term)
            if term_frequency == 0:
                continue

            docs_with_term = sum(1 for tokens in tokenized_docs if term in tokens)
            idf = math.log(1 + (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))
            k1 = 1.5
            b = 0.75
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * numerator / denominator

        scores.append((score, doc))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scores[:limit] if score > 0]


def ensemble_retriever(query, documents):
    bm25_results = bm25_retriever(query, documents, limit=len(documents))
    keyword_results = retrieve_by_keywords(query, documents, limit=len(documents))

    scores = {}
    for rank, doc in enumerate(bm25_results, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 0.5 / rank

    for rank, doc in enumerate(keyword_results, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 0.5 / rank

    ranked = sorted(
        documents,
        key=lambda doc: scores.get(doc.page_content, 0),
        reverse=True,
    )

    return [doc for doc in ranked if scores.get(doc.page_content, 0) > 0]


def time_weighted_retriever(query, documents):
    now = datetime.now()
    scored = []

    for doc in documents:
        relevance = keyword_score(query, doc.page_content)
        last_accessed = doc.metadata["last_accessed_at"]
        age_hours = max((now - last_accessed).total_seconds() / 3600, 0)
        recency = math.exp(-0.05 * age_hours)
        scored.append((relevance + recency, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored]


def tavily_style_retriever(query):
    return [
        Document(
            page_content=(
                "TavilySearchAPIRetriever is used when a RAG app needs live web "
                "context instead of only local documents."
            ),
            metadata={"source": "local web-style demo"},
        ),
        Document(
            page_content=(
                "For a production LangChain app, web retrieval should be paired "
                "with source checking, summarization, and rate-limit handling."
            ),
            metadata={"source": "local web-style demo"},
        ),
        Document(
            page_content=(
                "This offline lesson keeps the Tavily example deterministic, so "
                "it can run without API keys or network access."
            ),
            metadata={"source": "local web-style demo"},
        ),
    ]


def run_demo():
    print("===== LangChain Remaining Types of RAG Output =====\n")

    print("1. ParentDocumentRetriever\n")
    parent_results = parent_document_retriever("What are agents?")
    for doc in parent_results:
        print("Retrieved Parent Doc:", doc.page_content)

    print("\n2. BM25Retriever\n")
    bm25_docs = [
        Document("LangChain enables LLM applications."),
        Document("Vector search is powerful for semantic retrieval."),
        Document("BM25 is a classical keyword retrieval method."),
    ]
    bm25_results = bm25_retriever("How does BM25 retrieval work?", bm25_docs)
    for doc in bm25_results:
        print("BM25 Result:", doc.page_content)

    print("\n3. EnsembleRetriever\n")
    ensemble_docs = [
        Document("LangChain supports LLMs."),
        Document("You can build AI apps using LangChain."),
        Document("Keyword retrieval and vector retrieval can be combined."),
    ]
    ensemble_results = ensemble_retriever("AI apps using LangChain", ensemble_docs)
    for doc in ensemble_results:
        print("Ensemble Doc:", doc.page_content)

    print("\n4. TimeWeightedVectorStoreRetriever\n")
    time_docs = [
        Document(
            "LangChain is for LLM-based apps.",
            metadata={"last_accessed_at": datetime.now() - timedelta(minutes=5)},
        ),
        Document(
            "Vector search improves relevance.",
            metadata={"last_accessed_at": datetime.now() - timedelta(days=2)},
        ),
        Document(
            "Recently accessed memories can rank higher in time-weighted retrieval.",
            metadata={"last_accessed_at": datetime.now() - timedelta(minutes=1)},
        ),
    ]
    time_results = time_weighted_retriever("What is LangChain?", time_docs)
    for doc in time_results:
        print("TimeWeighted Result:", doc.page_content)

    print("\n5. TavilySearchAPIRetriever\n")
    tavily_results = tavily_style_retriever("LangChain web retrieval")
    for doc in tavily_results:
        print("Tavily-Style Result:", doc.page_content)


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output13.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
