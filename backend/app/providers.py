"""LLM and embedding provider factory.

The rest of the app only sees LangChain's BaseChatModel and Embeddings interfaces,
so swapping OpenAI -> Azure OpenAI -> Anthropic -> Bedrock is a config change.

The 'offline' provider lets the whole app run with no keys:
  - HashingEmbeddings: a deterministic bag-of-words vector, good enough for keyword-ish retrieval.
  - OfflineExtractiveLLM: a tiny chat model that answers by extracting the context sentences
    that best match the question. It implements streaming, so the API/UI path is identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .config import settings

_WORD = re.compile(r"[a-z0-9$%]+")
_STOP = set(
    "a an the is are was were be to of and or in on for with what how do does i my me if "
    "it this that by at as can much when within after before am have has".split()
)


def _tokens(text: str) -> list[str]:
    toks = [t for t in _WORD.findall(text.lower()) if t not in _STOP]
    # crude stemming so "visits" ~ "visit", "drugs" ~ "drug"
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks]


# --------------------------------------------------------------------------- embeddings
class HashingEmbeddings(Embeddings):
    """Feature-hashed bag of words + bigrams, L2-normalised. No network, fully deterministic."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- offline LLM
class OfflineExtractiveLLM(BaseChatModel):
    """Answers from the <context> block in the prompt by picking the best-matching sentences.

    Stands in for a real LLM so the RAG pipeline, LangGraph agent, streaming API and UI can be
    exercised end to end without API keys. Real providers produce a written answer instead.
    """

    max_sentences: int = 2

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    def _answer(self, messages: list[BaseMessage]) -> str:
        prompt = "\n".join(str(m.content) for m in messages)
        ctx = re.search(r"<context>(.*?)</context>", prompt, re.S)
        q = re.search(r"Question:\s*(.*)", prompt)
        if not ctx or not q:
            return "I don't have enough information to answer that."
        q_toks = set(_tokens(q.group(1)))
        sents = []
        for idx, text in re.findall(r"\[(\d+)\](.*?)(?=\n\[\d+\]|\Z)", ctx.group(1), re.S):
            body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                if sent.strip():
                    sents.append((sent.strip(), idx, set(_tokens(sent))))
        # IDF-style weighting: a word in every sentence ("prior") counts less than a rare one ("expedited")
        df: dict[str, int] = {}
        for _, _, toks in sents:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        scored = []
        for sent, idx, toks in sents:
            score = sum(1.0 / df[t] for t in q_toks & toks)
            if score:
                scored.append((score, sent, idx))
        if not scored:
            return "I don't know based on the plan documents. Please contact member services."
        scored.sort(key=lambda s: -s[0])
        top = scored[0][0]
        best, seen = [], set()
        for score, sent, idx in scored:
            if score < top * 0.6 or len(best) == self.max_sentences:
                break
            if sent not in seen:
                seen.add(sent)
                best.append(f"{sent} [{idx}]")
        return " ".join(best)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._answer(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for word in re.findall(r"\S+\s*", self._answer(messages)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk


# --------------------------------------------------------------------------- factories
def get_llm() -> BaseChatModel:
    p = settings.llm_provider.lower()
    if p == "offline":
        return OfflineExtractiveLLM()
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)
    if p == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=settings.azure_chat_deployment,
            api_version=settings.azure_api_version,
            temperature=0,
        )
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, temperature=0)
    if p == "bedrock":
        from langchain_aws import ChatBedrockConverse  # pip install langchain-aws

        return ChatBedrockConverse(model=settings.bedrock_model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def get_embeddings() -> Embeddings:
    p = settings.embedding_provider.lower()
    if p == "offline":
        return HashingEmbeddings()
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    if p == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_embedding_deployment,
            api_version=settings.azure_api_version,
        )
    if p == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=settings.bedrock_embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider} (Anthropic has no embeddings API)")
