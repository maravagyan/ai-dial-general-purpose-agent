from __future__ import annotations

import inspect
import json
from typing import Any, Optional

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


_SYSTEM_PROMPT = """
You are a helpful assistant answering a user question based ONLY on the provided document excerpts.

Rules:
- Use only the context given. If the answer isn't in the context, say you cannot find it in the document.
- Be concise and direct.
- If helpful, quote short phrases from the context (no long dumps).
""".strip()


async def _append(stage, text: str) -> None:
    res = stage.append_content(text)
    if inspect.isawaitable(res):
        await res


def _make_dial_client(endpoint: str, token: Optional[str]) -> AsyncDial:
    """
    token may be either:
    - DIAL api_key (often starts with 'dial-...')
    - Bearer token (JWT-like)
    AsyncDial supports either api_key OR bearer_token.
    """
    if not token:
        raise ValueError("No auth token provided to RagTool. Set DIAL_API_KEY or forward Authorization header.")

    # naive heuristic: DIAL api keys commonly start with dial-
    if token.startswith("dial-"):
        return AsyncDial(base_url=endpoint, api_key=token, api_version="2025-01-01-preview")

    return AsyncDial(base_url=endpoint, bearer_token=token, api_version="2025-01-01-preview")


class RagTool(BaseTool):
    """
    Performs semantic search in an uploaded document and answers based on the most relevant chunks.

    Use this when:
    - file extraction is paginated / large (Page #1. Total pages: N)
    - user asks a specific question about the doc (cleaning, troubleshooting, etc.)
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self._endpoint = endpoint
        self._deployment_name = deployment_name
        self._document_cache = document_cache

        # Embedding model (384 dims)
        self.model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2", device="cpu")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search INSIDE an attached document (TXT/PDF/CSV/HTML) and answer using only relevant excerpts. "
            "Prefer this tool for long/paginated documents (when extraction returns '**Page #1. Total pages: N**') "
            "or when user explicitly asks to use RAG. Use for questions like 'How should I clean the plate?' "
            "Return a direct answer grounded in the excerpts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document.",
                },
                "file_url": {
                    "type": "string",
                    "description": "URL of the attached file to search in.",
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        args = {}
        if tool_call_params.tool_call.function.arguments:
            args = json.loads(tool_call_params.tool_call.function.arguments)

        request = (args.get("request") or "").strip()
        file_url = (args.get("file_url") or "").strip()
        stage = tool_call_params.stage

        await _append(stage, "## Request arguments:\n")
        await _append(stage, f"**Request**: {request}\n\r")
        await _append(stage, f"**File URL**: {file_url}\n\r")

        if not request or not file_url:
            await _append(stage, "## Response:\n\rError: 'request' and 'file_url' are required.\n\r")
            return "Error: 'request' and 'file_url' are required."

        cache_document_key = f"{tool_call_params.conversation_id}::{file_url}"

        cached = self._document_cache.get(cache_document_key)
        if cached:
            index, chunks = cached
        else:
            extractor = DialFileContentExtractor(endpoint=self._endpoint, api_key=tool_call_params.api_key)
            text_content = await extractor.extract_text(file_url=file_url)

            if not text_content:
                await _append(stage, "## Response:\n\rError: File content not found.\n\r")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)

            embeddings = self.model.encode(chunks, convert_to_numpy=True).astype("float32")
            index = faiss.IndexFlatL2(384)
            index.add(np.asarray(embeddings, dtype="float32"))

            self._document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request], convert_to_numpy=True).astype("float32")

        distances, indices = index.search(query_embedding, k=3)

        retrieved_chunks: list[str] = []
        for idx in indices[0]:
            idx_int = int(idx)
            if 0 <= idx_int < len(chunks):
                retrieved_chunks.append(chunks[idx_int])

        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        await _append(stage, "## RAG Request:\n")
        await _append(stage, f"```text\n\r{augmented_prompt}\n\r```\n\r")
        await _append(stage, "## Response:\n")

        dial = _make_dial_client(self._endpoint, tool_call_params.api_key)

        stream = await dial.chat.completions.create(
            deployment_name=self._deployment_name,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
        )

        out = ""
        async for ev in stream:
            if not getattr(ev, "choices", None):
                continue
            if not ev.choices:
                continue

            delta = ev.choices[0].delta
            if not delta:
                continue

            content = getattr(delta, "content", None)
            if content:
                out += content
                await _append(stage, content)

        return out.strip()

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        ctx = "\n\n---\n\n".join(chunks) if chunks else "(no relevant context found)"
        return (
            f"User question:\n{request}\n\n"
            f"Relevant document excerpts:\n{ctx}\n\n"
            f"Answer the question using ONLY the excerpts above."
        )