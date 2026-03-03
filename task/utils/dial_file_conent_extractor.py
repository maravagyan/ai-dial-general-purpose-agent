from __future__ import annotations

import re
from urllib.parse import urljoin

import httpx


class DialFileContentExtractor:
    """
    Downloads attached file bytes from DIAL Core using the file_url from chat,
    trying several known DIAL route prefixes (because setups differ),
    then converts to text.

    Supports: txt, csv, html/htm, pdf (best-effort).
    """

    def __init__(self, endpoint: str, api_key: str | None):
        self._endpoint = endpoint.rstrip("/") + "/"
        self._api_key = api_key

    async def extract_text(self, file_url: str) -> str:
        if not file_url:
            return ""

        headers = {}
        if self._api_key:
            # different DIAL setups accept one or the other; we send both
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["api-key"] = self._api_key

        # file_url looks like: "files/<tenant>/uploads/...."
        # Different DIAL builds serve it under different prefixes.
        candidates = self._build_candidate_urls(file_url)

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            last_err = None
            for url in candidates:
                try:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200 and resp.content:
                        content_type = (resp.headers.get("content-type") or "").lower()
                        return self._bytes_to_text(resp.content, content_type, url)

                    # Keep last error for debugging
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                except Exception as e:
                    last_err = str(e)

        return (
            "Error: Failed to download file from DIAL.\n"
            f"Tried URLs:\n- " + "\n- ".join(candidates) + "\n"
            f"Last error: {last_err}"
        )

    def _build_candidate_urls(self, file_url: str) -> list[str]:
        # normalize
        rel = file_url.lstrip("/")
        # common prefixes for DIAL deployments
        prefixes = [
            "",             # http://host:8080/files/...
            "api/",          # http://host:8080/api/files/...
            "v1/",           # http://host:8080/v1/files/...
            "openai/",       # http://host:8080/openai/files/...
            "openai/v1/",    # http://host:8080/openai/v1/files/...
        ]

        urls = [urljoin(self._endpoint, p + rel) for p in prefixes]

        # Some setups expect /files/... without the leading "files/"
        if rel.startswith("files/"):
            stripped = rel[len("files/"):]
            urls.extend([
                urljoin(self._endpoint, "files/" + stripped),
                urljoin(self._endpoint, "api/files/" + stripped),
                urljoin(self._endpoint, "v1/files/" + stripped),
                urljoin(self._endpoint, "openai/files/" + stripped),
                urljoin(self._endpoint, "openai/v1/files/" + stripped),
            ])

        # de-dup preserve order
        seen = set()
        out = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _bytes_to_text(self, data: bytes, content_type: str, url: str) -> str:
        filename = url.lower()

        if "text/plain" in content_type or filename.endswith(".txt"):
            return self._decode_text(data)

        if "text/csv" in content_type or filename.endswith(".csv"):
            text = self._decode_text(data)
            return self._csv_to_markdown(text)

        if "text/html" in content_type or filename.endswith(".html") or filename.endswith(".htm"):
            text = self._decode_text(data)
            return self._html_to_text(text)

        if "application/pdf" in content_type or filename.endswith(".pdf"):
            try:
                return self._pdf_to_text(data)
            except Exception as e:
                return f"Error: Unable to extract PDF text ({e})."

        # Fallback: decode as text
        return self._decode_text(data)

    @staticmethod
    def _decode_text(data: bytes) -> str:
        for enc in ("utf-8", "utf-16", "cp1251"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _csv_to_markdown(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""

        rows = [ln.split(",") for ln in lines]
        max_len = max(len(r) for r in rows)
        rows = [r + [""] * (max_len - len(r)) for r in rows]

        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []

        def esc(cell: str) -> str:
            return cell.replace("|", "\\|").strip()

        md = []
        md.append("| " + " | ".join(esc(c) for c in header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in body:
            md.append("| " + " | ".join(esc(c) for c in r) + " |")

        return "\n".join(md)

    @staticmethod
    def _html_to_text(html: str) -> str:
        html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
        html = re.sub(r"(?s)<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html).strip()
        return html

    @staticmethod
    def _pdf_to_text(data: bytes) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError("pypdf is not installed") from e

        import io
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()