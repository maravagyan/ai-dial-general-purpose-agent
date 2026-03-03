import base64
import json
from typing import Any, Optional, Tuple, List

import httpx
from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


# -------------------------
# Small SDK compat helpers
# -------------------------

def _stage_append(stage: Any, text: str) -> None:
    """Compatibility helper: stage API differs across aidial_sdk versions."""
    if stage is None or not text:
        return
    for method in ("append_content", "add_content", "write", "append_text"):
        fn = getattr(stage, method, None)
        if callable(fn):
            try:
                fn(text)
                return
            except Exception:
                pass
    fn = getattr(stage, "add", None)
    if callable(fn):
        try:
            fn("content", text)
        except Exception:
            pass


def _stage_add_attachment(stage: Any, attachment: Attachment) -> None:
    if stage is None or attachment is None:
        return
    for method in ("add_attachment", "append_attachment"):
        fn = getattr(stage, method, None)
        if callable(fn):
            try:
                fn(attachment)
                return
            except Exception:
                pass
    fn = getattr(stage, "add", None)
    if callable(fn):
        try:
            fn("attachment", attachment)
        except Exception:
            pass


def _choice_add_attachment(choice: Any, attachment: Attachment) -> None:
    """Compatibility helper: Choice API differs by sdk version."""
    if choice is None or attachment is None:
        return
    for method in ("add_attachment", "append_attachment"):
        fn = getattr(choice, method, None)
        if callable(fn):
            try:
                fn(attachment)
                return
            except Exception:
                pass
    fn = getattr(choice, "add", None)
    if callable(fn):
        try:
            fn("attachment", attachment)
        except Exception:
            pass


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"raw": str(obj)}, ensure_ascii=False)


def _extract_text_result(execution_result: _ExecutionResult, tool_json: dict) -> str:
    """
    Provide a short result string to the model to avoid loops.
    Try common fields.
    """
    if getattr(execution_result, "result", None) is not None:
        return str(execution_result.result)

    out = getattr(execution_result, "output", None)
    if isinstance(out, list) and out:
        # Join first few lines only
        text = "\n".join(str(x) for x in out[:5])
        return text

    if "result" in tool_json:
        return str(tool_json["result"])
    if "output" in tool_json:
        return str(tool_json["output"])
    return ""


# -------------------------
# HTTP helpers
# -------------------------

async def _fetch_resource_fallback(url: str) -> Any:
    """
    Fallback if MCPClient doesn't have get_resource().
    Tries to GET resource and returns bytes (or json/text if possible).
    """
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            return r.json()
        return r.content


def _dial_upload_bytes(dial: Dial, data: bytes, upload_path: str, mime_type: str) -> None:
    """
    DIAL upload API differs by versions. Try several known signatures.
    """
    files = getattr(dial, "files", None)
    if files is None:
        raise RuntimeError("Dial client has no 'files' attribute")

    fn = getattr(files, "upload", None)
    if callable(fn):
        # 1) upload(data, path, mime_type=?)
        try:
            fn(data, upload_path, mime_type=mime_type)
            return
        except TypeError:
            pass
        except Exception:
            pass

        # 2) upload(path, data, mime_type=?)
        try:
            fn(upload_path, data, mime_type=mime_type)
            return
        except Exception:
            pass

    fn2 = getattr(files, "put", None)
    if callable(fn2):
        try:
            fn2(upload_path, data, mime_type=mime_type)
            return
        except Exception:
            pass

    raise RuntimeError("Could not upload file via Dial client (unsupported SDK signature).")


def _build_upload_path(dial: Dial, file_name: str) -> str:
    """
    Avoid hardcoding 'files/' twice.
    Typical dial.my_appdata_home is a pathlib-like object.
    We return a string path that Attachment(url=...) can use.
    """
    home = getattr(dial, "my_appdata_home", None)
    if home is None:
        # fallback – most cores accept 'files/<name>'
        return f"files/{file_name}"

    try:
        path_obj = home / file_name
        path_str = str(path_obj)
    except Exception:
        # if it's already a string
        path_str = f"{str(home).rstrip('/')}/{file_name}"

    # Ensure it starts with 'files/' exactly once if required by your core.
    # Some SDKs already include 'files/' prefix, some don't.
    if path_str.startswith("files/"):
        return path_str
    return f"files/{path_str.lstrip('/')}"


# -------------------------
# MCP response normalization
# -------------------------

def _normalize_mcp_tool_result(raw: Any) -> dict:
    """
    Normalize tool result to a dict compatible with _ExecutionResult.

    Supports:
      A) direct dict: {"output":[...], "files":[...], ...}
      B) MCP content blocks: {"content":[{"type":"text","text":"..."}, {"type":"resource","resource":{...}}]}
      C) stringified JSON
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {"raw": raw}

    if isinstance(raw, dict):
        # If already in expected format
        if "output" in raw or "files" in raw or "result" in raw:
            return raw

        # MCP content blocks shape
        content = raw.get("content")
        if isinstance(content, list):
            out: List[str] = []
            files: List[dict] = []

            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")

                # text block
                if t == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        out.append(text)

                # resource block (common mcp servers)
                elif t in ("resource", "file", "image"):
                    res = item.get("resource") or item.get("file") or item.get("image") or {}
                    if isinstance(res, dict):
                        # Common keys: uri, mimeType, name
                        uri = res.get("uri") or res.get("url")
                        name = res.get("name") or res.get("filename") or "artifact"
                        mime = res.get("mimeType") or res.get("mime_type") or "application/octet-stream"
                        if uri:
                            files.append({"url": uri, "name": name, "mime_type": mime})

            normalized: dict = {}
            if out:
                normalized["output"] = out
            if files:
                normalized["files"] = files

            # carry session_id if present in raw
            if "session_id" in raw:
                normalized["session_id"] = raw["session_id"]
            if "sessionId" in raw and "session_id" not in normalized:
                normalized["session_id"] = raw["sessionId"]

            # If nothing recognized, keep raw for debugging
            if not normalized:
                normalized["raw"] = raw

            return normalized

        # Unknown dict shape
        return {"raw": raw}

    # Unknown type
    return {"raw": str(raw)}


def _trim_for_stage(obj: Any, limit: int = 2000) -> Any:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
        if len(s) <= limit:
            return obj
        return {"trimmed": s[:limit] + "..."}
    except Exception:
        s = str(obj)
        if len(s) <= limit:
            return s
        return s[:limit] + "..."


def _resource_to_bytes(resource: Any, mime_type: str) -> bytes:
    """
    Convert fetched resource into bytes.
    - bytes: return as-is
    - str: treat as text OR base64
    - dict: try common keys
    """
    if isinstance(resource, bytes):
        return resource

    if isinstance(resource, str):
        # Try base64 for non-text
        if mime_type.startswith("text/") or mime_type in ("application/json", "application/xml"):
            return resource.encode("utf-8", errors="ignore")
        try:
            return base64.b64decode(resource)
        except Exception:
            return resource.encode("utf-8", errors="ignore")

    if isinstance(resource, dict):
        blob = resource.get("blob") or resource.get("data") or resource.get("content")
        if isinstance(blob, str):
            if mime_type.startswith("text/") or mime_type in ("application/json", "application/xml"):
                return blob.encode("utf-8", errors="ignore")
            try:
                return base64.b64decode(blob)
            except Exception:
                return blob.encode("utf-8", errors="ignore")
        return json.dumps(resource, ensure_ascii=False).encode("utf-8", errors="ignore")

    return json.dumps({"raw": str(resource)}, ensure_ascii=False).encode("utf-8", errors="ignore")


# -------------------------
# Tool
# -------------------------

class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    Wraps execution and (optionally) uploads produced artifacts to DIAL storage.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        mcp_tool_models: list[MCPToolModel],
        tool_name: str,
        dial_endpoint: str,
    ):
        """
        :param tool_name: actual name of tool that executes code. Often 'execute_code'.
        """
        self._dial_endpoint = dial_endpoint
        self._mcp_client = mcp_client

        self._code_execute_tool: Optional[MCPToolModel] = None
        for tm in mcp_tool_models:
            if getattr(tm, "name", None) == tool_name:
                self._code_execute_tool = tm
                break

        if self._code_execute_tool is None:
            available = [getattr(t, "name", "<?>") for t in mcp_tool_models]
            raise ValueError(
                f"PythonCodeInterpreterTool cannot be initialized: tool '{tool_name}' not found. "
                f"Available tools: {available}"
            )

    @classmethod
    async def create(cls, mcp_url: str, tool_name: str, dial_endpoint: str) -> "PythonCodeInterpreterTool":
        """
        Async factory method to create PythonCodeInterpreterTool.
        """
        mcp_client = MCPClient(mcp_url)
        tools_raw = await mcp_client.list_tools()

        mcp_tool_models: list[MCPToolModel] = []
        for t in tools_raw:
            try:
                mcp_tool_models.append(MCPToolModel.model_validate(t))  # type: ignore[attr-defined]
            except Exception:
                # If it's already MCPToolModel or unknown structure
                mcp_tool_models.append(t)  # type: ignore[arg-type]

        return cls(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tool_models,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint,
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name  # type: ignore[union-attr]

    @property
    def description(self) -> str:
        return getattr(self._code_execute_tool, "description", "")  # type: ignore[union-attr]

    @property
    def parameters(self) -> dict[str, Any]:
        params = getattr(self._code_execute_tool, "parameters", None)  # type: ignore[union-attr]
        return params or {"type": "object", "properties": {}}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        raw_args = tool_call_params.tool_call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}

        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return "Error: 'code' is required for python interpreter tool."

        session_id = args.get("session_id", None)

        stage = getattr(tool_call_params, "stage", None)
        choice = getattr(tool_call_params, "choice", None)

        _stage_append(stage, "## Request arguments:\n")
        _stage_append(stage, f"```python\n{code}\n```\n")
        if session_id:
            _stage_append(stage, f"**session_id**: {session_id}\n")
        else:
            _stage_append(stage, "New session will be created\n")

        payload: dict[str, Any] = {"code": code}
        if session_id is not None:
            payload["session_id"] = session_id

        # 1) Call MCP tool
        raw_tool_result = await self._mcp_client.call_tool(self.name, payload)

        # 2) Normalize MCP response
        tool_json = _normalize_mcp_tool_result(raw_tool_result)

        # 3) Validate into execution model
        try:
            execution_result = _ExecutionResult.model_validate(tool_json)
        except Exception as e:
            # If model validation fails, keep debug and return useful hint
            _stage_append(stage, f"\n⚠️ Failed to parse interpreter response: {e}\n")
            _stage_append(stage, f"```json\n{json.dumps(_trim_for_stage(tool_json), ensure_ascii=False, indent=2)}\n```\n")
            return _safe_json_dumps(
                {
                    "status": "error",
                    "error": "Interpreter response format unexpected",
                    "details": str(e),
                    "raw": tool_json,
                }
            )

        # 4) Upload artifacts (if any) and attach
        attachments: list[Attachment] = []

        if getattr(execution_result, "files", None):
            dial = Dial(
                base_url=self._dial_endpoint,
                api_key=tool_call_params.api_key,
                api_version="2025-01-01-preview",
            )

            for f in execution_result.files:
                file_name = getattr(f, "name", None) or "artifact"
                mime_type = getattr(f, "mime_type", None) or "application/octet-stream"
                resource_url = str(getattr(f, "url", ""))

                # Fetch resource via MCP client if method exists, else HTTP GET
                try:
                    get_res = getattr(self._mcp_client, "get_resource", None)
                    if callable(get_res):
                        resource = await get_res(resource_url)
                    else:
                        resource = await _fetch_resource_fallback(resource_url)

                    content_bytes = _resource_to_bytes(resource, mime_type)

                    upload_path = _build_upload_path(dial, file_name)
                    _dial_upload_bytes(dial, content_bytes, upload_path, mime_type)

                    att = Attachment(url=upload_path, type=mime_type, title=file_name)
                    attachments.append(att)

                    _stage_add_attachment(stage, att)
                    _choice_add_attachment(choice, att)

                except Exception as e:
                    _stage_append(stage, f"\n⚠️ Failed to fetch/upload resource '{file_name}': {e}\n")

            tool_json["dial_attachments"] = [a.model_dump() for a in attachments]

        # 5) Stage debug (normalized)
        _stage_append(stage, "\n## Interpreter response (normalized)\n")
        _stage_append(stage, f"```json\n{json.dumps(_trim_for_stage(tool_json, 6000), ensure_ascii=False, indent=2)}\n```\n")

        # 6) Return small payload to model (prevents infinite loops)
        short_result = _extract_text_result(execution_result, tool_json)

        response_for_model = {
            "status": "ok",
            "note": "Use this tool result to answer the user. Do NOT call execute_code again unless there is an error.",
            "result": (short_result or "")[:1500],
            "session_id": tool_json.get("session_id") or session_id,
            "attachments": tool_json.get("dial_attachments", []),
        }

        return _safe_json_dumps(response_for_model)