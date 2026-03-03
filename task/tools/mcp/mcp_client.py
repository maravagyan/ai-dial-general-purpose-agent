import json
from typing import Any, Optional

import httpx


class MCPClient:
    def __init__(self, mcp_url: str):
        url = mcp_url.rstrip("/")
        if not url.endswith("/mcp"):
            url = f"{url}/mcp"
        self._url = url + "/"
        self._session_id: Optional[str] = None
        self._initialized: bool = False

    async def _initialize_session(self) -> None:
        if self._initialized:
            return

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "general-purpose-agent",
                        "version": "0.1"
                    },
                },
            }

            async with client.stream(
                "POST",
                self._url,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload).encode("utf-8"),
            ) as r:

                if r.status_code >= 400:
                    txt = await r.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {r.status_code} - {txt.decode()}",
                        request=r.request,
                        response=r,
                    )

                # store session id
                sid = r.headers.get("mcp-session-id")
                if not sid:
                    raise RuntimeError("MCP initialize did not return mcp-session-id")

                self._session_id = sid

                # consume SSE result
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        obj = json.loads(data)
                        if obj.get("id") == "init":
                            self._initialized = True
                            return

        raise RuntimeError("MCP initialize failed")

    async def _rpc(self, method: str, params: Any, rpc_id: str) -> dict[str, Any]:
        await self._initialize_session()

        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream(
                "POST",
                self._url,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                    "mcp-session-id": self._session_id or "",
                },
                content=json.dumps(payload).encode("utf-8"),
            ) as r:

                if r.status_code >= 400:
                    txt = await r.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {r.status_code} - {txt.decode()}",
                        request=r.request,
                        response=r,
                    )

                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if not data:
                            continue
                        obj = json.loads(data)
                        if obj.get("id") == rpc_id:
                            if "error" in obj:
                                raise RuntimeError(obj["error"])
                            return obj

        raise TimeoutError("No JSON-RPC response received")

    async def list_tools(self) -> list[dict[str, Any]]:
        resp = await self._rpc("tools/list", None, "1")
        result = resp.get("result") or {}
        tools = result.get("tools") or result
        return tools if isinstance(tools, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        resp = await self._rpc("tools/call", {"name": name, "arguments": arguments}, "2")
        return resp.get("result", resp)