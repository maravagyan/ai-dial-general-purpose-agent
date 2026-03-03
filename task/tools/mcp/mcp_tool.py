import json
from typing import Any, Dict

from aidial_sdk.chat_completion import Message, Role
from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.mcp.mcp_client import MCPClient


class MCPTool(BaseTool):
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any], client: MCPClient):
        self.name = name
        self.description = description
        self._schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema or {"type": "object", "properties": {}},
            },
        }
        self.client = client

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema

    async def execute(self, params: ToolCallParams):
        raw = params.tool_call.function.arguments or "{}"
        try:
            arguments = json.loads(raw)
        except Exception:
            arguments = {}

        result = await self.client.call_tool(self.name, arguments)

        # Return text to model (and visible in stage)
        return Message(role=Role.TOOL, content=json.dumps(result, ensure_ascii=False))