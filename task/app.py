import os
import uvicorn
import asyncio

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool

from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool

# Step 4: DDG MCP
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool

# Step 5: Python Interpreter MCP tool
# IMPORTANT: use the correct module path that exists in your repo.

DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT", "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

# MCP endpoints (recommend explicit /mcp)
DDG_MCP_URL = os.getenv("DDG_MCP_URL", "http://localhost:8051/mcp")
PY_INTERPRETER_MCP_URL = os.getenv("PY_INTERPRETER_MCP_URL", "http://localhost:8050/mcp")

# Interpreter MCP tool name (per task: execute_code)
PY_INTERPRETER_TOOL_NAME = os.getenv("PY_INTERPRETER_TOOL_NAME", "execute_code")


class GeneralPurposeAgentApplication(ChatCompletion):
    def __init__(self):
        self.tools: list[BaseTool] = []
        self._doc_cache: DocumentCache | None = None

    async def _create_tools(self) -> list[BaseTool]:
        self._doc_cache = self._doc_cache or DocumentCache.create()

        tools: list[BaseTool] = [
            # Step 2
            FileContentExtractionTool(endpoint=DIAL_ENDPOINT),
            # Step 3
            RagTool(endpoint=DIAL_ENDPOINT, deployment_name=DEPLOYMENT_NAME, document_cache=self._doc_cache),
            # Step 3 (Image gen)
            ImageGenerationTool(endpoint=DIAL_ENDPOINT, deployment_name="dall-e-3"),
        ]

        # -------------------------
        # Step 5: Python Interpreter
        # -------------------------
        try:
            py_tool = await asyncio.wait_for(
                PythonCodeInterpreterTool.create(
                    mcp_url=PY_INTERPRETER_MCP_URL,
                    tool_name=PY_INTERPRETER_TOOL_NAME,
                    dial_endpoint=DIAL_ENDPOINT,
                ),
                timeout=20,
            )
            tools.append(py_tool)
            print(f"[PY] Loaded Python interpreter from {PY_INTERPRETER_MCP_URL} (tool={PY_INTERPRETER_TOOL_NAME})")
        except Exception as e:
            print(f"[PY] Failed to init Python interpreter from {PY_INTERPRETER_MCP_URL}: {type(e).__name__}: {e}")

        # -------------------------
        # Step 4: DDG MCP tools
        # -------------------------
        try:
            ddg_client = MCPClient(DDG_MCP_URL)
            tools_spec = await asyncio.wait_for(ddg_client.list_tools(), timeout=20)

            if not isinstance(tools_spec, list):
                print(f"[MCP] Unexpected tools list type from {DDG_MCP_URL}: {type(tools_spec)}")
                tools_spec = []

            print(f"[MCP] Loaded {len(tools_spec)} tools from {DDG_MCP_URL}")

            for t in tools_spec:
                # Be defensive: MCP servers sometimes use inputSchema vs input_schema
                name = t.get("name")
                if not name:
                    continue
                tools.append(
                    MCPTool(
                        name=name,
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}},
                        client=ddg_client,
                    )
                )
        except Exception as e:
            print(f"[MCP] Failed to load tools from {DDG_MCP_URL}: {type(e).__name__}: {e}")

        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )

            await agent.handle_request(
                deployment_name=DEPLOYMENT_NAME,
                choice=choice,
                request=request,
                response=response,
            )


app = DIALApp()
agent_app = GeneralPurposeAgentApplication()
app.add_chat_completion(deployment_name="general-purpose-agent", impl=agent_app)

if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")