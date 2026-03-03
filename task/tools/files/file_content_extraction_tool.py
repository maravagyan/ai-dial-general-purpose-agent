import json
import inspect
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


async def _append(stage, text: str) -> None:
    """
    stage.append_content() can be sync (returns None) or async (returns awaitable),
    depending on SDK version. This helper supports both.
    """
    res = stage.append_content(text)
    if inspect.isawaitable(res):
        await res


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.

    PAGINATION: Files >10,000 chars are paginated.
    Response format: `**Page #X. Total pages: Y**` appears at end if paginated.

    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "file_content_extractor"

    @property
    def description(self) -> str:
        return (
            "Extracts text content from files (PDF, TXT, CSV, HTML/HTM). "
            "Supports pagination for large files (>10,000 chars). "
            "Returns markdown table for CSV. Use this tool when the user attaches a file and requests its content, "
            "or asks questions that require reading the file. For long files, fetch content page by page. "
            "Response format includes page info if paginated."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "URL to the file to extract content from."
                },
                "page": {
                    "type": "integer",
                    "description": "For large documents pagination is enabled. Each page consists of 10000 characters.",
                    "default": 1
                }
            },
            "required": ["file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments
        args = {}
        if tool_call_params.tool_call.function.arguments:
            args = json.loads(tool_call_params.tool_call.function.arguments)

        # 2. file_url
        file_url = args.get("file_url")

        # 3. page (default 1)
        page = args.get("page", 1)
        if page is None:
            page = 1

        # 4. stage
        stage = tool_call_params.stage

        # 5-8. Stage content (safe for sync/async append_content)
        await _append(stage, "## Request arguments: \n")
        await _append(stage, f"**File URL**: {file_url}\n\r")
        if isinstance(page, int) and page > 1:
            await _append(stage, f"**Page**: {page}\n\r")
        await _append(stage, "## Response: \n")

        # 9. Extract content
        extractor = DialFileContentExtractor(endpoint=self.endpoint, api_key=tool_call_params.api_key)
        content = await extractor.extract_text(file_url=file_url)

        # 10. Handle no content
        if not content:
            content = "Error: File content not found."

        # 11. Pagination
        if len(content) > 10_000:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size

            if not isinstance(page, int) or page < 1:
                page = 1

            if page > total_pages:
                content = f"Error: Page {page} does not exist. Total pages: {total_pages}"
            else:
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                page_content = content[start_index:end_index]
                content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"

        # 12. Show in stage
        await _append(stage, f"```text\n\r{content}\n\r```\n\r")

        # 13. Return
        return content