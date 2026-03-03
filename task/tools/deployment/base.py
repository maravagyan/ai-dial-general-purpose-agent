import json
from abc import ABC
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):
    """
    Generic tool that calls another DIAL deployment (model/app) via chat.completions.
    It streams text into stage and forwards attachments (images, files) to stage.
    """

    def __init__(self, endpoint: str, deployment_name: str):
        self.endpoint = endpoint
        self._deployment_name = deployment_name

    @property
    def deployment_name(self) -> str:
        return self._deployment_name

    @property
    def tool_parameters(self) -> dict[str, Any]:
        """
        Extra params for chat.completions.create(), e.g. {"temperature": 0}.
        Override if needed.
        """
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )

        raw_args = tool_call_params.tool_call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_args)
        except Exception:
            arguments = {}

        prompt = arguments.get("prompt", "")
        if "prompt" in arguments:
            del arguments["prompt"]

        # Show request parameters in stage
        tool_call_params.stage.append_content("## Request arguments\n")
        tool_call_params.stage.append_content(f"**prompt**: {prompt}\n")
        for k, v in arguments.items():
            tool_call_params.stage.append_content(f"**{k}**: {v}\n")

        # Send to target deployment
        chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "custom_fields": {
                    "configuration": {**arguments},
                }
            },
            **self.tool_parameters,
        )

        content = ""
        custom_content = CustomContent(attachments=[])

        tool_call_params.stage.append_content("\n## Response\n")

        async for chunk in chunks:
            if not getattr(chunk, "choices", None):
                continue
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            if getattr(delta, "content", None):
                content += delta.content
                tool_call_params.stage.append_content(delta.content)

            # Attachments (image) come via custom_content.attachments
            if getattr(delta, "custom_content", None) and delta.custom_content.attachments:
                attachments = delta.custom_content.attachments
                custom_content.attachments.extend(attachments)

                for a in attachments:
                    tool_call_params.stage.add_attachment(
                        type=a.type,
                        title=a.title,
                        data=a.data,
                        url=a.url,
                        reference_url=a.reference_url,
                        reference_type=a.reference_type,
                    )

        # Return Message so agent can append it to choice (so image shows in chat)
        return Message(
            role=Role.TOOL,
            content=StrictStr(content),
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )