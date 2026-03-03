from __future__ import annotations

import abc
from typing import Any

from aidial_sdk.chat_completion import Message
from aidial_client.types.chat import ToolParam, FunctionParam

from task.tools.models import ToolCallParams


class BaseTool(abc.ABC):
    """
    Base class for all tools.
    """

    @property
    @abc.abstractmethod
    def show_in_stage(self) -> bool:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    # ✅ THIS is where schema must be
    @property
    def schema(self) -> ToolParam:
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            ),
        )

    async def execute(self, tool_call_params: ToolCallParams) -> str | Message:
        return await self._execute(tool_call_params)

    @abc.abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        raise NotImplementedError()