import os
import json
from dataclasses import dataclass
from typing import Optional, Mapping, List, Tuple, Dict, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Choice, Request, Response, Role, Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


# ----------------------------
# Compatibility helpers (SDK)
# ----------------------------

def _choice_emit_text(choice: Choice, response: Response, text: str) -> None:
    """
    Compatibility writer for different aidial_sdk versions.
    Tries common methods on Choice/Response to stream assistant text.
    """
    if not text:
        return

    msg = Message(role=Role.ASSISTANT, content=text)

    for obj in (choice, response):
        # message-based methods
        for method_name in ("append_message", "add_message", "write_message", "push_message"):
            fn = getattr(obj, method_name, None)
            if callable(fn):
                fn(msg)
                return

        # text/delta-based methods
        for method_name in ("append_content", "write", "send_text", "delta"):
            fn = getattr(obj, method_name, None)
            if callable(fn):
                try:
                    fn(text)
                    return
                except TypeError:
                    # e.g. delta expects a Message
                    try:
                        fn(msg)
                        return
                    except Exception:
                        pass


def _choice_emit_message(choice: Choice, response: Response, message: Message) -> None:
    """
    Compatibility writer for tool-returned Message objects (attachments, etc.).
    """
    if message is None:
        return

    for obj in (choice, response):
        for method_name in ("append_message", "add_message", "write_message", "push_message"):
            fn = getattr(obj, method_name, None)
            if callable(fn):
                fn(message)
                return


class _NoopStage:
    def open(self) -> None:
        return

    def close(self) -> None:
        return


def _create_stage(choice: Choice, response: Response, name: str):
    """
    Some SDK versions may not have choice.create_stage. Try alternatives, else no-op.
    """
    for obj in (choice, response):
        fn = getattr(obj, "create_stage", None)
        if callable(fn):
            try:
                return fn(name)
            except Exception:
                pass
    return _NoopStage()


async def _force_final_answer(
    dial: AsyncDial,
    deployment_name: str,
    messages: List[Dict[str, Any]],
    choice: Choice,
    response: Response,
) -> None:
    """
    If the streamed turn produced no assistant content (common after tool execution),
    force one more completion WITHOUT tools to generate final answer text.
    """
    followup = await dial.chat.completions.create(
        deployment_name=deployment_name,
        stream=True,
        messages=messages,
    )

    async for ch in followup:
        if not getattr(ch, "choices", None):
            continue
        delta = getattr(ch.choices[0], "delta", None)
        if not delta:
            continue
        if getattr(delta, "content", None):
            _choice_emit_text(choice, response, delta.content)


# ----------------------------
# Auth helpers
# ----------------------------

def _extract_bearer(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    if not headers:
        return None
    auth = headers.get("Authorization") or headers.get("authorization")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def build_dial_client(endpoint: str, headers: Optional[Mapping[str, str]]) -> Tuple[AsyncDial, str]:
    """
    Returns (AsyncDial client, api_key_used).
    api_key_used is needed because AsyncDial doesn't expose api_key as a property.
    """
    env_key = os.getenv("DIAL_API_KEY")
    bearer = _extract_bearer(headers)
    api_key = bearer or env_key

    if not api_key:
        raise ValueError("No auth provided. Set DIAL_API_KEY env var or forward Authorization header.")

    return AsyncDial(base_url=endpoint, api_key=api_key, api_version="2025-01-01-preview"), api_key


def _get_conversation_id(request: Request) -> str:
    """
    aidial_sdk Request schema differs by version.
    Some versions don't have request.conversation_id.
    This helper safely extracts a stable id for caching/tools.
    """
    for attr in ("conversation_id", "conversationId", "conversation", "thread_id", "threadId"):
        val = getattr(request, attr, None)
        if isinstance(val, str) and val.strip():
            return val

    meta = getattr(request, "metadata", None) or getattr(request, "meta", None)
    if isinstance(meta, dict):
        for k in ("conversation_id", "conversationId", "thread_id", "threadId"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v

    return "default"


# ----------------------------
# Tool-call stream merge helpers
# ----------------------------

def _merge_tool_call_delta(state: Dict[int, Dict[str, Any]], tc_delta: Any) -> None:
    """
    Merge streamed tool_call delta into state keyed by tc_delta.index.
    """
    idx = getattr(tc_delta, "index", None)
    if idx is None:
        return

    item = state.get(idx) or {
        "id": None,
        "type": None,
        "function": {"name": None, "arguments": ""},
    }

    if getattr(tc_delta, "id", None):
        item["id"] = tc_delta.id

    if getattr(tc_delta, "type", None):
        item["type"] = tc_delta.type

    fn = getattr(tc_delta, "function", None)
    if fn is not None:
        if getattr(fn, "name", None):
            item["function"]["name"] = fn.name
        if getattr(fn, "arguments", None):
            item["function"]["arguments"] += fn.arguments  # streamed chunks

    state[idx] = item


def _finalize_tool_calls(state: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert merged state into tool_calls list for messages[].tool_calls.
    Ensures:
      - type defaults to "function"
      - arguments defaults to "{}"
      - includes only entries with id + function.name
    """
    result: List[Dict[str, Any]] = []
    for idx in sorted(state.keys()):
        item = state[idx]
        fn = item.get("function") or {}
        name = fn.get("name")
        call_id = item.get("id")

        if not name or not call_id:
            continue

        tc_type = item.get("type") or "function"
        args = (fn.get("arguments") or "").strip()
        if not args:
            args = "{}"

        try:
            json.loads(args)
        except Exception:
            args = "{}"

        result.append(
            {
                "id": call_id,
                "type": tc_type,
                "function": {"name": name, "arguments": args},
            }
        )
    return result


# --- Small wrappers so tools can use attribute access: tool_call.function.arguments ---

@dataclass
class _ToolCallFunction:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _ToolCallFunction


# ----------------------------
# Agent
# ----------------------------

class GeneralPurposeAgent:
    """
    General agent that:
    - calls LLM with tool schemas
    - executes tool calls
    - streams assistant output to choice/response
    """

    def __init__(self, endpoint: str, system_prompt: str, tools: List[BaseTool]):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools

    async def handle_request(
        self,
        deployment_name: str,
        choice: Choice,
        request: Request,
        response: Response,
    ) -> None:
        dial, api_key_used = build_dial_client(self.endpoint, request.headers)

        tool_schemas = [t.schema for t in self.tools]
        tool_by_name = {t.name: t for t in self.tools}

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if getattr(request, "messages", None):
            for m in request.messages:
                messages.append({"role": m.role, "content": m.content})

        max_tool_rounds = 8
        tool_round = 0

        while True:
            tool_round += 1
            if tool_round > max_tool_rounds:
                _choice_emit_text(choice, response, "Stopped: too many tool-call rounds (possible tool loop).")
                return

            stream = await dial.chat.completions.create(
                deployment_name=deployment_name,
                stream=True,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )

            assistant_text_parts: List[str] = []
            tool_call_state: Dict[int, Dict[str, Any]] = {}

            async for ch in stream:
                if not getattr(ch, "choices", None):
                    continue
                delta = getattr(ch.choices[0], "delta", None)
                if not delta:
                    continue

                if getattr(delta, "content", None):
                    assistant_text_parts.append(delta.content)
                    _choice_emit_text(choice, response, delta.content)

                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        _merge_tool_call_delta(tool_call_state, tc)

            assistant_text = "".join(assistant_text_parts).strip()
            tool_calls = _finalize_tool_calls(tool_call_state)

            # No tool calls -> finalize response
            if not tool_calls:
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
                    return

                # If assistant text is empty but we likely have tool results in messages,
                # force one more completion (no tools) to get final answer text.
                await _force_final_answer(dial, deployment_name, messages, choice, response)
                return

            # Add assistant tool-call message to history
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text or "",
                    "tool_calls": tool_calls,
                }
            )

            # Execute tools
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool = tool_by_name.get(tool_name)

                if not tool:
                    tool_text_for_model = f"Error: tool '{tool_name}' not found."
                else:
                    stage = _create_stage(choice, response, tool_name)
                    try:
                        stage.open()
                    except Exception:
                        pass

                    tool_call_obj = _ToolCall(
                        id=tc["id"],
                        type=tc.get("type") or "function",
                        function=_ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"].get("arguments") or "{}",
                        ),
                    )

                    params = ToolCallParams(
                        tool_call=tool_call_obj,
                        api_key=api_key_used,
                        conversation_id=_get_conversation_id(request),
                        stage=stage,
                        choice=choice,  # required by your ToolCallParams
                    )

                    try:
                        res = await tool.execute(params)

                        # If tool returned Message (with attachments), forward it to chat
                        if isinstance(res, Message):
                            _choice_emit_message(choice, response, res)
                            tool_text_for_model = (res.content or "").strip()
                        else:
                            tool_text_for_model = (res or "").strip()

                    except Exception as e:
                        tool_text_for_model = f"Error while executing tool '{tool_name}': {e}"
                    finally:
                        try:
                            stage.close()
                        except Exception:
                            pass

                # Tool result for the model's context
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": tool_text_for_model,
                    }
                )