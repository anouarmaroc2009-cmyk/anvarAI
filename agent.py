import json
from typing import Any
from openai import OpenAI
import config
import tools

TOOL_MAP: dict[str, Any] = {
    "read_file": tools.read_file,
    "write_file": tools.write_file,
    "run_command": tools.run_command,
    "web_search": tools.web_search,
    "git_operation": tools.git_operation,
}

SYSTEM_PROMPT = """You are Anvar AI, a coding agent created by Anouar, a web developer from Morocco. Everyone calls him Anvar and he prefers to stay anonymous. If anyone asks who made you or created you, tell them: "I was created by Anouar, a web developer from Morocco. Everyone calls him Anvar. He wants to stay anonymous."

You help users with software development tasks.

You have access to tools for:
- Reading files from the filesystem
- Writing files (create/edit code)
- Running shell commands
- Searching the web
- Running git operations

Think step by step. Use tools as needed to accomplish the user's goals.
When writing code, ensure correctness and follow best practices.
When running commands, check output and handle errors appropriately."""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if they don't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "workdir": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_operation",
            "description": "Run a git command (status, diff, add, commit, push, log, branch, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string", "description": "Git arguments, e.g. 'status'"},
                    "workdir": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["args"],
            },
        },
    },
]


class CodingAgent:
    def __init__(self):
        self.client = None

    def _get_client(self) -> OpenAI:
        if self.client is None:
            if not config.OPENCODE_API_KEY:
                raise ValueError(
                    "OPENCODE_API_KEY not set. "
                    "Get one free at https://opencode.ai/zen"
                )
            self.client = OpenAI(
                base_url="https://opencode.ai/zen/v1",
                api_key=config.OPENCODE_API_KEY,
            )
        return self.client

    @staticmethod
    def _extract_reasoning(obj) -> str:
        """Extract reasoning_content from an OpenAI response object."""
        if obj is None:
            return ""
        rc = getattr(obj, "reasoning_content", None)
        if rc:
            return rc
        extra = getattr(obj, "model_extra", None) or {}
        return extra.get("reasoning_content", "") or ""

    def _parse_tool_calls(self, choice) -> list[dict]:
        """Extract tool calls from an OpenAI chat completion choice."""
        if not choice.message.tool_calls:
            return []
        parts = []
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            parts.append({"functionCall": {"name": tc.function.name, "args": args}})
        return parts

    def _accumulate_tool_calls(self, tool_calls_accum: dict) -> list[dict]:
        """Convert accumulated streaming tool call chunks to BP format."""
        sorted_calls = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
        parts = []
        for tc in sorted_calls:
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})
        return parts

    def _build_messages(self, msgs: list) -> list:
        """Convert internal message list to OpenAI-format messages."""
        result = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in msgs:
            role = m.get("role", "user")
            parts = m.get("parts", [])
            reasoning = m.get("reasoning_content", "")
            content = ""
            tool_calls = None
            tool_call_id = None
            for p in parts:
                if "text" in p:
                    content += p["text"]
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls = [{
                        "id": f"call_{fc['name']}",
                        "type": "function",
                        "function": {"name": fc["name"], "arguments": json.dumps(fc.get("args", {}))},
                    }]
                elif "functionResponse" in p:
                    fr = p["functionResponse"]
                    tool_call_id = f"call_{fr['name']}"
                    content = json.dumps(fr["response"])

            msg = {"role": role}
            if tool_calls:
                msg["content"] = content or None
                msg["tool_calls"] = tool_calls
            elif tool_call_id:
                msg["role"] = "tool"
                msg["tool_call_id"] = tool_call_id
                msg["content"] = content
            else:
                msg["content"] = content

            if role == "assistant" and reasoning:
                msg["reasoning_content"] = reasoning

            result.append(msg)
        return result

    @staticmethod
    def execute_tool(name: str, args: dict) -> str:
        handler = TOOL_MAP.get(name)
        if not handler:
            return f"Error: unknown tool '{name}'"
        try:
            return str(handler(**args))
        except Exception as e:
            return f"Error: {e}"

    def process_message(self, user_message: str, stream: bool = False):
        """Process a message using Big Pickle via OpenCode Zen API. Yields dict events."""
        messages: list = []

        def add_msg(role: str, parts: list, reasoning: str = ""):
            msg = {"role": role, "parts": parts}
            if reasoning:
                msg["reasoning_content"] = reasoning
            messages.append(msg)

        add_msg("user", [{"text": user_message}])

        while True:
            try:
                bp_messages = self._build_messages(messages)
                kwargs = dict(
                    model=config.MODEL,
                    messages=bp_messages,
                    tools=TOOL_DEFS,
                    max_tokens=config.MAX_TOKENS,
                    stream=stream,
                )

                if stream:
                    collected_text = ""
                    collected_reasoning = ""
                    tool_calls_accum = {}
                    finish_reason = None

                    for chunk in self._get_client().chat.completions.create(**kwargs):
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if not delta:
                            continue

                        rc = self._extract_reasoning(delta)
                        if rc:
                            collected_reasoning += rc

                        if delta.content:
                            collected_text += delta.content
                            yield {"type": "text", "content": delta.content}

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_accum:
                                    tool_calls_accum[idx] = {"id": tc.id, "function": {"name": "", "arguments": ""}}
                                if tc.id:
                                    tool_calls_accum[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_accum[idx]["function"]["name"] += tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_accum[idx]["function"]["arguments"] += tc.function.arguments

                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason

                    if finish_reason == "tool_calls" and tool_calls_accum:
                        parts = self._accumulate_tool_calls(tool_calls_accum)
                        add_msg("assistant", parts, collected_reasoning)

                        result_parts = []
                        for tc in [tool_calls_accum[i] for i in sorted(tool_calls_accum)]:
                            name = tc["function"]["name"]
                            try:
                                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            result = self.execute_tool(name, args)
                            result_parts.append({
                                "functionResponse": {"name": name, "response": {"result": result}}
                            })
                            yield {"type": "tool", "name": name, "content": result[:200]}

                        add_msg("user", result_parts)
                        continue
                    else:
                        if collected_text:
                            yield {"type": "done"}
                        return

                else:
                    response = self._get_client().chat.completions.create(**kwargs)
                    choice = response.choices[0] if response.choices else None
                    if not choice:
                        yield {"type": "error", "content": "No response from model"}
                        return

                    reasoning = self._extract_reasoning(choice.message)

                    if choice.message.content:
                        add_msg("assistant", [{"text": choice.message.content}], reasoning)
                        yield {"type": "text", "content": choice.message.content}
                        yield {"type": "done"}
                        return

                    parts = self._parse_tool_calls(choice)
                    if parts:
                        add_msg("assistant", parts, reasoning)
                        result_parts = []
                        for tc in choice.message.tool_calls:
                            try:
                                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            except json.JSONDecodeError:
                                args = {}
                            result = self.execute_tool(tc.function.name, args)
                            result_parts.append({
                                "functionResponse": {"name": tc.function.name, "response": {"result": result}}
                            })
                            yield {"type": "tool", "name": tc.function.name, "content": result[:200]}

                        add_msg("user", result_parts)
                        continue

                    yield {"type": "done"}
                    return

            except Exception as e:
                yield {"type": "error", "content": f"Big Pickle API error: {e}"}
                return
