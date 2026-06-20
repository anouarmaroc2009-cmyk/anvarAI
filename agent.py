import json
import requests
from typing import Any
from openai import OpenAI
from google import genai
from google.genai import types
import config
import tools

TOOL_DEFS = [
    types.FunctionDeclaration(
        name="read_file",
        description="Read the contents of a file from the local filesystem.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(type=types.Type.STRING, description="Path to the file"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Write content to a file. Creates parent directories if they don't exist.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "path": types.Schema(type=types.Type.STRING, description="Path to the file"),
                "content": types.Schema(type=types.Type.STRING, description="Content to write"),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_command",
        description="Execute a shell command and return its output.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "command": types.Schema(type=types.Type.STRING, description="Shell command to execute"),
                "workdir": types.Schema(
                    type=types.Type.STRING,
                    description="Working directory (optional)",
                ),
            },
            required=["command"],
        ),
    ),
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current information.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="Search query"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="git_operation",
        description="Run a git command (status, diff, add, commit, push, log, branch, etc.).",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "args": types.Schema(
                    type=types.Type.STRING,
                    description="Git arguments, e.g. 'status', 'log --oneline -5'",
                ),
                "workdir": types.Schema(type=types.Type.STRING, description="Working directory (optional)"),
            },
            required=["args"],
        ),
    ),
]

TOOL_MAP: dict[str, Any] = {
    "read_file": tools.read_file,
    "write_file": tools.write_file,
    "run_command": tools.run_command,
    "web_search": tools.web_search,
    "git_operation": tools.git_operation,
}

# Tool definitions as plain dicts for REST API calls
TOOL_DEFS_DICT = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the local filesystem.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"path": {"type": "STRING", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if they don't exist.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {"type": "STRING", "description": "Path to the file"},
                "content": {"type": "STRING", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command and return its output.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "Shell command to execute"},
                "workdir": {"type": "STRING", "description": "Working directory (optional)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "git_operation",
        "description": "Run a git command (status, diff, add, commit, push, log, branch, etc.).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "args": {"type": "STRING", "description": "Git arguments, e.g. 'status'"},
                "workdir": {"type": "STRING", "description": "Working directory (optional)"},
            },
            "required": ["args"],
        },
    },
]

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


class CodingAgent:
    def __init__(self):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.messages: list[types.Content] = []
        self.tool_config = types.Tool(function_declarations=TOOL_DEFS)

    def process_message(self, user_message: str, stream: bool = False):
        self.messages.append(types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        ))

        while True:
            try:
                if stream:
                    collected_parts = []
                    for chunk in self.client.models.generate_content_stream(
                        model=config.MODEL,
                        contents=self.messages,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=[self.tool_config],
                            max_output_tokens=config.MAX_TOKENS,
                        ),
                    ):
                        if not chunk.candidates:
                            continue
                        for p in chunk.candidates[0].content.parts:
                            if p.text:
                                yield {"type": "text", "content": p.text}
                            collected_parts.append(p)
                    response_content = types.Content(role="model", parts=collected_parts) if collected_parts else None
                else:
                    response = self.client.models.generate_content(
                        model=config.MODEL,
                        contents=self.messages,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=[self.tool_config],
                            max_output_tokens=config.MAX_TOKENS,
                        ),
                    )
                    if not response.candidates:
                        feedback = response.prompt_feedback
                        reason = feedback.block_reason if feedback else "unknown"
                        return f"Response blocked by safety filters ({reason})"
                    candidate = response.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        break
                    response_content = candidate.content

            except Exception as e:
                error = f"Gemini API error: {e}"
                if stream:
                    yield {"type": "error", "content": error}
                    return
                return error

            if not response_content:
                break

            self.messages.append(response_content)
            function_calls = [p for p in response_content.parts if p.function_call]

            if not function_calls:
                break

            result_parts = []
            for part in function_calls:
                fc = part.function_call
                handler = TOOL_MAP.get(fc.name)
                if not handler:
                    result = f"Error: unknown tool '{fc.name}'"
                else:
                    try:
                        args = {k: v for k, v in fc.args.items()}
                        result = handler(**args)
                    except Exception as e:
                        result = f"Error executing {fc.name}: {e}"
                result_parts.append(types.Part.from_function_response(
                    name=fc.name,
                    response={"result": str(result)},
                ))
                if stream:
                    yield {"type": "tool", "name": fc.name, "content": str(result)[:200]}

            self.messages.append(types.Content(role="user", parts=result_parts))

        if stream:
            yield {"type": "done"}
        else:
            text = response.text if response.text else ""
            text_parts = [p.text for p in (response_content.parts if response_content else []) if p.text]
            return text or "\n".join(text_parts) or "(no text response)"

    def get_history(self) -> list[dict]:
        cleaned = []
        for msg in self.messages:
            parts = []
            for p in msg.parts:
                if p.text:
                    parts.append({"type": "text", "text": p.text})
                elif p.function_call:
                    parts.append({"type": "function_call", "name": p.function_call.name, "args": dict(p.function_call.args)})
                elif p.function_response:
                    parts.append({"type": "function_response", "name": p.function_response.name, "response": dict(p.function_response.response)})
            cleaned.append({"role": msg.role, "parts": parts})
        return cleaned

    def reset(self) -> None:
        self.messages = []

    # === User OAuth token methods (uses user's own Gemini quota) ===

    @staticmethod
    def execute_tool(name: str, args: dict) -> str:
        handler = TOOL_MAP.get(name)
        if not handler:
            return f"Error: unknown tool '{name}'"
        try:
            return str(handler(**args))
        except Exception as e:
            return f"Error: {e}"

    def _call_gemini_rest(self, token: str, contents: list, stream: bool = False):
        url = f"https://generativelanguage.googleapis.com/v1/models/{config.MODEL}:{'streamGenerateContent?alt=sse' if stream else 'generateContent'}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "tools": [{"functionDeclarations": TOOL_DEFS_DICT}],
            "generationConfig": {"maxOutputTokens": config.MAX_TOKENS},
        }
        resp = requests.post(url, headers=headers, json=body, timeout=120, stream=stream)
        if resp.status_code != 200:
            raise Exception(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        return resp

    def process_with_token(self, user_message: str, token: str):
        """Process message using user's Google OAuth token. Yields dict events."""
        messages: list = []

        def add_content(role: str, parts: list):
            messages.append({"role": role, "parts": parts})

        add_content("user", [{"text": user_message}])

        while True:
            try:
                resp = self._call_gemini_rest(token, messages)
                data = resp.json()
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            candidates = data.get("candidates", [])
            if not candidates:
                yield {"type": "error", "content": "No response from model"}
                return

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                break

            add_content(content.get("role", "model"), parts)

            function_calls = [p for p in parts if "functionCall" in p]
            if not function_calls:
                text = "".join(p.get("text", "") for p in parts)
                yield {"type": "text", "content": text}
                yield {"type": "done"}
                return

            result_parts = []
            for p in function_calls:
                fc = p["functionCall"]
                name = fc["name"]
                args = fc.get("args", {})
                result = self.execute_tool(name, args)
                result_parts.append({
                    "functionResponse": {
                        "name": name,
                        "response": {"result": result},
                    }
                })
                yield {"type": "tool", "name": name, "content": result[:200]}

            add_content("user", result_parts)

    # === Big Pickle (OpenCode Zen) methods ===

    ZEN_BASE_URL = "https://opencode.ai/zen/v1"

    TOOL_DEFS_OPENAI = [
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

    def _bp_messages_from_list(self, msgs: list) -> list:
        """Convert internal message list to OpenAI-format messages."""
        result = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in msgs:
            role = m.get("role", "user")
            parts = m.get("parts", [])
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

            if tool_calls:
                result.append({"role": role, "content": content or None, "tool_calls": tool_calls})
            elif tool_call_id:
                result.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
            else:
                result.append({"role": role, "content": content})
        return result

    def process_with_bigpickle(self, user_message: str, stream: bool = False):
        """Process message using Big Pickle via OpenCode Zen API. Yields dict events."""
        client = OpenAI(
            base_url=self.ZEN_BASE_URL,
            api_key=config.OPENCODE_API_KEY,
        )
        messages: list = []

        def add_msg(role: str, parts: list):
            messages.append({"role": role, "parts": parts})

        add_msg("user", [{"text": user_message}])

        while True:
            try:
                bp_messages = self._bp_messages_from_list(messages)
                kwargs = dict(
                    model="big-pickle",
                    messages=bp_messages,
                    tools=self.TOOL_DEFS_OPENAI,
                    max_tokens=config.MAX_TOKENS,
                    stream=stream,
                )

                if stream:
                    collected_text = ""
                    tool_calls_accum = {}
                    finish_reason = None

                    for chunk in client.chat.completions.create(**kwargs):
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if not delta:
                            continue

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
                        sorted_calls = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
                        parts = []
                        for tc in sorted_calls:
                            name = tc["function"]["name"]
                            try:
                                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            parts.append({
                                "functionCall": {"name": name, "args": args}
                            })
                        add_msg("assistant", parts)

                        result_parts = []
                        for tc in sorted_calls:
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
                    response = client.chat.completions.create(**kwargs)
                    choice = response.choices[0] if response.choices else None
                    if not choice:
                        yield {"type": "error", "content": "No response from model"}
                        return

                    if choice.message.content:
                        add_msg("assistant", [{"text": choice.message.content}])
                        yield {"type": "text", "content": choice.message.content}
                        yield {"type": "done"}
                        return

                    if choice.message.tool_calls:
                        parts = []
                        for tc in choice.message.tool_calls:
                            try:
                                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            except json.JSONDecodeError:
                                args = {}
                            parts.append({
                                "functionCall": {"name": tc.function.name, "args": args}
                            })
                        add_msg("assistant", parts)

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

    def process_with_api_key(self, user_message: str, api_key: str, stream: bool = False):
        """Process message using user's own Gemini API key. Yields dict events."""
        from google import genai as genai2
        client = genai2.Client(api_key=api_key)
        tool_config = types.Tool(function_declarations=TOOL_DEFS)

        msgs: list[types.Content] = []
        msgs.append(types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        ))

        while True:
            try:
                if stream:
                    collected_parts = []
                    for chunk in client.models.generate_content_stream(
                        model=config.MODEL,
                        contents=msgs,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=[tool_config],
                            max_output_tokens=config.MAX_TOKENS,
                        ),
                    ):
                        if not chunk.candidates:
                            continue
                        for p in chunk.candidates[0].content.parts:
                            if p.text:
                                yield {"type": "text", "content": p.text}
                            collected_parts.append(p)
                    response_content = types.Content(role="model", parts=collected_parts) if collected_parts else None
                else:
                    response = client.models.generate_content(
                        model=config.MODEL,
                        contents=msgs,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=[tool_config],
                            max_output_tokens=config.MAX_TOKENS,
                        ),
                    )
                    if not response.candidates:
                        yield {"type": "error", "content": "Response blocked by safety filters"}
                        return
                    response_content = response.candidates[0].content
            except Exception as e:
                error = f"Gemini API error: {e}"
                if stream:
                    yield {"type": "error", "content": error}
                else:
                    yield {"type": "error", "content": error}
                return

            if not response_content:
                break

            msgs.append(response_content)
            function_calls = [p for p in response_content.parts if p.function_call]

            if not function_calls:
                break

            result_parts = []
            for part in function_calls:
                fc = part.function_call
                handler = TOOL_MAP.get(fc.name)
                if not handler:
                    result = f"Error: unknown tool '{fc.name}'"
                else:
                    try:
                        args = {k: v for k, v in fc.args.items()}
                        result = handler(**args)
                    except Exception as e:
                        result = f"Error executing {fc.name}: {e}"
                result_parts.append(types.Part.from_function_response(
                    name=fc.name,
                    response={"result": str(result)},
                ))
                if stream:
                    yield {"type": "tool", "name": fc.name, "content": str(result)[:200]}

            msgs.append(types.Content(role="user", parts=result_parts))

        if stream:
            yield {"type": "done"}
        else:
            text = response.text if response.text else ""
            text_parts = [p.text for p in (response_content.parts if response_content else []) if p.text]
            yield {"type": "text", "content": text or "\n".join(text_parts) or "(no response)"}
            yield {"type": "done"}
