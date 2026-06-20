from typing import Any
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
