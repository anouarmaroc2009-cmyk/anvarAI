from typing import List
import anthropic
import config
import tools

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if they don't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "workdir": {"type": "string", "description": "Working directory for the command (optional)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Use this for documentation, APIs, or recent events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "git_operation",
        "description": "Run a git command (status, diff, add, commit, push, log, branch, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "args": {"type": "string", "description": "Git arguments, e.g. 'status', 'log --oneline -5', 'diff'"},
                "workdir": {"type": "string", "description": "Working directory (optional)"},
            },
            "required": ["args"],
        },
    },
]

TOOL_MAP = {
    "read_file": tools.read_file,
    "write_file": tools.write_file,
    "run_command": tools.run_command,
    "web_search": tools.web_search,
    "git_operation": tools.git_operation,
}

VERCEL_WARNING = (
    "NOTE: You are running on Vercel (serverless). The write_file, run_command, and git_operation tools "
    "are NOT available. Only read_file and web_search work. For code generation, just output the code "
    "in your response and tell the user to save it locally."
) if config.IS_VERCEL else ""

SYSTEM_PROMPT = f"""You are an AI coding agent that helps users with software development.

You have access to tools for:
- Reading files from the filesystem
- Writing files (create/edit code)
- Running shell commands
- Searching the web
- Running git operations

Think step by step. Use tools as needed to accomplish the user's goals.
When writing code, ensure correctness and follow best practices.
When running commands, check output and handle errors appropriately.
{VERCEL_WARNING}"""


class CodingAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.messages: List[dict] = []

    def process_message(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOL_DEFINITIONS,
            )

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = TOOL_MAP.get(block.name)
                        if handler:
                            result = handler(**block.input)
                        else:
                            result = f"Unknown tool: {block.name}"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                if tool_results:
                    self.messages.append({"role": "user", "content": tool_results})
            else:
                break

        text_parts: List[str] = []
        for block in self.messages[-1]["content"]:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)
