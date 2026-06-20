import subprocess
from pathlib import Path


def read_file(path: str) -> str:
    path = Path(path).resolve()
    if not path.exists():
        return f"Error: file not found: {path}"
    return path.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


def run_command(command: str, workdir: str = "") -> str:
    cwd = Path(workdir).resolve() if workdir else None
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    output = result.stdout or ""
    if result.stderr:
        output += "\n" + result.stderr if output else result.stderr
    return output if output else "(no output)"


def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"


def git_operation(args: str, workdir: str = "") -> str:
    cwd = Path(workdir).resolve() if workdir else None
    cmd = f"git {args}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    output = result.stdout or ""
    if result.stderr:
        output += "\n" + result.stderr if output else result.stderr
    return output if output else "(no output)"
