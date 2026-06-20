import os

IS_VERCEL = os.environ.get("VERCEL", "") == "1"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
# Use smaller max_tokens on Vercel to stay within 60s timeout
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096" if IS_VERCEL else "8192"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
