import os

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
MODEL = os.environ.get("BIG_PICKLE_MODEL", "big-pickle")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
