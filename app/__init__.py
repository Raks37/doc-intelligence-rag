"""Load environment variables from a local .env file as early as possible.

Every entry point (the FastAPI app, the test suite, helper scripts) imports
something from `app`, so putting `load_dotenv()` here guarantees the LLM
provider settings in `.env` are available before `app.llm` reads them.
"""
from dotenv import load_dotenv

load_dotenv()
