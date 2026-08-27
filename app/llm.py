"""
Single place to configure which LLM backs the pipeline.

Default is `local`: a GGUF model (e.g. `phi-2.Q4_K_M.gguf`) run in-process via
llama-cpp-python, so the whole pipeline works offline with no API key. Structured
output is produced with grammar-constrained JSON decoding, which guarantees the
model returns something that parses against the requested Pydantic schema even
when the model itself is small.

Interview talking point: in a regulated financial-services environment you'd
almost certainly be pointed at Amazon Bedrock (data stays in your AWS VPC,
audit/compliance already sorted) rather than calling a public API directly.
Swapping providers here is a one-line change in `.env`.
"""
import json
import os
from functools import lru_cache

PROVIDER = os.getenv("LLM_PROVIDER", "local")  # local | anthropic | openai | bedrock
MODEL_NAME = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# --- local (llama-cpp-python) settings -------------------------------------
_DEFAULT_MODEL_PATH = r"C:\Users\raksh\AI\projects\phi-2.Q4_K_M.gguf"
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", _DEFAULT_MODEL_PATH)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "2048"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))


def _to_openai_messages(prompt_value) -> list[dict]:
    """Convert a LangChain ChatPromptValue (or plain string) to the
    role/content dicts llama-cpp-python expects."""
    if isinstance(prompt_value, str):
        return [{"role": "user", "content": prompt_value}]

    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in prompt_value.to_messages()
    ]


def _loads_lenient(content: str) -> dict:
    """Parse model output as JSON, tolerating stray markdown fences / prose."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _clamp_to_schema(data: dict, schema: dict) -> dict:
    """Small models happily emit numbers outside a field's documented range,
    which would fail Pydantic validation. Clamp top-level numeric fields to
    their schema minimum/maximum so a weak model can't break the pipeline."""
    props = schema.get("properties", {})
    for name, spec in props.items():
        if name not in data or not isinstance(data[name], (int, float)):
            continue
        if "minimum" in spec:
            data[name] = max(data[name], spec["minimum"])
        if "maximum" in spec:
            data[name] = min(data[name], spec["maximum"])
    return data


class _LocalChatModel:
    """Wrapper exposing the slice of the LangChain chat-model API this project
    uses: `.with_structured_output(PydanticModel)` returning a Runnable, so
    callers can keep composing `PROMPT | model` and calling `.invoke({...})`."""

    def __init__(self, llm):
        self._llm = llm

    def with_structured_output(self, schema_cls):
        from langchain_core.runnables import RunnableLambda

        schema = schema_cls.model_json_schema()

        def _run(prompt_value):
            result = self._llm.create_chat_completion(
                messages=_to_openai_messages(prompt_value),
                temperature=0.0,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object", "schema": schema},
            )
            content = result["choices"][0]["message"]["content"]
            data = _clamp_to_schema(_loads_lenient(content), schema)
            return schema_cls(**data)

        return RunnableLambda(_run)


@lru_cache
def get_chat_model():
    if PROVIDER == "local":
        from llama_cpp import Llama

        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"LLM_MODEL_PATH does not exist: {LLM_MODEL_PATH}. "
                "Set it in .env to point at your .gguf file."
            )
        llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=LLM_N_CTX, verbose=False)
        return _LocalChatModel(llm)

    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=MODEL_NAME, temperature=0)

    if PROVIDER == "openai":
        # Also works against a local OpenAI-compatible server (LM Studio,
        # llama-cpp server, Ollama) -- set OPENAI_BASE_URL and a dummy key.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    if PROVIDER == "bedrock":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            model_kwargs={"temperature": 0},
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
