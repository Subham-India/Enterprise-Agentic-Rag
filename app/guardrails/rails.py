from nemoguardrails import LLMRails, RailsConfig

from app.gateway.client import get_llm
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT
from app.config import settings
_rails: LLMRails | None = None


def initialize_rails() -> None:
    """Build the NeMo rails once (the first build is slow)."""
    global _rails
    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)
    _rails = LLMRails(config, llm=get_llm("guardrails", model=settings.GUARD_MODEL))

def guard(message: str) -> tuple[bool, str | None]:
    """Run a message through the gate.

    Returns (True, canned_reply) if a rail fired: reply with it and skip the agent.
    Returns (False, None) if the message is fine: continue to the agent.
    """
    if _rails is None:
        initialize_rails()

    result = _rails.generate(messages=[{"role": "user", "content": message}])
    content = result.get("content", "") if isinstance(result, dict) else str(result)

    if any(indicator in content for indicator in RAIL_INDICATORS):
        return True, content
    return False, None