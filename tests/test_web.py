from app.fusion import Hit
from app.llm import SYSTEM, SYSTEM_WEB, build_messages


def _hit(text):
    return Hit(1, "d", text)


def test_web_passages_follow_kb_passages_and_switch_prompt():
    web = [{"title": "Aspirin", "url": "https://x", "text": "Aspirin is an NSAID."}]
    msgs = build_messages("q", [_hit("kb text")], web)
    assert msgs[0]["content"] == SYSTEM_WEB
    assert "[1] kb text" in msgs[1]["content"]
    assert "[2] WEB (Aspirin): Aspirin is an NSAID." in msgs[1]["content"]


def test_without_web_the_prompt_is_unchanged():
    assert build_messages("q", [_hit("kb")])[0]["content"] == SYSTEM
