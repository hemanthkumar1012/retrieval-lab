"""Live web search, used on top of the knowledge base when the caller asks for it.

Tavily is used when TAVILY_API_KEY is set (free tier: 1,000 searches a month).
Without a key it falls back to Wikipedia's search API, which needs no signup."""

import html
import re

import httpx

from .config import settings

TIMEOUT = httpx.Timeout(8.0, connect=4.0)
USER_AGENT = "retrieval-lab/1.0 (https://github.com/hemanthkumar1012/retrieval-lab)"
TAG = re.compile(r"<[^>]+>")


async def _tavily(client: httpx.AsyncClient, query: str, n: int) -> list[dict]:
    resp = await client.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json={"query": query, "max_results": n, "search_depth": "basic"},
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title") or r["url"], "url": r["url"], "text": (r.get("content") or "").strip()}
        for r in resp.json().get("results", [])
        if r.get("url") and r.get("content")
    ]


async def _wikipedia(client: httpx.AsyncClient, query: str, n: int) -> list[dict]:
    resp = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": n,
            "srprop": "snippet", "format": "json", "utf8": 1,
        },
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("query", {}).get("search", []):
        title = r["title"]
        # the search snippet is a fragment; the page summary reads better as a source
        summary = await client.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
        )
        text = summary.json().get("extract", "") if summary.status_code == 200 else ""
        text = text or html.unescape(TAG.sub("", r.get("snippet", "")))
        if text:
            out.append({"title": title, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                        "text": text})
    return out


async def web_search(query: str, n: int = 4) -> list[dict]:
    """Returns [{title, url, text}]. Raises httpx errors; the caller decides what a failure means."""
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        if settings.tavily_api_key:
            return await _tavily(client, query, n)
        return await _wikipedia(client, query, n)
