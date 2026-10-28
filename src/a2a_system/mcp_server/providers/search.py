"""Web search: Tavily when TAVILY_API_KEY is set, otherwise DuckDuckGo (no key needed)."""

import asyncio

import httpx2
from ddgs import DDGS

TAVILY_URL = "https://api.tavily.com/search"


class WebSearchProvider:
    def __init__(self, http: httpx2.AsyncClient, tavily_api_key: str = ""):
        self.http, self.tavily_api_key = http, tavily_api_key

    @property
    def backend(self) -> str:
        return "tavily" if self.tavily_api_key else "duckduckgo"

    async def search(self, query: str, max_results: int = 5) -> str:
        if self.tavily_api_key:
            results = await self._tavily(query, max_results)
        else:
            results = await self._duckduckgo(query, max_results)
        if not results:
            return f"No results for {query!r}."
        return "\n".join(f"- {title}\n  {url}\n  {snippet}" for title, url, snippet in results)

    async def _tavily(self, query: str, max_results: int) -> list[tuple[str, str, str]]:
        response = await self.http.post(TAVILY_URL, json={"query": query, "max_results": max_results},
                                        headers={"Authorization": f"Bearer {self.tavily_api_key}"})
        response.raise_for_status()
        return [(r["title"], r["url"], r.get("content", "")) for r in response.json().get("results", [])]

    async def _duckduckgo(self, query: str, max_results: int) -> list[tuple[str, str, str]]:
        # ddgs is synchronous, so run it in a worker thread to keep the event loop free.
        results = await asyncio.to_thread(lambda: DDGS().text(query, max_results=max_results))
        return [(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in results]
