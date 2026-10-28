"""Headlines from Google News RSS (free, no API key)."""

import xml.etree.ElementTree as ET

import httpx2

TOP_URL = "https://news.google.com/rss"
SEARCH_URL = "https://news.google.com/rss/search"
LOCALE = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}


class NewsProvider:
    def __init__(self, http: httpx2.AsyncClient):
        self.http = http

    async def headlines(self, topic: str = "", limit: int = 5) -> str:
        if topic:
            response = await self.http.get(SEARCH_URL, params={"q": topic, **LOCALE})
        else:
            response = await self.http.get(TOP_URL, params=LOCALE)
        response.raise_for_status()
        items = ET.fromstring(response.text).findall("./channel/item")[:limit]
        if not items:
            return f"No news found for {topic!r}."
        return "\n".join(f"- {item.findtext('title')} ({item.findtext('pubDate')})\n  {item.findtext('link')}"
                         for item in items)
