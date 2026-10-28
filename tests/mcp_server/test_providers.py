import json

import httpx2

from a2a_system.mcp_server.providers import search as search_module
from a2a_system.mcp_server.providers.flights import FlightProvider
from a2a_system.mcp_server.providers.news import NewsProvider
from a2a_system.mcp_server.providers.search import WebSearchProvider
from a2a_system.mcp_server.providers.trains import TrainProvider
from a2a_system.mcp_server.providers.weather import WeatherProvider, describe


def mock_http(routes: dict, seen: list | None = None) -> httpx2.AsyncClient:
    """routes maps a URL host+path to a Response (or a function of the request)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        route = routes[request.url.host + request.url.path]
        return route(request) if callable(route) else route

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


GEOCODE = {"results": [{"name": "Delhi", "country": "India", "latitude": 28.6, "longitude": 77.2}]}
FORECAST = {
    "current": {"temperature_2m": 31.5, "weather_code": 1, "wind_speed_10m": 9.4},
    "daily": {"time": ["2026-10-20", "2026-10-21"], "weather_code": [0, 63],
              "temperature_2m_max": [33.0, 30.1], "temperature_2m_min": [20.0, 21.2],
              "precipitation_probability_max": [0, 70]},
}


async def test_weather_forecast():
    seen = []
    http = mock_http({"geocoding-api.open-meteo.com/v1/search": httpx2.Response(200, json=GEOCODE),
                      "api.open-meteo.com/v1/forecast": httpx2.Response(200, json=FORECAST)}, seen)
    text = await WeatherProvider(http).forecast("Delhi", days=2)
    assert "Delhi, India: now 31.5°C, partly cloudy, wind 9.4 km/h" in text
    assert "2026-10-21: 21.2–30.1°C, rain, rain chance 70%" in text
    assert seen[1].url.params["latitude"] == "28.6"
    assert seen[1].url.params["forecast_days"] == "2"


async def test_weather_unknown_city():
    http = mock_http({"geocoding-api.open-meteo.com/v1/search": httpx2.Response(200, json={})})
    assert "No place" in await WeatherProvider(http).forecast("Atlantis")


def test_wmo_codes():
    assert describe(0) == "clear sky"
    assert describe(95) == "thunderstorm"
    assert describe(500) == "unknown"


RSS = """<?xml version="1.0"?><rss><channel><title>News</title>
<item><title>Monsoon ends</title><link>https://n.example/1</link><pubDate>Mon, 20 Oct 2026</pubDate></item>
<item><title>Markets up</title><link>https://n.example/2</link><pubDate>Mon, 20 Oct 2026</pubDate></item>
</channel></rss>"""


async def test_news_search():
    seen = []
    http = mock_http({"news.google.com/rss/search": httpx2.Response(200, text=RSS)}, seen)
    text = await NewsProvider(http).headlines("monsoon", limit=1)
    assert "Monsoon ends" in text and "https://n.example/1" in text
    assert "Markets up" not in text
    assert seen[0].url.params["q"] == "monsoon"


async def test_news_top_stories():
    http = mock_http({"news.google.com/rss": httpx2.Response(200, text=RSS)})
    assert "Markets up" in await NewsProvider(http).headlines()


async def test_search_uses_tavily_when_key_set():
    seen = []
    body = {"results": [{"title": "Python", "url": "https://python.org", "content": "The language"}]}
    http = mock_http({"api.tavily.com/search": httpx2.Response(200, json=body)}, seen)
    provider = WebSearchProvider(http, tavily_api_key="tvly-key")
    assert provider.backend == "tavily"
    text = await provider.search("python")
    assert "https://python.org" in text
    assert seen[0].headers["Authorization"] == "Bearer tvly-key"
    assert json.loads(seen[0].content)["query"] == "python"


async def test_search_falls_back_to_duckduckgo(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results):
            return [{"title": f"About {query}", "href": "https://ddg.example", "body": "snippet"}][:max_results]

    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    provider = WebSearchProvider(mock_http({}))
    assert provider.backend == "duckduckgo"
    assert "About cats" in await provider.search("cats")


TRAINS = {"status": True, "data": [{"train_number": "12951", "train_name": "MUMBAI RAJDHANI",
                                    "from_std": "16:55", "to_sta": "08:35", "duration": "15:40"}]}


async def test_trains_between_stations():
    seen = []
    http = mock_http({"irctc1.p.rapidapi.com/api/v3/trainBetweenStations": httpx2.Response(200, json=TRAINS)}, seen)
    text = await TrainProvider(http, rapidapi_key="k").between_stations("ndls", "bct", "2026-11-02")
    assert "12951 MUMBAI RAJDHANI" in text
    assert seen[0].headers["X-RapidAPI-Key"] == "k"
    assert seen[0].url.params["fromStationCode"] == "NDLS"
    assert seen[0].url.params["dateOfJourney"] == "2026-11-02"


async def test_trains_need_a_key():
    assert "not configured" in await TrainProvider(mock_http({})).between_stations("NDLS", "BCT", "2026-11-02")


FLIGHTS = {"data": [{"flight_status": "scheduled", "flight": {"iata": "AI101"}, "airline": {"name": "Air India"},
                     "departure": {"iata": "DEL", "scheduled": "2026-10-20T10:00:00+00:00"},
                     "arrival": {"iata": "BOM", "scheduled": "2026-10-20T12:10:00+00:00"}}]}


async def test_flights_search():
    seen = []
    http = mock_http({"api.aviationstack.com/v1/flights": httpx2.Response(200, json=FLIGHTS)}, seen)
    text = await FlightProvider(http, api_key="k").search(departure_iata="del", arrival_iata="bom")
    assert "AI101 (Air India) scheduled: DEL" in text
    assert seen[0].url.scheme == "http"  # AviationStack's free tier is HTTP-only
    assert seen[0].url.params["dep_iata"] == "DEL"
    assert "flight_iata" not in seen[0].url.params


async def test_flights_api_error():
    body = {"error": {"code": "invalid_access_key", "message": "Bad key"}}
    http = mock_http({"api.aviationstack.com/v1/flights": httpx2.Response(401, json=body)})
    assert "Bad key" in await FlightProvider(http, api_key="x").search(flight_number="AI101")


async def test_flights_need_a_key():
    assert "not configured" in await FlightProvider(mock_http({})).search("DEL", "BOM")
