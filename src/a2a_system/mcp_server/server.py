"""The MCP server: each tool is a thin wrapper around a provider."""

import httpx2
from mcp.server.mcpserver import MCPServer

from a2a_system.config import Settings
from a2a_system.mcp_server.providers.flights import FlightProvider
from a2a_system.mcp_server.providers.news import NewsProvider
from a2a_system.mcp_server.providers.search import WebSearchProvider
from a2a_system.mcp_server.providers.trains import TrainProvider
from a2a_system.mcp_server.providers.weather import WeatherProvider


def build_server(http: httpx2.AsyncClient, settings: Settings) -> MCPServer:
    weather = WeatherProvider(http)
    news = NewsProvider(http)
    search = WebSearchProvider(http, settings.tavily_api_key)
    trains = TrainProvider(http, settings.rapidapi_key)
    flights = FlightProvider(http, settings.aviationstack_api_key)

    server = MCPServer("information-tools", instructions="Live weather, news, web search, Indian trains, flights.")

    @server.tool()
    async def get_weather(city: str, days: int = 3) -> str:
        """Current weather and a daily forecast (1-7 days) for a city."""
        return await weather.forecast(city, days)

    @server.tool()
    async def get_news(topic: str = "", limit: int = 5) -> str:
        """Latest news headlines with links. Leave topic empty for top stories."""
        return await news.headlines(topic, limit)

    @server.tool()
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web. Returns titles, links and snippets."""
        return await search.search(query, max_results)

    @server.tool()
    async def search_trains(from_station: str, to_station: str, date: str) -> str:
        """Indian Railways trains between two station codes (e.g. NDLS, BCT) on a date (YYYY-MM-DD)."""
        return await trains.between_stations(from_station, to_station, date)

    @server.tool()
    async def search_flights(departure_iata: str = "", arrival_iata: str = "", flight_number: str = "") -> str:
        """Live and scheduled flights filtered by airport IATA codes (e.g. DEL, BOM) and/or flight number (e.g. AI101)."""
        return await flights.search(departure_iata, arrival_iata, flight_number)

    return server
