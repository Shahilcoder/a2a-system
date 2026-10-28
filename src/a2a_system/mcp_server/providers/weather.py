"""Weather from Open-Meteo (free, no API key)."""

import httpx2

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes, grouped: (highest code in the group, description)
WMO_GROUPS = [(0, "clear sky"), (3, "partly cloudy"), (48, "fog"), (57, "drizzle"), (67, "rain"),
              (77, "snow"), (82, "rain showers"), (86, "snow showers"), (99, "thunderstorm")]


def describe(code: int) -> str:
    return next((text for limit, text in WMO_GROUPS if code <= limit), "unknown")


class WeatherProvider:
    def __init__(self, http: httpx2.AsyncClient):
        self.http = http

    async def forecast(self, city: str, days: int = 3) -> str:
        response = await self.http.get(GEOCODE_URL, params={"name": city, "count": 1})
        response.raise_for_status()
        places = response.json().get("results")
        if not places:
            return f"No place called {city!r} was found."
        place = places[0]

        response = await self.http.get(FORECAST_URL, params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 7)),
        })
        response.raise_for_status()
        data = response.json()
        now, daily = data["current"], data["daily"]
        current = (f"{place['name']}, {place.get('country', '')}: now {now['temperature_2m']}°C, "
                   f"{describe(now['weather_code'])}, wind {now['wind_speed_10m']} km/h")
        lines = [current]
        for day, code, high, low, rain in zip(daily["time"], daily["weather_code"], daily["temperature_2m_max"],
                                              daily["temperature_2m_min"], daily["precipitation_probability_max"]):
            lines.append(f"{day}: {low}–{high}°C, {describe(code)}, rain chance {rain}%")
        return "\n".join(lines)
