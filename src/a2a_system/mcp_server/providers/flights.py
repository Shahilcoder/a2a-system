"""Flights from AviationStack. The free tier only allows plain HTTP."""

import httpx2

URL = "http://api.aviationstack.com/v1/flights"


class FlightProvider:
    def __init__(self, http: httpx2.AsyncClient, api_key: str = ""):
        self.http, self.api_key = http, api_key

    async def search(self, departure_iata: str = "", arrival_iata: str = "", flight_number: str = "") -> str:
        if not self.api_key:
            return "Flight search is not configured (AVIATIONSTACK_API_KEY is missing)."
        filters = {"dep_iata": departure_iata.upper(), "arr_iata": arrival_iata.upper(),
                   "flight_iata": flight_number.upper().replace(" ", "")}
        params = {"access_key": self.api_key, "limit": 10, **{k: v for k, v in filters.items() if v}}
        response = await self.http.get(URL, params=params)
        body = response.json()
        if "error" in body:
            return f"Flight search failed: {body['error'].get('message', body['error'])}"
        flights = body.get("data") or []
        if not flights:
            return "No flights found."
        return "\n".join(
            f"- {f['flight'].get('iata')} ({f['airline'].get('name')}) {f.get('flight_status')}: "
            f"{f['departure'].get('iata')} {f['departure'].get('scheduled')} -> "
            f"{f['arrival'].get('iata')} {f['arrival'].get('scheduled')}"
            for f in flights
        )
