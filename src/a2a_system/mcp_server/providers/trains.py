"""Indian Railways trains between two stations, via the RapidAPI "IRCTC" API (irctc1).

Endpoint path, params and response fields are from memory: check them in the RapidAPI dashboard.
"""

import httpx2

HOST = "irctc1.p.rapidapi.com"
URL = f"https://{HOST}/api/v3/trainBetweenStations"


class TrainProvider:
    def __init__(self, http: httpx2.AsyncClient, rapidapi_key: str = ""):
        self.http, self.rapidapi_key = http, rapidapi_key

    async def between_stations(self, from_station: str, to_station: str, date: str) -> str:
        if not self.rapidapi_key:
            return "Train search is not configured (RAPIDAPI_KEY is missing)."
        response = await self.http.get(
            URL,
            params={"fromStationCode": from_station.upper(), "toStationCode": to_station.upper(),
                    "dateOfJourney": date},
            headers={"X-RapidAPI-Key": self.rapidapi_key, "X-RapidAPI-Host": HOST},
        )
        response.raise_for_status()
        body = response.json()
        trains = body.get("data") or []
        if not body.get("status") or not trains:
            return f"No trains found from {from_station} to {to_station} on {date}. {body.get('message', '')}".strip()
        return "\n".join(
            f"- {t.get('train_number')} {t.get('train_name')}: departs {t.get('from_std')}, "
            f"arrives {t.get('to_sta')}, duration {t.get('duration')}"
            for t in trains[:15]
        )
