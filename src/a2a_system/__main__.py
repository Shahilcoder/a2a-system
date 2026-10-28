"""Entry point: `python -m a2a_system <sprout|information|coding|mcp-server|gateway>`."""

import asyncio
import logging
import sys

import httpx2
import uvicorn

from a2a_system.config import Settings
from a2a_system.gateway.app import build_gateway
from a2a_system.mcp_server.server import build_server
from a2a_system.agents.app import build_agent
from a2a_system.protocol.client import A2AClient

SERVICES = ("sprout", "information", "coding", "mcp-server", "gateway")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = sys.argv[1] if len(sys.argv) > 1 else ""
    if service not in SERVICES:
        sys.exit(f"usage: python -m a2a_system <{'|'.join(SERVICES)}>")
    settings = Settings()

    if service == "mcp-server":
        server = build_server(httpx2.AsyncClient(timeout=30, follow_redirects=True), settings)
        # Binding 0.0.0.0 (not localhost) also leaves DNS-rebinding protection off, so "mcp-server:8000" is accepted.
        asyncio.run(server.run_streamable_http_async(host="0.0.0.0", port=settings.port, stateless_http=True))
    elif service == "gateway":
        uvicorn.run(build_gateway(A2AClient(), settings.sprout_url), host="0.0.0.0", port=settings.port)
    else:
        uvicorn.run(build_agent(service, settings), host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
