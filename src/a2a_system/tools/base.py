"""A Tool is anything the LLM can call: a name, a description and a JSON-schema for its arguments."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object

    @abstractmethod
    async def call(self, **arguments: Any) -> str: ...

    def schema(self) -> dict[str, Any]:
        """The OpenAI function-calling definition of this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
