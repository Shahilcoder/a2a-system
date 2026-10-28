"""JSON-RPC errors: the standard codes plus the A2A ones this project uses."""

from __future__ import annotations
from typing import Any


class JsonRpcError(Exception):
    code = -32603
    message = "Internal error"

    def __init__(self, message: str | None = None, data: Any = None):
        self.message = message or self.message
        self.data = data
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        error = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error

    @classmethod
    def from_dict(cls, error: dict[str, Any]) -> JsonRpcError:
        """Rebuild the matching subclass from an error object received over the wire."""
        by_code = {sub.code: sub for sub in JsonRpcError.__subclasses__()}
        return by_code.get(error.get("code"), JsonRpcError)(error.get("message"), error.get("data"))


class ParseError(JsonRpcError):
    code, message = -32700, "Parse error"


class InvalidRequestError(JsonRpcError):
    code, message = -32600, "Invalid request"


class MethodNotFoundError(JsonRpcError):
    code, message = -32601, "Method not found"


class InvalidParamsError(JsonRpcError):
    code, message = -32602, "Invalid params"


class InternalError(JsonRpcError):
    code, message = -32603, "Internal error"


class TaskNotFoundError(JsonRpcError):
    code, message = -32001, "Task not found"


class TaskNotCancelableError(JsonRpcError):
    code, message = -32002, "Task cannot be canceled"


class UnsupportedOperationError(JsonRpcError):
    code, message = -32004, "This operation is not supported"


class InvalidAgentResponseError(JsonRpcError):
    code, message = -32006, "Invalid agent response"


class VersionNotSupportedError(JsonRpcError):
    code, message = -32009, "A2A version not supported (send header A2A-Version: 1.0)"
