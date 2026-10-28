from a2a_system.protocol.errors import JsonRpcError, TaskNotFoundError, VersionNotSupportedError


def test_codes_and_wire_shape():
    assert TaskNotFoundError("no task x").to_dict() == {"code": -32001, "message": "no task x"}
    assert VersionNotSupportedError().code == -32009


def test_from_dict_rebuilds_subclass():
    error = JsonRpcError.from_dict({"code": -32001, "message": "gone"})
    assert isinstance(error, TaskNotFoundError)
    assert error.message == "gone"
    assert type(JsonRpcError.from_dict({"code": 1, "message": "?"})) is JsonRpcError
