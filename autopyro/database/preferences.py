"""Connection default merging and session-only credentials, independent of Qt."""

_session_credentials = None


def _signature(parameters):
    return tuple(str(parameters[key]) for key in ("host", "port", "dbname", "user"))


def connection_parameters(base, preferences):
    """Never reuse a password for another endpoint or user."""
    result = dict(base, host=preferences.database_host, port=str(preferences.database_port),
                  dbname=preferences.database_name, user=preferences.database_user)
    if _signature(result) != _signature(base):
        result["password"] = ""
    if _session_credentials is not None and _session_credentials[0] == _signature(result):
        result["password"] = _session_credentials[1]
    return result


def remember_password(preferences, password):
    global _session_credentials
    signature = (preferences.database_host, str(preferences.database_port), preferences.database_name, preferences.database_user)
    _session_credentials = (signature, password)
