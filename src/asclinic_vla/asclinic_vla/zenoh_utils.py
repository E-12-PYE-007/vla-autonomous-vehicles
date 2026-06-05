import json


def load_zenoh():
    try:
        import zenoh
        return zenoh
    except ImportError as exc:
        raise RuntimeError(
            'The split VLA bridge requires the zenoh Python package. '
            'Install it with: pip install eclipse-zenoh'
        ) from exc


def normalize_endpoints(endpoints):
    if endpoints is None:
        return []
    if isinstance(endpoints, str):
        return [endpoints] if endpoints else []
    return list(endpoints)


def open_zenoh_session(connect_endpoints=None, listen_endpoints=None):
    zenoh = load_zenoh()
    config = zenoh.Config()

    connect_endpoints = normalize_endpoints(connect_endpoints)
    listen_endpoints = normalize_endpoints(listen_endpoints)

    if connect_endpoints:
        config.insert_json5('connect/endpoints', json.dumps(connect_endpoints))

    if listen_endpoints:
        config.insert_json5('listen/endpoints', json.dumps(listen_endpoints))

    return zenoh.open(config)