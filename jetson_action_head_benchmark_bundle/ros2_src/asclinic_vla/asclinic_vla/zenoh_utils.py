import json


def load_zenoh():
    """Import Zenoh lazily so non-Zenoh ROS nodes can still run without it installed."""
    try:
        import zenoh
    except ImportError as exc:
        raise RuntimeError(
            'The split VLA bridge requires the zenoh Python package. '
            'Install it on the Jetson with: pip install eclipse-zenoh'
        ) from exc
    return zenoh


def open_zenoh_session(connect_endpoints=None, listen_endpoints=None):
    """Create a Zenoh session from launch/config endpoint lists.

    connect_endpoints are remote routers/peers this process should dial.
    listen_endpoints are local addresses this process should expose for peers to dial.
    """
    zenoh = load_zenoh()
    config = zenoh.Config()

    connect_endpoints = list(connect_endpoints or [])
    listen_endpoints = list(listen_endpoints or [])

    if connect_endpoints:
        config.insert_json5('connect/endpoints', json.dumps(connect_endpoints))
    if listen_endpoints:
        config.insert_json5('listen/endpoints', json.dumps(listen_endpoints))

    return zenoh.open(config)
