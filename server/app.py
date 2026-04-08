"""
FastAPI application for the Bugs Buddy Environment.

Exposes BugsBuddyEnvironment over HTTP and WebSocket endpoints.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import BugsBuddyAction, BugsBuddyObservation
    from .environment import BugsBuddyEnvironment
except ModuleNotFoundError:
    from models import BugsBuddyAction, BugsBuddyObservation
    from server.environment import BugsBuddyEnvironment


app = create_app(
    BugsBuddyEnvironment,
    BugsBuddyAction,
    BugsBuddyObservation,
    env_name="bugs_buddy",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
