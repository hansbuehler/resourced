"""resourced: cooperative GPU allocation and advisory resource limits."""

__version__ = "0.1.0"

from .client import Client, ResourcedError  # noqa: E402

__all__ = ["Client", "ResourcedError", "__version__"]
