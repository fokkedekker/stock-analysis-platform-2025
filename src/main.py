"""Main entry point for the Stock Analysis API."""

import logging
import sys

import uvicorn

from src.config import get_settings
from src.database.schema import create_all_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Run the API server."""
    settings = get_settings()

    # Initialize database
    logger.info("Initializing database...")
    create_all_tables()

    # Run server
    logger.info("Starting API server...")
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
