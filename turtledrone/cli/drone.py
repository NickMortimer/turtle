"""Drone data management CLI commands."""

# Import and re-export the tdrone Typer app
# This allows 'drone' command to use all existing tdrone functionality
from turtledrone.tdrone import tdrone

# Create alias for the app
app = tdrone

if __name__ == "__main__":
    app()
