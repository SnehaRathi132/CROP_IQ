import os

# Prefer environment variable, fall back to bundled key for local demos.
weather_api_key = os.getenv("OPENWEATHER_API_KEY") or None
