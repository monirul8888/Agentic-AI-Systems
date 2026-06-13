import requests
import uvicorn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=10)
    data = response.json()
    current = data["current_condition"][0]
    return {
        "city": city,
        "temperature": current["temp_C"],
        "feels_like": current["FeelsLikeC"],
        "weather": current["weatherDesc"][0]["value"],
        "humidity": current["humidity"],
        "wind_speed": current["windspeedKmph"]
    }


if __name__ == "__main__":
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)