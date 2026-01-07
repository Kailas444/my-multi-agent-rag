import re
import requests

def weather_tool_func(location: str) -> str:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        r = requests.get(url, params={"name": location, "count": 1}, timeout=5).json()
        if not r.get("results"): return f"Location '{location}' not found."
        
        lat, lon = r["results"][0]["latitude"], r["results"][0]["longitude"]
        w_url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "current_weather": True}
        w_res = requests.get(w_url, params=params).json()
        temp = w_res['current_weather']['temperature']
        return f"Current temperature in {location}: {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {e}"

def calculator_tool_func(expression: str) -> str:
    try:
        if not re.match(r"^[0-9+\-*/().\s]+$", expression):
            return "Invalid characters in expression."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"
