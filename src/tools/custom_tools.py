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

# --- NEW: Stable Wikipedia Search Tool (Replaces DuckDuckGo) ---
def wikipedia_search_tool(query: str) -> str:
    """Searches Wikipedia using MediaWiki API. No extra dependencies required."""
    try:
        # Wikipedia API endpoint
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "",
            "srlimit": 3
        }
        
        response = requests.get(url, params=params, timeout=5).json()
        search_results = response.get("query", {}).get("search", [])
        
        if not search_results:
            return f"Wikipedia found no results for '{query}'."
            
        # Extract snippets from top results
        results_text = []
        for item in search_results:
            title = item.get("title")
            snippet = item.get("snippet", "").replace("</span>", "").replace("<span class=\"searchmatch\">", "")
            results_text.append(f"• {title}: {snippet}")
            
        return "\n".join(results_text)
        
    except Exception as e:
        return f"Wikipedia search error: {e}"
