# Tool calling demo preset for LiteRT-LM CLI

system_instruction = (
    "You are a helpful assistant that uses tool calls when appropriate."
)


def get_current_weather(location: str) -> str:
  """Get the current weather for a given location.

  Args:
      location: The city and state, e.g. San Francisco, CA or Tokyo.
  """
  location_lower = location.lower()
  if "tokyo" in location_lower:
    return "The weather in Tokyo is sunny and 22°C."
  elif "paris" in location_lower:
    return "The weather in Paris is rainy and 15°C."
  elif "san francisco" in location_lower:
    return "The weather in San Francisco is foggy and 16°C."
  else:
    return f"The weather in {location} is clear and 20°C."
