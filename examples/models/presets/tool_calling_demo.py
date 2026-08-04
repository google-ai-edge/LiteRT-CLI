# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
