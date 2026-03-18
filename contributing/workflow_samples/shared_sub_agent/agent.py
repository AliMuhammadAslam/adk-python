# Copyright 2026 Google LLC
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

"""Shared sub-agent sample: same LlmAgent in multiple tree positions."""

from google.adk.agents.llm_agent import LlmAgent
from google.genai import types

# Shared sub-agent: same instance reused in travel_agent and
# shopping_agent. Disallow transfers so it always calls the tool
# and reports results back to its parent.
search_agent = LlmAgent(
    name='search_agent',
    description='Searches the web and returns raw results.',
    mode='single_turn',
    instruction="""\
You emulate a web search agent.
If the user asks about travel, return:
    1. NYC to London round-trip from $450 (Delta, non-stop)
    2.  NYC to London round-trip from $520 (British Airways, non-stop)
If the user asks about shopping, return:
    1. ThinkPad X1 Carbon - $949, 14" display, 16GB RAM
    2. MacBook Air M3 - $999, 13.6" display, 16GB RAM
""",
)

travel_agent = LlmAgent(
    name='travel_agent',
    description='Handles travel-related questions (flights, hotels, trips).',
    instruction="""\
You help with travel questions. Delegate to search_agent to find
flights, hotels, or travel information, then summarize the results.
""",
    sub_agents=[search_agent],
)

shopping_agent = LlmAgent(
    name='shopping_agent',
    description='Handles shopping and product questions.',
    instruction="""\
You help with shopping questions. Delegate to search_agent to find
products and prices, then summarize the best options.
""",
    sub_agents=[search_agent],
)

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='Routes questions to specialized agents.',
    instruction="""\
You route questions to the appropriate agent:
- Travel questions (flights, hotels, trips): delegate to travel_agent
- Shopping questions (products, prices): delegate to shopping_agent
Only delegate when the question clearly matches one of the above categories.
For general questions (e.g. greetings, what you can do), answer directly.
""",
    sub_agents=[travel_agent, shopping_agent],
)
