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

from __future__ import annotations

import json
from pathlib import Path

from google.adk.apps.app import App
from google.adk.cli.agent_test_runner import make_sort_key
from google.adk.cli.agent_test_runner import normalize_events
from google.adk.cli.utils.agent_loader import AgentLoader
from google.genai import types
import pytest

from . import testing_utils

SAMPLES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "contributing"
    / "new_workflow_samples"
)


def get_test_files():
  """Yields (sample_name, test_file_path)."""
  if not SAMPLES_DIR.exists():
    return
  for sample_dir in SAMPLES_DIR.iterdir():
    if sample_dir.is_dir():
      tests_dir = sample_dir / "tests"
      if tests_dir.exists() and tests_dir.is_dir():
        for test_file in tests_dir.glob("*.json"):
          yield sample_dir.name, test_file


@pytest.mark.parametrize(
    "sample_name, test_file",
    list(get_test_files()),
    ids=lambda val: val.name if isinstance(val, Path) else val,
)
def test_sample(sample_name, test_file, monkeypatch):
  """Tests a sample by replaying exported session events."""
  # Load agent
  loader = AgentLoader(str(SAMPLES_DIR))
  agent_or_app = loader.load_agent(sample_name)

  # Load session file
  with open(test_file, "r") as f:
    session_data = json.load(f)

  # Extract input
  events_data = session_data.get("events", [])
  if not events_data:
    pytest.skip(f"No events in {test_file}")

  first_event = events_data[0]
  user_message = ""
  if first_event.get("author") == "user":
    parts = first_event.get("content", {}).get("parts", [])
    if parts and "text" in parts[0]:
      user_message = parts[0]["text"]

  if not user_message:
    pytest.skip(f"Could not find user message in {test_file}")

  # Extract expected events (excluding the first user message)
  expected_events = events_data[1:]

  # Extract expected model responses for mocking
  mock_responses = []
  for ev in expected_events:
    if "modelVersion" in ev and "content" in ev:
      content = ev["content"]
      if content.get("role") == "model":
        parts = content.get("parts", [])
        if parts and "text" in parts[0]:
          mock_responses.append(parts[0]["text"])

  if mock_responses:
    from google.adk.models.base_llm import BaseLlm

    mock_model = testing_utils.MockModel.create(responses=mock_responses)

    async def mock_gen_async(instance, llm_request, stream=False):
      async for resp in mock_model.generate_content_async(llm_request, stream):
        yield resp

    from google.adk.models.google_llm import Gemini

    monkeypatch.setattr(BaseLlm, "generate_content_async", mock_gen_async)
    monkeypatch.setattr(Gemini, "generate_content_async", mock_gen_async)

  # Run agent
  from .testing_utils import InMemoryRunner

  runner = (
      InMemoryRunner(app=agent_or_app)
      if isinstance(agent_or_app, App)
      else InMemoryRunner(root_agent=agent_or_app)
  )

  actual_events = []

  # First turn
  first_run_events = runner.run(user_message)
  actual_events.extend(first_run_events)

  # Check for subsequent user messages in the trace and replay them
  from google.adk.events.event import Event as AdkEvent

  for event in events_data[1:]:
    if event.get("author") == "user":
      content_dict = event.get("content", {})
      if content_dict:
        parts = content_dict.get("parts", [])
        real_parts = []
        for p in parts:
          if "functionResponse" in p:
            fr = p["functionResponse"]
            real_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=fr.get("id"),
                        name=fr.get("name"),
                        response=fr.get("response"),
                    )
                )
            )
          elif "text" in p:
            real_parts.append(types.Part(text=p["text"]))
          elif "functionCall" in p:
            fc = p["functionCall"]
            real_parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=fc.get("id"),
                        name=fc.get("name"),
                        args=fc.get("args"),
                    )
                )
            )

        if real_parts:
          # Add the user event to actual_events to match the expected trace
          actual_events.append(
              AdkEvent(
                  author="user",
                  content=types.Content(role="user", parts=real_parts),
              )
          )

          # Run the runner with these parts
          next_run_events = runner.run(
              types.Content(role="user", parts=real_parts)
          )
          actual_events.extend(next_run_events)

  # Filter out partial events (e.g. streaming chunks) as they are typically not in session files
  actual_events = [e for e in actual_events if not getattr(e, "partial", False)]

  # Normalize both for comparison
  actual_dicts = normalize_events(actual_events, is_json=False)
  expected_dicts = normalize_events(expected_events, is_json=True)

  # Sort both for comparison
  actual_dicts.sort(key=make_sort_key)
  expected_dicts.sort(key=make_sort_key)

  # Compare
  assert actual_dicts == expected_dicts
