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
import os
from pathlib import Path
from typing import AsyncGenerator
from typing import Optional

from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events.event import Event as AdkEvent
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

# Read target folder from environment
TARGET_FOLDER = os.environ.get("ADK_TEST_FOLDER")


def get_test_files():
  """Yields (agent_dir, test_file_path) recursively."""
  if not TARGET_FOLDER:
    return
  target_dir = Path(TARGET_FOLDER)
  if not target_dir.exists():
    return

  for test_file in target_dir.rglob("tests/*.json"):
    agent_dir = test_file.parent.parent
    # Verify it looks like an agent directory
    if (
        (agent_dir / "agent.py").exists()
        or (agent_dir / "__init__.py").exists()
        or (agent_dir / "root_agent.yaml").exists()
    ):
      yield agent_dir, test_file


class MockModel(BaseLlm):
  model: str = "mock"
  requests: list[LlmRequest] = []
  responses: list[LlmResponse] = []
  response_index: int = -1

  @classmethod
  def create(cls, responses: list[str]):
    llm_responses = [
        LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=item)]
            )
        )
        for item in responses
    ]
    return cls(responses=llm_responses)

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["mock"]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.response_index += 1
    self.requests.append(llm_request)
    yield self.responses[self.response_index]


class InMemoryRunner:

  def __init__(self, root_agent=None, app=None):
    if app:
      self.app_name = app.name
      self.runner = Runner(
          app=app,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
      )
    else:
      self.app_name = "test_app"
      self.runner = Runner(
          app_name="test_app",
          agent=root_agent,
          artifact_service=InMemoryArtifactService(),
          session_service=InMemorySessionService(),
          memory_service=InMemoryMemoryService(),
      )
    self.session_id = None

  @property
  def session(self):
    if not self.session_id:
      session = self.runner.session_service.create_session_sync(
          app_name=self.app_name, user_id="test_user"
      )
      self.session_id = session.id
      return session
    return self.runner.session_service.get_session_sync(
        app_name=self.app_name, user_id="test_user", session_id=self.session_id
    )

  def run(self, new_message) -> list[AdkEvent]:
    content = (
        new_message
        if isinstance(new_message, types.Content)
        else types.Content(
            role="user", parts=[types.Part.from_text(text=new_message)]
        )
    )
    return list(
        self.runner.run(
            user_id=self.session.user_id,
            session_id=self.session.id,
            new_message=content,
        )
    )


def normalize_events(events, is_json=False):
  normalized = []
  for e in events:
    if is_json:
      try:
        e_obj = AdkEvent.model_validate(e)
        d = e_obj.model_dump(
            mode="json",
            by_alias=True,
            exclude={
                "id",
                "timestamp",
                "invocation_id",
                "model_version",
                "finish_reason",
                "usage_metadata",
            },
            exclude_none=True,
        )
      except Exception:
        d = dict(e)
        d.pop("id", None)
        d.pop("timestamp", None)
        d.pop("invocationId", None)
    else:
      try:
        e_obj = AdkEvent.model_validate(e.model_dump())
        d = e_obj.model_dump(
            mode="json",
            by_alias=True,
            exclude={
                "id",
                "timestamp",
                "invocation_id",
                "model_version",
                "finish_reason",
                "usage_metadata",
            },
            exclude_none=True,
        )
      except Exception:
        d = e.model_dump(
            mode="json",
            by_alias=True,
            exclude={
                "id",
                "timestamp",
                "invocation_id",
                "model_version",
                "finish_reason",
                "usage_metadata",
            },
            exclude_none=True,
        )

    actions = d.get("actions", {})
    state_delta = actions.get("stateDelta", {}) if actions else {}
    if state_delta:
      keys_to_remove = [k for k in state_delta if k.endswith("_join_state")]
      for k in keys_to_remove:
        del state_delta[k]

    normalized.append(d)
  return normalized


def make_sort_key(d):
  node_path = d.get("nodeInfo", {}).get("path", "")
  author = d.get("author", "")
  return (author, node_path, json.dumps(d, sort_keys=True))


@pytest.mark.parametrize(
    "agent_dir, test_file",
    list(get_test_files()),
    ids=lambda val: val.name if isinstance(val, Path) else val,
)
def test_agent_replay(agent_dir, test_file, monkeypatch):
  # Add agent_dir.parent to sys.path so relative imports work
  import sys

  sys_path_saved = list(sys.path)
  sys.path.insert(0, str(agent_dir.parent))

  try:
    loader = AgentLoader(str(agent_dir.parent))
    agent_or_app = loader.load_agent(agent_dir.name)

    with open(test_file, "r") as f:
      session_data = json.load(f)

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

    expected_events = events_data[1:]

    mock_responses = []
    for ev in expected_events:
      if "modelVersion" in ev and "content" in ev:
        content = ev["content"]
        if content.get("role") == "model":
          parts = content.get("parts", [])
          if parts and "text" in parts[0]:
            mock_responses.append(parts[0]["text"])

    if mock_responses:
      mock_model = MockModel.create(responses=mock_responses)

      async def mock_gen_async(instance, llm_request, stream=False):
        async for resp in mock_model.generate_content_async(
            llm_request, stream
        ):
          yield resp

      from google.adk.models.base_llm import BaseLlm
      from google.adk.models.google_llm import Gemini

      monkeypatch.setattr(BaseLlm, "generate_content_async", mock_gen_async)
      monkeypatch.setattr(Gemini, "generate_content_async", mock_gen_async)

    runner = (
        InMemoryRunner(app=agent_or_app)
        if isinstance(agent_or_app, App)
        else InMemoryRunner(root_agent=agent_or_app)
    )

    actual_events = []
    first_run_events = runner.run(user_message)
    actual_events.extend(first_run_events)

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
            actual_events.append(
                AdkEvent(
                    author="user",
                    content=types.Content(role="user", parts=real_parts),
                )
            )
            next_run_events = runner.run(
                types.Content(role="user", parts=real_parts)
            )
            actual_events.extend(next_run_events)

    actual_events = [
        e for e in actual_events if not getattr(e, "partial", False)
    ]

    actual_dicts = normalize_events(actual_events, is_json=False)
    expected_dicts = normalize_events(expected_events, is_json=True)

    actual_dicts.sort(key=make_sort_key)
    expected_dicts.sort(key=make_sort_key)

    assert actual_dicts == expected_dicts
  finally:
    sys.path = sys_path_saved
