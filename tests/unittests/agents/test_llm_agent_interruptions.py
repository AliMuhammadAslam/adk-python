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

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
import pytest

from tests.unittests import testing_utils
from tests.unittests.agents.llm.event_utils import text_parts

_USER_ID = 'test_user'
_SESSION_ID = 'test_session'

# ---------------------------------------------------------------------------
# Tests: Single Agent
# ---------------------------------------------------------------------------


class TestSingleAgentInterruptions:
  """Tests for single agent triggering interruptions."""

  async def setup_runner(self, mock_model, tools=None, **agent_kwargs):
    """Setup runner with LlmAgent directly."""
    llm_agent = LlmAgent(
        name='test_agent',
        model=mock_model,
        tools=tools or [],
        **agent_kwargs,
    )
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
    )
    runner = Runner(
        app_name='test',
        agent=llm_agent,
        session_service=session_service,
    )
    return runner

  async def run_turn(self, runner, user_message):
    """Run a single turn."""
    return [
        e
        async for e in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            new_message=types.Content(
                role='user', parts=[types.Part(text=user_message)]
            ),
        )
    ]

  async def resume_turn(
      self, runner, prev_events, tool_name, tool_response_value='done'
  ):
    """Resume after an interrupt."""
    fc_ids = []
    for e in prev_events:
      if e.content and e.content.parts:
        for p in e.content.parts:
          if (
              p.function_call
              and p.function_call.name == tool_name
              and p.function_call.id
          ):
            fc_ids.append(p.function_call.id)
      if getattr(e.output, 'function_calls', None):
        for fc in e.output.function_calls:
          if fc.name == tool_name and fc.id:
            fc_ids.append(fc.id)

    if not fc_ids:
      for e in prev_events:
        if e.long_running_tool_ids:
          fc_ids = list(e.long_running_tool_ids)
          break

    invocation_id = prev_events[0].invocation_id

    fr_parts = [
        types.Part(
            function_response=types.FunctionResponse(
                name=tool_name,
                id=fc_id,
                response={'result': tool_response_value},
            )
        )
        for fc_id in fc_ids
    ]
    resume_msg = types.Content(role='user', parts=fr_parts)

    return [
        e
        async for e in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            invocation_id=invocation_id,
            new_message=resume_msg,
        )
    ]

  async def test_single_agent_yields_on_long_running_tool(self):
    """Single agent yields on Long Running Tool.

    Arrange: Set up a single agent with a long running tool.
    Act: Run the agent with a prompt that triggers the tool.
    Assert: Verify that the execution yields a long running tool interrupt.
    """

    def create_lro_tool(name='long_running_op'):
      def _impl() -> None:
        return None

      _impl.__name__ = name
      return LongRunningFunctionTool(_impl)

    fc = types.Part.from_function_call(name='long_running_op', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Final answer'])

    lro_tool = create_lro_tool()
    runner = await self.setup_runner(mock_model, tools=[lro_tool])

    # Act: Run first turn
    events = await self.run_turn(runner, 'Go')

    # Assert: Should have triggered function call
    assert any(
        any(
            p.function_call and p.function_call.name == 'long_running_op'
            for p in e.content.parts or []
        )
        for e in events
    )
    assert len(mock_model.requests) == 1

    # Act: Resume
    resume_events = await self.resume_turn(runner, events, 'long_running_op')

    # Assert: Should have completed
    assert any('Final answer' in t for t in text_parts(resume_events))
    assert len(mock_model.requests) == 2

  async def test_single_agent_confirmation(self):
    """Single agent yields on Confirmation.

    Arrange: Set up a single agent with a tool requiring confirmation.
    Act: Run the agent with a prompt that triggers the tool.
    Assert: Verify that the execution yields a tool confirmation request.
    """

    def dangerous_action() -> str:
      return 'done'

    tool = FunctionTool(dangerous_action, require_confirmation=True)

    fc = types.Part.from_function_call(name='dangerous_action', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Final answer'])

    runner = await self.setup_runner(mock_model, tools=[tool])

    # Act: Run first turn
    events = await self.run_turn(runner, 'Go')

    # Assert: Should have triggered function call
    assert any(
        any(
            p.function_call and p.function_call.name == 'dangerous_action'
            for p in e.content.parts or []
        )
        for e in events
    )
    assert len(mock_model.requests) == 1

    # Act: Resume
    resume_events = await self.resume_turn(runner, events, 'dangerous_action')

    # Assert: Should have completed
    assert any('Final answer' in t for t in text_parts(resume_events))
    assert len(mock_model.requests) == 2
