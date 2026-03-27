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

"""Tests for Runner(node=...).

Verifies that Runner can execute standalone BaseNode instances and
produces the expected output events.
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.llm._single_agent_react_node import SingleAgentReactNode
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.genai import types
import pytest

from .. import testing_utils


async def _run_node(node, message='Hi'):
  """Run a BaseNode via Runner(node=...) and return events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events


@pytest.mark.asyncio
async def test_runner_executes_simple_node():
  """Runner produces output from a simple BaseNode."""

  class _EchoNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Any, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'echo'

  events = await _run_node(_EchoNode(name='echo'))

  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 1
  assert output_events[0].output == 'echo'


@pytest.mark.asyncio
async def test_react_node_text_response_produces_single_output():
  """SingleAgentReactNode produces exactly one output event for text."""
  mock_model = testing_utils.MockModel.create(responses=['Hello!'])
  llm_agent = LlmAgent(name='llm', model=mock_model, tools=[])
  react_node = SingleAgentReactNode(name='react', agent=llm_agent)

  events = await _run_node(react_node)

  output_events = [e for e in events if e.output is not None]
  assert len(output_events) == 1
  assert output_events[0].output == 'Hello!'


@pytest.mark.asyncio
async def test_react_node_tool_loop_produces_single_output():
  """After a tool call loop, only one text output event appears."""

  def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

  fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
  mock_model = testing_utils.MockModel.create(responses=[fc, 'Result is 3.'])
  llm_agent = LlmAgent(name='llm', model=mock_model, tools=[add])
  react_node = SingleAgentReactNode(name='react', agent=llm_agent)

  events = await _run_node(react_node, message='Add 1+2')

  text_outputs = [
      e.output
      for e in events
      if e.output is not None and isinstance(e.output, str)
  ]
  assert len(text_outputs) == 1
  assert 'Result is 3.' in text_outputs[0]
