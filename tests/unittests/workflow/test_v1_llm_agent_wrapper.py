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

from google.adk.agents.context import Context
from google.adk.agents.llm_agent_1x import LlmAgent as V1LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow._v1_llm_agent_wrapper import _V1LlmAgentWrapper
from google.genai import types
import pytest


def _make_v1_agent(mode='task'):
  return V1LlmAgent(
      name='test_v1_agent',
      model='gemini-2.5-flash',
      instruction='Test instruction',
      mode=mode,
  )


def test_task_mode_sets_wait_for_output():
  agent = _make_v1_agent(mode='task')
  wrapper = _V1LlmAgentWrapper(agent=agent)
  assert wrapper.wait_for_output is True


def test_single_turn_does_not_set_wait_for_output():
  agent = _make_v1_agent(mode='single_turn')
  wrapper = _V1LlmAgentWrapper(agent=agent)
  assert wrapper.wait_for_output is False


def test_chat_mode_sets_wait_for_output():
  agent = _make_v1_agent(mode='chat')
  wrapper = _V1LlmAgentWrapper(agent=agent)
  assert wrapper.wait_for_output is True


@pytest.mark.asyncio
async def test_task_mode_proceeds_on_finish_task():
  agent = _make_v1_agent(mode='task')
  wrapper = _V1LlmAgentWrapper(agent=agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        actions=EventActions(finish_task={'output': 'done_output'}),
    )

  object.__setattr__(agent, 'run_async', mock_run_async)

  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ctx._invocation_context = MagicMock()
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  assert len(events) == 1
  assert events[0].output == 'done_output'


@pytest.mark.asyncio
async def test_task_mode_does_not_proceed_without_finish_task():
  agent = _make_v1_agent(mode='task')
  wrapper = _V1LlmAgentWrapper(agent=agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(parts=[types.Part(text='Working...')]),
    )

  object.__setattr__(agent, 'run_async', mock_run_async)

  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ctx._invocation_context = MagicMock()
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  assert len(events) == 1
  assert events[0].output is None


@pytest.mark.asyncio
async def test_chat_mode_yields_events_directly():
  agent = _make_v1_agent(mode='chat')
  wrapper = _V1LlmAgentWrapper(agent=agent)

  async def mock_run_async(*args, **kwargs):
    yield Event(
        invocation_id='inv',
        author='test_v1_agent',
        content=types.Content(parts=[types.Part(text='Hello from chat')]),
    )

  object.__setattr__(agent, 'run_async', mock_run_async)

  from unittest.mock import MagicMock

  ctx = MagicMock(spec=Context)
  ctx._invocation_context = MagicMock()
  ctx.node_path = 'wf'

  events = []
  async for e in wrapper._run_impl(ctx=ctx, node_input='hello'):
    events.append(e)

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'Hello from chat'
  assert events[0].output is None
