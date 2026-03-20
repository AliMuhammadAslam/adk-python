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

"""Tests for Runner driving a BaseNode root.

Verifies that Runner can accept a BaseNode (not just BaseAgent),
drive it through NodeRunner, persist events to session, and
yield them to the caller.
"""

from typing import Any
from typing import AsyncGenerator

from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.genai import types
import pytest

# --- Test helper nodes ---


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Any, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    text = node_input.parts[0].text if node_input else 'empty'
    yield f'Echo: {text}'


class _TwoOutputStepsNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Any, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(state={'step': 'processing'})
    yield 'final_result'


class _ErrorNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Any, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    raise RuntimeError('node failure')
    yield


# --- Helpers ---


async def _run(node, message='hello'):
  """Run a node through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


# --- Tests ---


@pytest.mark.asyncio
async def test_node_output_returned_to_caller():
  """Runner yields the node's output event to the caller."""
  events, _, _ = await _run(_EchoNode(name='echo'), message='hi')
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Echo: hi']


@pytest.mark.asyncio
async def test_events_persisted_to_session():
  """Non-partial events are persisted to the session."""
  _, ss, session = await _run(_EchoNode(name='echo'), message='hi')
  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  session_outputs = [e.output for e in updated.events if e.output is not None]
  assert 'Echo: hi' in session_outputs


@pytest.mark.asyncio
async def test_non_output_events_also_yielded():
  """Runner yields intermediate events (e.g. state), not just output."""
  events, _, _ = await _run(_TwoOutputStepsNode(name='steps'))
  state_events = [e for e in events if e.actions and e.actions.state_delta]
  assert len(state_events) >= 1
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['final_result']


@pytest.mark.asyncio
async def test_node_error_completes_without_events():
  """A node that raises an error ends the invocation cleanly."""
  # TODO: Propagate node errors to the caller. Currently the error
  # is swallowed by the background task and Runner completes with
  # no output events.
  events, _, _ = await _run(_ErrorNode(name='error'))
  outputs = [e.output for e in events if e.output is not None]
  assert outputs == []


@pytest.mark.asyncio
async def test_event_author_defaults_to_node_name():
  """Events are attributed to the node's name by default."""
  events, _, _ = await _run(_EchoNode(name='my_node'), message='hi')
  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'
