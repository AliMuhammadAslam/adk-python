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

"""Tests for Workflow error handling and graceful shutdown."""

import asyncio

from google.adk import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow._node import node
from google.adk.workflow._workflow_class import Workflow
from google.genai import types
import pytest


class CustomError(Exception):
  """A custom error for testing."""


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  try:
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      events.append(event)
  except CustomError:
    pass
  return events, ss, session


@pytest.mark.asyncio
async def test_workflow_returns_normally_on_node_failure():
  """Workflow returns normally when a node fails, without duplicate error events.

  Setup: Workflow with a single node that raises CustomError.
  Act: Run the workflow through Runner.
  Assert:
    - Collected events contain exactly one error event from the failing node.
    - No duplicate error event is emitted for the workflow itself.
  """

  # Given a workflow with a failing node
  @node()
  def failing_node(ctx: Context):
    raise CustomError('Node failed')
    yield 'output'

  wf = Workflow(
      name='test_error_workflow',
      edges=[
          (START, failing_node),
      ],
  )

  # When the workflow is executed
  events, ss, session = await _run_workflow(wf)

  # Then the result matches expectations
  # Verify that we have an error event from the failing node
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Node failed'

  # Verify that there is NO duplicate error event from the workflow itself
  # The workflow event would have path "test_error_workflow@1"
  workflow_error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_info
      and e.node_info.path == 'test_error_workflow@1'
  ]
  assert len(workflow_error_events) == 0


@pytest.mark.asyncio
async def test_node_cancellation_on_sibling_failure():
  """Node is cancelled and does not produce output when a sibling node fails.

  Setup: Workflow with a slow node and a node that fails quickly.
  Act: Run the workflow through Runner.
  Assert:
    - Collected events contain an error event from the failing node.
    - The slow node does not produce any output event.
  """

  # Given a workflow with a slow node and a failing node
  @node()
  async def slow_node(ctx: Context):
    await asyncio.sleep(10)
    yield 'Slow'

  @node()
  async def fail_node(ctx: Context):
    await asyncio.sleep(0.1)
    raise CustomError('Fail')
    yield 'Fail'

  wf = Workflow(
      name='test_workflow_cancellation_sibling',
      edges=[
          (START, slow_node),
          (START, fail_node),
      ],
  )

  # When the workflow is executed
  events, ss, session = await _run_workflow(wf)

  # Then the result matches expectations
  # Verify that we have an error event from fail_node
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Fail'

  # Verify that slow_node did NOT produce output 'Slow'
  slow_outputs = [
      e
      for e in events
      if isinstance(e, Event) and hasattr(e, 'message') and e.message == 'Slow'
  ]
  assert len(slow_outputs) == 0


@pytest.mark.asyncio
async def test_nested_workflow_cancellation_on_sibling_failure():
  """Nested workflow and its internal nodes are cancelled when a sibling fails.

  Setup: Outer workflow with an inner workflow and a failing node.
  Act: Run the outer workflow through Runner.
  Assert:
    - Collected events contain an error event from the failing node.
    - The inner workflow's slow node does not produce any output event.
  """

  # Given an outer workflow with an inner workflow and a failing node
  @node()
  async def inner_slow_node(ctx: Context):
    await asyncio.sleep(10)
    yield 'Inner Slow'

  inner_wf = Workflow(
      name='inner_workflow',
      edges=[
          (START, inner_slow_node),
      ],
  )

  @node()
  async def fail_node(ctx: Context):
    await asyncio.sleep(0.1)
    raise CustomError('Fail')
    yield 'Fail'

  outer_wf = Workflow(
      name='outer_workflow',
      edges=[
          (START, inner_wf),
          (START, fail_node),
      ],
  )

  # When the outer workflow is executed
  events, ss, session = await _run_workflow(outer_wf)

  # Then the result matches expectations
  # Verify that we have an error event from fail_node
  error_events = [
      e
      for e in events
      if isinstance(e, Event) and e.error_code == 'CustomError'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_message == 'Fail'

  # Verify that inner_slow_node did NOT produce output 'Inner Slow'
  slow_outputs = [
      e
      for e in events
      if isinstance(e, Event)
      and hasattr(e, 'message')
      and e.message == 'Inner Slow'
  ]
  assert len(slow_outputs) == 0
