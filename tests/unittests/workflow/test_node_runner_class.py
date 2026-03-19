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

"""Tests for NodeRunner class.

Verifies that NodeRunner correctly drives a node, captures its output,
detects interrupts, and delivers events to the session.
"""

from typing import Any
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._node_runner_class import NodeRunner
from google.genai import types
import pytest

# --- Test helper nodes ---


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield node_input


class _EmptyNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    return
    yield


class _MultiEventNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(author='step1')
    yield Event(author='step2')
    yield Event(author='step3')


class _InterruptNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='long_tool', args={}, id='fc-1'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-1'},
    )


class _InterruptThenMoreNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name='long_tool', args={}, id='fc-2'
                    )
                )
            ]
        ),
        long_running_tool_ids={'fc-2'},
    )
    yield Event(author='after_interrupt_1')
    yield Event(author='after_interrupt_2')


class _ErrorNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    raise RuntimeError('node failure')
    yield  # pylint: disable=unreachable


class _OutputWithRouteNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output='routed_output', route='next')


class _StateMutatingNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ctx.state['key1'] = 'value1'
    ctx.state['key2'] = 42
    yield 'done'


class _ResumeInputReadingNode(BaseNode):
  captured: list[Any] = []

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.captured.append(ctx.resume_inputs)
    yield 'resumed'


class _ArtifactSavingNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    ctx.actions.artifact_delta['doc.txt'] = 1
    yield 'saved'


# --- Helpers ---


def _make_ctx(invocation_id='inv-test', enqueue_events=None):
  """Create a minimal Context mock with IC."""
  ic = MagicMock()
  ic.invocation_id = invocation_id
  ic.session = MagicMock()
  ic.session.state = {}
  ic.session.app_name = 'test_app'
  ic.session.user_id = 'test_user'
  ic.run_config = None

  collected = enqueue_events if enqueue_events is not None else []

  async def _enqueue(event):
    collected.append(event)

  ic.enqueue_event = AsyncMock(side_effect=_enqueue)

  ctx = MagicMock()
  ctx._invocation_context = ic
  ctx.node_path = ''
  ctx.schedule_dynamic_node = None
  return ctx, collected


# --- Tests ---


@pytest.mark.asyncio
async def test_node_output_returned_in_result():
  """Running a node that produces output returns it in the result."""
  node = _EchoNode(name='echo')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run(node_input='hello')
  assert result.output == 'hello'
  assert result.interrupt_ids == []


@pytest.mark.asyncio
async def test_no_output_returns_none():
  """Running a node that produces no output returns None."""
  node = _EmptyNode(name='empty')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert result.output is None
  assert result.interrupt_ids == []


@pytest.mark.asyncio
async def test_event_author_is_node_name():
  """Events are authored by the node's name."""
  node = _EchoNode(name='my_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run(node_input='data')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'


@pytest.mark.asyncio
async def test_event_path_contains_node_name():
  """Event node_info.path includes the node name and execution context."""
  node = _EchoNode(name='path_test')
  ctx, events = _make_ctx(invocation_id='inv-123')
  runner = NodeRunner(node=node, parent_ctx=ctx, execution_id='exec-456')
  await runner.run(node_input='data')

  output_events = [e for e in events if e.output is not None]
  event = output_events[0]
  assert event.node_info.path == 'path_test'
  assert event.node_info.execution_id == 'exec-456'
  assert event.invocation_id == 'inv-123'


@pytest.mark.asyncio
async def test_interrupt_captured_in_result():
  """A node that signals an interrupt reports it in the result."""
  node = _InterruptNode(name='interrupt_node')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert 'fc-1' in result.interrupt_ids


@pytest.mark.asyncio
async def test_node_continues_after_interrupt():
  """A node that interrupts can still produce more events before finishing."""
  node = _InterruptThenMoreNode(name='flag_finish')
  ctx, events = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert 'fc-2' in result.interrupt_ids
  assert len(events) >= 3


@pytest.mark.asyncio
async def test_state_mutations_emitted_as_delta():
  """State changes made by a node are delivered as a separate event."""
  node = _StateMutatingNode(name='state_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  all_deltas = {}
  for e in events:
    if e.actions and e.actions.state_delta:
      all_deltas.update(e.actions.state_delta)
  assert all_deltas.get('key1') == 'value1'
  assert all_deltas.get('key2') == 42


@pytest.mark.asyncio
async def test_artifact_delta_emitted():
  """Artifact saves made by a node are delivered as a delta event."""
  node = _ArtifactSavingNode(name='artifact_node')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  artifact_deltas = {}
  for e in events:
    if e.actions and e.actions.artifact_delta:
      artifact_deltas.update(e.actions.artifact_delta)
  assert 'doc.txt' in artifact_deltas


@pytest.mark.asyncio
async def test_events_enqueued_in_yield_order():
  """Multiple events from a node arrive in the order they were produced."""
  node = _MultiEventNode(name='multi')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  authors = [
      e.author for e in events if e.author in ('step1', 'step2', 'step3')
  ]
  assert authors == ['step1', 'step2', 'step3']


@pytest.mark.asyncio
async def test_node_exception_propagates():
  """A node that raises an error surfaces it to the caller."""
  node = _ErrorNode(name='error_node')
  ctx, _ = _make_ctx()
  with pytest.raises(RuntimeError, match='node failure'):
    await NodeRunner(node=node, parent_ctx=ctx).run()


@pytest.mark.asyncio
async def test_resume_inputs_available_on_context():
  """Resume inputs are accessible to the node during execution."""
  node = _ResumeInputReadingNode(name='resume_node')
  node.captured = []
  ctx, _ = _make_ctx()
  resume = {'int-1': 'user_response'}
  await NodeRunner(node=node, parent_ctx=ctx).run(resume_inputs=resume)
  assert node.captured[0] == resume


@pytest.mark.asyncio
async def test_node_path_includes_parent():
  """A child node's node_path is parent_node_path/child_name."""
  node = _EchoNode(name='child')
  ctx, events = _make_ctx()
  ctx.node_path = 'parent_path'
  runner = NodeRunner(node=node, parent_ctx=ctx)
  await runner.run(node_input='x')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].node_info.path == 'parent_path/child'


@pytest.mark.asyncio
async def test_execution_id_generated_when_omitted():
  """Each node run gets a unique execution ID by default."""
  node = _EchoNode(name='auto_id')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run(node_input='data')

  output_events = [e for e in events if e.output is not None]
  exec_id = output_events[0].node_info.execution_id
  assert exec_id
  assert isinstance(exec_id, str)


@pytest.mark.asyncio
async def test_explicit_execution_id_used():
  """A caller-specified execution ID is used for the run."""
  node = _EchoNode(name='explicit_id')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx, execution_id='my-exec-id').run(
      node_input='data'
  )

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].node_info.execution_id == 'my-exec-id'


@pytest.mark.asyncio
async def test_route_captured_in_result():
  """A node's routing decision is available in the result."""
  node = _OutputWithRouteNode(name='route_node')
  ctx, _ = _make_ctx()
  result = await NodeRunner(node=node, parent_ctx=ctx).run()
  assert result.output == 'routed_output'
  assert result.route == 'next'


@pytest.mark.asyncio
async def test_preset_author_preserved():
  """A node that sets its own author on events has that respected."""
  node = _MultiEventNode(name='multi')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run()

  authors = [
      e.author for e in events if e.author in ('step1', 'step2', 'step3')
  ]
  assert authors == ['step1', 'step2', 'step3']


class _MultiOutputNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output='first')
    yield Event(output='second')


@pytest.mark.asyncio
async def test_multiple_outputs_raises():
  """A node that produces more than one output is rejected."""
  node = _MultiOutputNode(name='multi_out')
  ctx, _ = _make_ctx()
  with pytest.raises(ValueError, match='at most one output'):
    await NodeRunner(node=node, parent_ctx=ctx).run()


@pytest.mark.asyncio
async def test_all_events_delivered():
  """All events from a node are delivered to the session."""
  node = _EchoNode(name='enqueue_test')
  ctx, events = _make_ctx()
  await NodeRunner(node=node, parent_ctx=ctx).run(node_input='data')
  assert len(events) >= 1
