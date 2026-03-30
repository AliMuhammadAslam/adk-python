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

Verifies that Runner can execute standalone BaseNode instances,
persist events to session, handle resume (HITL), and yield events correctly.
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow_class import Workflow
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    text = node_input.parts[0].text if node_input else 'empty'
    yield f'Echo: {text}'


async def _run_node(node, message='hello'):
  """Run a BaseNode via Runner(node=...) and return (events, ss, session)."""
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


def _make_interrupt_event(fc_name='get_input', fc_id='fc-1'):
  """Create an interrupt Event with a long-running function call."""
  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=fc_name, args={}, id=fc_id
                  )
              )
          ]
      ),
      long_running_tool_ids={fc_id},
  )


def _make_resume_message(fc_name='get_input', fc_id='fc-1', response=None):
  """Create a user message with a function response for resuming."""
  return types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=fc_name,
                  id=fc_id,
                  response=response or {},
              )
          )
      ],
      role='user',
  )


async def _run_two_turns(node, msg1_text, resume_msg):
  """Run a node for two turns: initial message then resume."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text=msg1_text)], role='user')
  events1 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  events2 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=resume_msg
  ):
    events2.append(event)

  return events1, events2, runner, ss, session


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_node_output():
  """Runner yields output from a simple BaseNode."""
  events, _, _ = await _run_node(_EchoNode(name='echo'), message='hi')

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Echo: hi']


@pytest.mark.asyncio
async def test_intermediate_events_yielded():
  """Runner yields intermediate events (e.g. state), not just output."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'step': 'processing'})
      yield 'final_result'

  events, _, _ = await _run_node(_Node(name='steps'))

  state_events = [e for e in events if e.actions and e.actions.state_delta]
  assert len(state_events) >= 1
  assert [e.output for e in events if e.output is not None] == ['final_result']


@pytest.mark.asyncio
async def test_event_author_defaults_to_node_name():
  """Events are attributed to the node's name by default."""
  events, _, _ = await _run_node(_EchoNode(name='my_node'), message='hi')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'


@pytest.mark.asyncio
async def test_node_error_completes_without_output():
  """A node that raises completes the invocation with no output."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      raise RuntimeError('node failure')
      yield  # pylint: disable=unreachable

  events, _, _ = await _run_node(_Node(name='error'))

  assert [e.output for e in events if e.output is not None] == []


@pytest.mark.asyncio
async def test_node_yielding_none_produces_no_output():
  """A node that yields None produces no output event."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield None

  events, _, _ = await _run_node(_Node(name='nil'))

  assert [e.output for e in events if e.output is not None] == []


@pytest.mark.asyncio
async def test_workflow_node_output():
  """Runner drives a Workflow and yields its terminal output."""

  def upper(node_input: str) -> str:
    return node_input.upper()

  wf = Workflow(name='wf', edges=[(START, upper)])
  events, _, _ = await _run_node(wf, message='hi')

  outputs = [e.output for e in events if e.output is not None]
  assert 'HI' in outputs


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_persisted_to_session():
  """Non-partial events are persisted to the session."""
  _, ss, session = await _run_node(_EchoNode(name='echo'), message='hi')

  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  session_outputs = [e.output for e in updated.events if e.output is not None]
  assert 'Echo: hi' in session_outputs


@pytest.mark.asyncio
async def test_multiple_invocations_accumulate_events():
  """Each invocation appends events; session accumulates across runs."""
  node = _EchoNode(name='echo')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  for msg_text in ['first', 'second', 'third']:
    async for _ in runner.run_async(
        user_id='u',
        session_id=session.id,
        new_message=types.Content(
            parts=[types.Part(text=msg_text)], role='user'
        ),
    ):
      pass

  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  outputs = [e.output for e in updated.events if e.output is not None]
  assert outputs == ['Echo: first', 'Echo: second', 'Echo: third']


# ---------------------------------------------------------------------------
# yield_user_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_yield_user_message_true():
  """When yield_user_message=True, user event is yielded before node events."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='hi')], role='user')

  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=msg,
      yield_user_message=True,
  ):
    events.append(event)

  user_events = [e for e in events if e.author == 'user']
  assert len(user_events) == 1
  assert user_events[0].content.parts[0].text == 'hi'
  assert events[0].author == 'user'


@pytest.mark.asyncio
async def test_yield_user_message_false_by_default():
  """By default, user event is not yielded to the caller."""
  events, _, _ = await _run_node(_EchoNode(name='echo'), message='hi')

  user_events = [e for e in events if e.author == 'user']
  assert user_events == []


# ---------------------------------------------------------------------------
# Resume (HITL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_node_resume():
  """A standalone node resumes with resume_inputs from function response."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'result: {ctx.resume_inputs["fc-1"]["value"]}'
        return
      yield _make_interrupt_event()

  events1, events2, _, _, _ = await _run_two_turns(
      _Node(name='standalone'),
      'go',
      _make_resume_message(response={'value': 42}),
  )

  assert any(e.long_running_tool_ids for e in events1)
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'result: 42' in outputs


@pytest.mark.asyncio
async def test_resume_preserves_original_user_content():
  """On resume, Runner passes the original text as node_input, not the FR."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        text = (
            node_input.parts[0].text
            if node_input and hasattr(node_input, 'parts')
            else str(node_input)
        )
        yield f'original:{text}'
        return
      yield _make_interrupt_event(fc_name='tool')

  events1, events2, _, _, _ = await _run_two_turns(
      _Node(name='node'),
      'my original input',
      _make_resume_message(fc_name='tool', response={'v': 1}),
  )

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'original:my original input' in outputs


@pytest.mark.asyncio
async def test_plain_text_does_not_trigger_resume():
  """Sending plain text (no FR) starts fresh, does not enter resume path."""
  node = _EchoNode(name='echo')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='first')], role='user'),
  ):
    pass

  # Run 2: plain text — should start fresh
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='second')], role='user'),
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert outputs == ['Echo: second']


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_raises_on_unmatched_fr():
  """Runner raises when function response has no matching FC in session."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = _make_resume_message(fc_name='unknown', fc_id='no-such-fc')

  with pytest.raises(ValueError, match='Function call not found'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_resume_raises_on_multi_invocation_fr():
  """Runner raises when FRs resolve to different invocations."""
  call_count = [0]

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_count[0] += 1
      fc_id = f'fc-{call_count[0]}'
      yield _make_interrupt_event(fc_name='tool', fc_id=fc_id)

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptNode(name='ask'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts with fc-1
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='go')], role='user'),
  ):
    pass

  # Run 2: interrupts with fc-2 (different invocation)
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text='go again')], role='user'
      ),
  ):
    pass

  # Run 3: send FRs for both fc-1 and fc-2 (different invocations)
  msg3 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'r': 1}
              )
          ),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-2', response={'r': 2}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='resolve to multiple invocations'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg3
    ):
      pass


@pytest.mark.asyncio
async def test_mixed_fr_and_text_raises():
  """Message with both function responses and text is rejected."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(
      parts=[
          types.Part(text='some text'),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'v': 1}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='cannot contain both'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


# ---------------------------------------------------------------------------
# Default scheduler & ctx.create_task cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_node_works_without_workflow():
  """ctx.run_node() works in a standalone BaseNode (default scheduler)."""

  class _ChildNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield f'child got: {node_input}'

  class _ParentNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      result = await ctx.run_node(_ChildNode(name='child'), 'hello')
      yield f'parent got: {result}'

  events, _, _ = await _run_node(_ParentNode(name='parent'), message='go')

  outputs = [e.output for e in events if e.output is not None]
  assert 'parent got: child got: hello' in outputs


@pytest.mark.asyncio
async def test_run_node_use_as_output_attributes_child_output_to_parent():
  """Child output with use_as_output=True is attributed to the parent node."""

  class _ChildNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'child result'

  class _ParentNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      result = await ctx.run_node(
          _ChildNode(name='child'), 'hello', use_as_output=True
      )
      yield f'parent got: {result}'

  events, _, _ = await _run_node(_ParentNode(name='parent'), message='go')

  # The child's output event should list the parent's path in output_for
  child_output = next(e for e in events if e.output == 'child result')
  parent_path = next(
      e for e in events if e.output is not None and e.output != 'child result'
  ).node_info.path
  assert parent_path in child_output.node_info.output_for


