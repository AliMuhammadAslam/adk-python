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

"""Tests for LlmAgentWrapper and build_node auto-wrapping.

Verifies that LlmAgentWrapper correctly adapts LlmAgent for use as a
workflow graph node, including mode validation, input conversion,
content isolation, and output extraction.
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.llm.task._task_models import TaskResult
from google.adk.agents.llm_agent import LlmAgent as WorkflowLlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper as LlmAgentWrapper
from google.adk.workflow.utils._workflow_graph_utils import build_node
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import TestingNode

# --- Helpers ---


class StoryOutput(BaseModel):
  title: str
  content: str


class StoryInput(BaseModel):
  topic: str
  style: str = 'narrative'


def _make_task_agent(
    name: str = 'test_agent',
    mode: str = 'task',
    output_schema: type[BaseModel] | None = None,
    input_schema: type[BaseModel] | None = None,
) -> WorkflowLlmAgent:
  """Creates a WorkflowLlmAgent with mocked run()."""
  agent = WorkflowLlmAgent(
      name=name,
      model='gemini-2.5-flash',
      instruction='Test agent.',
      mode=mode,
      output_schema=output_schema,
      input_schema=input_schema,
  )
  return agent


def _mock_agent_run(
    agent, finish_output=None, content_text=None, event_output=None
):
  """Patches agent.run to yield events with optional finish_task.

  Uses object.__setattr__ to bypass Pydantic's frozen model protection.
  Returns a context manager that restores the original run method.

  Args:
    agent: The agent to mock.
    finish_output: If set, yields a finish_task event with this output.
    content_text: If set, yields a content event with this text.
    event_output: If set, yields an event with output= set directly
        (simulates single_turn mode output from LlmAgent._extract_output).
  """

  async def fake_run(*, ctx, node_input):
    if content_text:
      yield Event(
          invocation_id='test_inv',
          author=agent.name,
          content=types.Content(parts=[types.Part(text=content_text)]),
      )
    if finish_output is not None:
      yield Event(
          invocation_id='test_inv',
          author=agent.name,
          actions=EventActions(
              finish_task=TaskResult(output=finish_output).model_dump(),
          ),
      )
    if event_output is not None:
      from google.adk.events.event import NodeInfo

      # Simulate call_llm internal event (subgraph, not direct output).
      yield Event(
          invocation_id='test_inv',
          author=agent.name,
          output=event_output,
          node_info=NodeInfo(
              path=f'parent/{agent.name}/call_llm',
              execution_id='inner_exec',
          ),
      )
      # Simulate agent-level output emitted by
      # LlmAgent.run_node_impl() for single_turn mode.
      yield Event(output=event_output)

  original = agent.run
  object.__setattr__(agent, 'run', fake_run)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(agent, 'run', original)

  return _Ctx()


def _mock_wrapper_run(wrapper, content_text=None):
  """Patches wrapper._single.run to yield a final LLM response event.

  For single_turn agents using the _SingleLlmAgent path, the wrapper
  extracts output from final LLM response events.

  Simulates realistic _SingleLlmAgent output: LLM events carry a
  subgraph path (e.g. '{name}/call_llm') so the outer node_runner
  treats them as subgraph events, not as the wrapper's direct output.

  Args:
    wrapper: The LlmAgentWrapper to mock.
    content_text: The text for the final LLM response event.
  """
  target = wrapper._single if wrapper._single is not None else wrapper.agent
  name = wrapper.name

  async def fake_run(*, ctx, node_input):
    if content_text:
      event = Event(
          invocation_id='test_inv',
          author=name,
          content=types.Content(parts=[types.Part(text=content_text)]),
      )
      # Set a subgraph path matching real _SingleLlmAgent behavior.
      # ctx.node_path already includes the wrapper's name (e.g.
      # 'test_wf/test_agent'), so call_llm is a direct child.
      event.node_info.path = f'{ctx.node_path}/call_llm'
      yield event

  original = target.run
  object.__setattr__(target, 'run', fake_run)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(target, 'run', original)

  return _Ctx()


# --- Validation ---


class TestLlmAgentWrapperValidation:

  def test_task_mode_accepted(self):
    """Wrapping a task-mode agent succeeds."""
    agent = _make_task_agent(mode='task')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'test_agent'

  def test_single_turn_mode_accepted(self):
    """Wrapping a single_turn-mode agent succeeds."""
    agent = _make_task_agent(mode='single_turn')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'test_agent'

  def test_chat_mode_rejected(self):
    """Wrapping a chat-mode agent raises ValueError."""
    agent = _make_task_agent(mode='chat')
    with pytest.raises(ValueError, match='task and single_turn'):
      LlmAgentWrapper(agent=agent)

  def test_name_defaults_to_agent_name(self):
    """Wrapper name defaults to the inner agent's name."""
    agent = _make_task_agent(name='my_agent')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'my_agent'

  def test_name_override(self):
    """Explicit name overrides the agent's name."""
    agent = _make_task_agent(name='my_agent')
    wrapper = LlmAgentWrapper(agent=agent, name='custom_name')
    assert wrapper.name == 'custom_name'

  def test_rerun_on_resume_defaults_true(self):
    """Wrapper defaults to rerun_on_resume=True."""
    agent = _make_task_agent()
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.rerun_on_resume is True

  def test_workflow_as_sub_agent_rejected(self):
    """Using a Workflow as a sub_agent of LlmAgent raises ValueError."""
    wf = Workflow(
        name='my_workflow',
        edges=[(START, lambda: 'done')],
    )
    with pytest.raises(
        ValueError, match='Workflow.*cannot be used as a sub_agent'
    ):
      WorkflowLlmAgent(
          name='parent',
          model='gemini-2.5-flash',
          instruction='Test.',
          sub_agents=[wf],
      )


# --- build_node auto-wrapping ---


class TestBuildNodeAutoWrap:

  def test_task_mode_wrapped(self):
    """build_node wraps a task-mode LlmAgent in LlmAgentWrapper."""
    agent = _make_task_agent(mode='task')
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)
    assert node.agent is agent

  def test_single_turn_mode_wrapped(self):
    """build_node wraps a single_turn-mode LlmAgent in LlmAgentWrapper."""
    agent = _make_task_agent(mode='single_turn')
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)

  def test_default_mode_auto_converted_to_single_turn(self):
    """LlmAgent with default mode is auto-converted to single_turn."""
    agent = WorkflowLlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        instruction='Test.',
    )
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)
    assert agent.mode == 'single_turn'

  def test_explicit_chat_mode_rejected(self):
    """build_node rejects LlmAgent with explicit chat mode."""
    agent = _make_task_agent(mode='chat')
    with pytest.raises(ValueError, match="mode='chat'.*not supported"):
      build_node(agent)

  def test_default_mode_synced_in_workflow(self):
    """LlmAgent mode is synced to single_turn when used in Workflow edges."""
    agent = WorkflowLlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        instruction='Test.',
    )
    assert agent.mode == 'chat'
    Workflow(name='wf', edges=[(START, agent)])
    assert agent.mode == 'single_turn'

  def test_name_override_in_build_node(self):
    """build_node respects explicit name override."""
    agent = _make_task_agent(mode='task')
    node = build_node(agent, name='override')
    assert isinstance(node, LlmAgentWrapper)
    assert node.name == 'override'


# --- Integration tests ---


@pytest.mark.asyncio
async def test_task_mode_finish_task_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Task mode wrapper extracts finish_task output for downstream nodes."""
  agent = _make_task_agent(mode='task')
  wrapper = LlmAgentWrapper(agent=agent)

  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, wrapper),
          (wrapper, capture),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(
      agent,
      finish_output={'title': 'My Story', 'content': 'Once upon a time...'},
      content_text='Writing story...',
  ):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == {
      'title': 'My Story',
      'content': 'Once upon a time...',
  }


def test_task_mode_sets_wait_for_output():
  """Task mode wrapper sets wait_for_output=True."""
  agent = _make_task_agent(mode='task')
  wrapper = LlmAgentWrapper(agent=agent)
  assert wrapper.wait_for_output is True


def test_single_turn_mode_no_wait_for_output():
  """Single_turn mode wrapper does not set wait_for_output."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)
  assert wrapper.wait_for_output is False


@pytest.mark.asyncio
async def test_single_turn_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper output is received by downstream nodes."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, wrapper),
          (wrapper, capture),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_wrapper_run(wrapper, content_text='Done processing.'):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == 'Done processing.'


@pytest.mark.asyncio
async def test_valid_input_schema_passes_through(
    request: pytest.FixtureRequest,
):
  """Valid dict input matching input_schema is accepted."""
  agent = _make_task_agent(
      mode='task',
      input_schema=StoryInput,
  )
  wrapper = LlmAgentWrapper(agent=agent)

  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, wrapper),
          (wrapper, capture),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(agent, finish_output={'result': 'ok'}):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == {'result': 'ok'}


@pytest.mark.asyncio
async def test_invalid_input_schema_raises_validation_error(
    request: pytest.FixtureRequest,
):
  """Invalid input that doesn't match input_schema raises ValidationError."""
  agent = _make_task_agent(
      mode='task',
      input_schema=StoryInput,
  )
  wrapper = LlmAgentWrapper(agent=agent)

  wf = Workflow(
      name='test_wf',
      edges=[(START, wrapper)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)
  ic = ctx.model_copy(update={'branch': None})
  agent_ctx = Context(
      invocation_context=ic,
      node_path='test_wf',
      execution_id='test_exec',
  )

  with _mock_agent_run(agent, finish_output={'result': 'ok'}):
    with pytest.raises(ValidationError):
      async for _ in wrapper.run(
          ctx=agent_ctx,
          node_input={'style': 'comedy'},
      ):
        pass


@pytest.mark.asyncio
async def test_auto_wrap_in_workflow_edges(request: pytest.FixtureRequest):
  """LlmAgent used directly in Workflow edges is auto-wrapped."""
  agent = _make_task_agent(mode='task')

  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, agent),
          (agent, capture),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(
      agent,
      finish_output={'result': 'auto-wrapped'},
  ):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == {'result': 'auto-wrapped'}


@pytest.mark.asyncio
async def test_single_turn_extracts_output_from_llm_response(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper extracts text output from final LLM response."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, wrapper),
          (wrapper, capture),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_wrapper_run(wrapper, content_text='LLM response text'):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == 'LLM response text'


@pytest.mark.asyncio
async def test_single_turn_sets_branch_for_content_isolation(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper isolates content via a branch on the context."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  captured_branches = []
  target = wrapper._single if wrapper._single is not None else agent

  async def fake_run(*, ctx, node_input):
    captured_branches.append(ctx._invocation_context.branch)
    event = Event(
        invocation_id='test_inv',
        author=agent.name,
        content=types.Content(parts=[types.Part(text='response')]),
    )
    event.node_info.path = f'{ctx.node_path}/call_llm'
    yield event

  wf = Workflow(
      name='test_wf',
      edges=[(START, wrapper)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  original = target.run
  object.__setattr__(target, 'run', fake_run)
  try:
    events = [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(target, 'run', original)

  assert len(captured_branches) == 1
  assert captured_branches[0].startswith('node:')
  assert agent.name in captured_branches[0]


@pytest.mark.asyncio
async def test_task_mode_does_not_set_branch(
    request: pytest.FixtureRequest,
):
  """Task mode wrapper does not set a branch, preserving HITL visibility."""
  agent = _make_task_agent(mode='task')
  wrapper = LlmAgentWrapper(agent=agent)

  captured_branches = []

  async def fake_run(*, ctx, node_input):
    captured_branches.append(ctx._invocation_context.branch)
    yield Event(
        invocation_id='test_inv',
        author=agent.name,
        actions=EventActions(
            finish_task={'output': {'result': 'done'}},
        ),
    )

  wf = Workflow(
      name='test_wf',
      edges=[(START, wrapper)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  object.__setattr__(agent, 'run', fake_run)
  try:
    events = [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(agent, 'run', agent.__class__.run)

  assert len(captured_branches) == 1
  assert captured_branches[0] is None


@pytest.mark.asyncio
async def test_single_turn_converts_string_input_to_content(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper converts string node_input to types.Content."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  captured_inputs = []
  target = wrapper._single if wrapper._single is not None else agent

  async def fake_run(*, ctx, node_input):
    captured_inputs.append(node_input)
    event = Event(
        invocation_id='test_inv',
        author=agent.name,
        content=types.Content(parts=[types.Part(text='response')]),
    )
    event.node_info.path = f'{ctx.node_path}/call_llm'
    yield event

  predecessor = TestingNode(name='predecessor', output='hello world')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, predecessor),
          (predecessor, wrapper),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  original = target.run
  object.__setattr__(target, 'run', fake_run)
  try:
    events = [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(target, 'run', original)

  assert len(captured_inputs) == 1
  assert isinstance(captured_inputs[0], types.Content)
  assert captured_inputs[0].parts[0].text == 'hello world'


@pytest.mark.asyncio
async def test_single_turn_first_node_receives_user_content(
    request: pytest.FixtureRequest,
):
  """Single_turn LlmAgent as first node sees the user message in LLM request."""
  from google.adk.apps.app import App

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(
      responses=['extracted output'],
  )
  agent = WorkflowLlmAgent(
      name='process_request',
      model=mock_model,
      instruction='Extract info from the user message.',
  )

  wf = Workflow(
      name='test_wf',
      edges=[('START', agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=wf,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(
      testing_utils.get_user_content('I want 3 days off for vacation')
  )

  # The mock model should have been called and its request should
  # contain the user message.
  assert len(mock_model.requests) == 1
  contents = mock_model.requests[0].contents
  user_texts = [
      part.text
      for c in contents
      if c.role == 'user'
      for part in c.parts or []
      if part.text
  ]
  assert any(
      '3 days' in t for t in user_texts
  ), f'User content not visible to LLM. Contents: {contents}'
