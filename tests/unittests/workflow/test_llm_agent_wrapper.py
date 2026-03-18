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

"""Tests for LlmAgentWrapper and build_node auto-wrapping."""

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

# -- Helpers --


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


# -- Unit tests for LlmAgentWrapper --


class TestLlmAgentWrapperValidation:

  def test_task_mode_accepted(self):
    agent = _make_task_agent(mode='task')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'test_agent'

  def test_single_turn_mode_accepted(self):
    agent = _make_task_agent(mode='single_turn')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'test_agent'

  def test_chat_mode_rejected(self):
    agent = _make_task_agent(mode='chat')
    with pytest.raises(ValueError, match='task and single_turn'):
      LlmAgentWrapper(agent=agent)

  def test_name_defaults_to_agent_name(self):
    agent = _make_task_agent(name='my_agent')
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.name == 'my_agent'

  def test_name_override(self):
    agent = _make_task_agent(name='my_agent')
    wrapper = LlmAgentWrapper(agent=agent, name='custom_name')
    assert wrapper.name == 'custom_name'

  def test_rerun_on_resume_defaults_true(self):
    agent = _make_task_agent()
    wrapper = LlmAgentWrapper(agent=agent)
    assert wrapper.rerun_on_resume is True

  def test_workflow_as_sub_agent_rejected(self):
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


# -- Unit tests for build_node auto-wrapping --


class TestBuildNodeAutoWrap:

  def test_task_mode_wrapped(self):
    agent = _make_task_agent(mode='task')
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)
    assert node.agent is agent

  def test_single_turn_mode_wrapped(self):
    agent = _make_task_agent(mode='single_turn')
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)

  def test_default_mode_auto_converted_to_single_turn(self):
    # No explicit mode → defaults to 'chat', auto-converted to single_turn
    agent = WorkflowLlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        instruction='Test.',
    )
    node = build_node(agent)
    assert isinstance(node, LlmAgentWrapper)
    assert agent.mode == 'single_turn'
    # Internal coordinator must also be updated.
    assert agent._coordinator.mode == 'single_turn'

  def test_explicit_chat_mode_rejected(self):
    agent = _make_task_agent(mode='chat')
    with pytest.raises(ValueError, match="mode='chat'.*not supported"):
      build_node(agent)

  def test_default_mode_coordinator_synced_in_workflow(self):
    """Coordinator mode is synced when LlmAgent is used in Workflow edges."""
    agent = WorkflowLlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        instruction='Test.',
    )
    assert agent.mode == 'chat'
    assert agent._coordinator.mode == 'chat'
    Workflow(name='wf', edges=[(START, agent)])
    assert agent.mode == 'single_turn'
    assert agent._coordinator.mode == 'single_turn'

  def test_name_override_in_build_node(self):
    agent = _make_task_agent(mode='task')
    node = build_node(agent, name='override')
    assert isinstance(node, LlmAgentWrapper)
    assert node.name == 'override'


# -- Integration tests: finish_task extraction --


@pytest.mark.asyncio
async def test_wrapper_extracts_finish_task_output(
    request: pytest.FixtureRequest,
):
  """Wrapper emits Event(output=output) from finish_task action."""
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
async def test_wrapper_single_turn_mode(request: pytest.FixtureRequest):
  """Single turn mode agents work the same way through the wrapper."""
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

  with _mock_agent_run(
      agent,
      event_output={'result': 'Done processing.'},
  ):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == {'result': 'Done processing.'}


@pytest.mark.asyncio
async def test_wrapper_validates_input_schema_dict(
    request: pytest.FixtureRequest,
):
  """Wrapper validates dict node_input against input_schema."""
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
async def test_wrapper_validates_input_schema_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Wrapper raises ValidationError for invalid input against schema."""
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
  # Manually set node_input to invalid data by providing it via a predecessor.
  # Since we can't easily inject invalid input through the workflow,
  # test the wrapper's run() directly with a real context.
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
async def test_wrapper_auto_wrap_in_workflow(request: pytest.FixtureRequest):
  """WorkflowLlmAgent in task mode is auto-wrapped when used in edges."""
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
async def test_wrapper_single_turn_event_output(
    request: pytest.FixtureRequest,
):
  """Single_turn agents emit event.output directly; wrapper propagates it."""
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

  # Simulate single_turn: LlmAgent._extract_output sets event.output
  # directly (no finish_task).
  with _mock_agent_run(agent, event_output='LLM response text'):
    events = [e async for e in wf.run_async(ctx)]

  assert len(capture.received_inputs) == 1
  assert capture.received_inputs[0] == 'LLM response text'


@pytest.mark.asyncio
async def test_wrapper_sets_branch_for_isolation(
    request: pytest.FixtureRequest,
):
  """Wrapper sets a branch on IC for content isolation."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  # Track the branch set on the context passed to the agent.
  captured_branches = []

  async def fake_run(*, ctx, node_input):
    captured_branches.append(ctx._invocation_context.branch)
    yield Event(
        invocation_id='test_inv',
        author=agent.name,
        content=types.Content(parts=[types.Part(text='response')]),
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

  # The branch should start with 'node:' (not 'task:') and include
  # the agent name.
  assert len(captured_branches) == 1
  assert captured_branches[0].startswith('node:')
  assert agent.name in captured_branches[0]


@pytest.mark.asyncio
async def test_wrapper_task_mode_no_branch(
    request: pytest.FixtureRequest,
):
  """Task mode wrapper does NOT set a branch (needed for HITL)."""
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

  # Task mode should NOT set a branch — it needs to see unbranched
  # user messages for HITL multi-turn interaction.
  assert len(captured_branches) == 1
  assert captured_branches[0] is None


@pytest.mark.asyncio
async def test_wrapper_passes_node_input_as_content(
    request: pytest.FixtureRequest,
):
  """Wrapper converts string node_input to types.Content for the agent."""
  agent = _make_task_agent(mode='single_turn')
  wrapper = LlmAgentWrapper(agent=agent)

  # Track the node_input passed to the agent.
  captured_inputs = []

  async def fake_run(*, ctx, node_input):
    captured_inputs.append(node_input)
    yield Event(
        invocation_id='test_inv',
        author=agent.name,
        content=types.Content(parts=[types.Part(text='response')]),
    )

  # Use a predecessor node that outputs a string, so the wrapper
  # receives it as node_input and converts to Content.
  predecessor = TestingNode(name='predecessor', output='hello world')
  wf = Workflow(
      name='test_wf',
      edges=[
          (START, predecessor),
          (predecessor, wrapper),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  object.__setattr__(agent, 'run', fake_run)
  try:
    events = [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(agent, 'run', agent.__class__.run)

  # node_input should be a types.Content (converted from string).
  assert len(captured_inputs) == 1
  assert isinstance(captured_inputs[0], types.Content)
  assert captured_inputs[0].parts[0].text == 'hello world'


@pytest.mark.asyncio
async def test_single_turn_first_node_receives_user_content(
    request: pytest.FixtureRequest,
):
  """Single_turn LlmAgent as first node sees user content in LLM request.

  Regression test: the wrapper creates a branched context for content
  isolation. Previously, _SingleLlmAgent skipped re-appending user
  content under the branch because the identity check matched the
  runner's original Content object, making it invisible to the
  branch-filtered content processor.
  """
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
