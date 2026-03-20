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

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.loop_agent import _DEFAULT_ROUTE
from google.adk.agents.loop_agent import LoopAgent
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import NodeState
from google.adk.workflow._workflow import WorkflowAgentState
from google.genai import types
import pytest
from typing_extensions import override

from ..workflow import testing_utils

END_OF_AGENT = testing_utils.END_OF_AGENT


class _TestingAgent(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, async {self.name}!')]
        ),
    )


class _TestingAgentWithEscalateAction(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, async {self.name}!')]
        ),
        actions=EventActions(escalate=True),
    )


async def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent, resumable: bool = False
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id=f'{test_name}_invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_run_async(request: pytest.FixtureRequest, resumable: bool):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent,
      ],
  )
  inc_node_name = '_increment_loop_count'
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=resumable
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)
  if resumable:
    expected_events = [
        (
            loop_agent.name,
            {
                'node_states': {
                    agent.name: NodeStatus.RUNNING.value,
                },
            },
        ),
        (agent.name, f'Hello, async {agent.name}!'),
        (
            loop_agent.name,
            {
                'node_states': {
                    agent.name: NodeStatus.COMPLETED.value,
                    inc_node_name: NodeStatus.RUNNING.value,
                },
            },
        ),
        (
            loop_agent.name,
            {
                'node_states': {
                    agent.name: NodeStatus.RUNNING.value,
                    inc_node_name: NodeStatus.COMPLETED.value,
                },
            },
        ),
        (agent.name, f'Hello, async {agent.name}!'),
        (
            loop_agent.name,
            {
                'node_states': {
                    agent.name: NodeStatus.COMPLETED.value,
                    inc_node_name: NodeStatus.RUNNING.value,
                },
            },
        ),
        (
            loop_agent.name,
            {
                'node_states': {
                    agent.name: NodeStatus.COMPLETED.value,
                    inc_node_name: NodeStatus.COMPLETED.value,
                },
            },
        ),
        (loop_agent.name, END_OF_AGENT),
    ]
  else:
    expected_events = [
        (agent.name, f'Hello, async {agent.name}!'),
        (agent.name, f'Hello, async {agent.name}!'),
    ]
  assert simplified_events == expected_events
  assert loop_agent._loop_count_key not in parent_ctx.session.state


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_run_async_twice_on_same_session(
    request: pytest.FixtureRequest, resumable: bool
):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=resumable
  )
  # Run agent once to populate session state if resumable.
  _ = [e async for e in loop_agent.run_async(parent_ctx)]

  # Test run agent twice
  parent_ctx_2 = await _create_parent_invocation_context(
      f'{request.function.__name__}_2', loop_agent, resumable=resumable
  )
  parent_ctx_2.session = parent_ctx.session
  events_2 = [e async for e in loop_agent.run_async(parent_ctx_2)]
  testing_utils.simplify_resumable_app_events(events_2)
  assert loop_agent._loop_count_key not in parent_ctx_2.session.state


@pytest.mark.asyncio
async def test_resume_async(request: pytest.FixtureRequest):
  agent_1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent_2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      max_iterations=2,
      sub_agents=[
          agent_1,
          agent_2,
      ],
  )
  inc_node_name = '_increment_loop_count'
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=True
  )
  parent_ctx.agent_states[loop_agent.name] = WorkflowAgentState(
      nodes={
          START.name: NodeState(status=NodeStatus.COMPLETED),
          agent_1.name: NodeState(status=NodeStatus.COMPLETED),
          agent_2.name: NodeState(status=NodeStatus.PENDING),
          inc_node_name: NodeState(status=NodeStatus.COMPLETED),
      },
  )
  parent_ctx.session.state[loop_agent._loop_count_key] = 1

  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)

  expected_events = [
      (
          loop_agent.name,
          {
              'node_states': {
                  START.name: NodeStatus.COMPLETED.value,
                  agent_1.name: NodeStatus.COMPLETED.value,
                  agent_2.name: NodeStatus.RUNNING.value,
                  inc_node_name: NodeStatus.COMPLETED.value,
              },
          },
      ),
      (agent_2.name, f'Hello, async {agent_2.name}!'),
      (
          loop_agent.name,
          {
              'node_states': {
                  START.name: NodeStatus.COMPLETED.value,
                  agent_1.name: NodeStatus.COMPLETED.value,
                  agent_2.name: NodeStatus.COMPLETED.value,
                  inc_node_name: NodeStatus.RUNNING.value,
              },
          },
      ),
      (
          loop_agent.name,
          {
              'node_states': {
                  START.name: NodeStatus.COMPLETED.value,
                  agent_1.name: NodeStatus.COMPLETED.value,
                  agent_2.name: NodeStatus.COMPLETED.value,
                  inc_node_name: NodeStatus.COMPLETED.value,
              },
          },
      ),
      (loop_agent.name, END_OF_AGENT),
  ]
  assert simplified_events == expected_events
  assert loop_agent._loop_count_key not in parent_ctx.session.state


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_run_async_with_escalate_action(
    request: pytest.FixtureRequest, resumable: bool
):
  non_escalating_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_non_escalating_agent'
  )
  escalating_agent = _TestingAgentWithEscalateAction(
      name=f'{request.function.__name__}_test_escalating_agent'
  )
  ignored_agent = _TestingAgent(
      name=f'{request.function.__name__}_test_ignored_agent'
  )
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_test_loop_agent',
      sub_agents=[non_escalating_agent, escalating_agent, ignored_agent],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=resumable
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)

  if resumable:
    expected_events = [
        (
            loop_agent.name,
            {
                'node_states': {
                    non_escalating_agent.name: NodeStatus.RUNNING.value,
                },
            },
        ),
        (
            non_escalating_agent.name,
            f'Hello, async {non_escalating_agent.name}!',
        ),
        (
            loop_agent.name,
            {
                'node_states': {
                    non_escalating_agent.name: NodeStatus.COMPLETED.value,
                    escalating_agent.name: NodeStatus.RUNNING.value,
                },
            },
        ),
        (
            escalating_agent.name,
            f'Hello, async {escalating_agent.name}!',
        ),
        (
            loop_agent.name,
            {
                'node_states': {
                    non_escalating_agent.name: NodeStatus.COMPLETED.value,
                    escalating_agent.name: NodeStatus.COMPLETED.value,
                },
            },
        ),
        (loop_agent.name, END_OF_AGENT),
    ]
  else:
    expected_events = [
        (
            non_escalating_agent.name,
            f'Hello, async {non_escalating_agent.name}!',
        ),
        (
            escalating_agent.name,
            f'Hello, async {escalating_agent.name}!',
        ),
    ]
  assert simplified_events == expected_events
  assert loop_agent._loop_count_key not in parent_ctx.session.state


@pytest.mark.asyncio
async def test_grandchild_escalation_ignored(request: pytest.FixtureRequest):
  """Verifies that an escalate action from a grandchild agent does not break the loop."""
  # Grandchild agent that escalates
  grandchild = _TestingAgentWithEscalateAction(
      name='grandchild_agent',
  )
  # Child agent (Workflow) that wraps the grandchild.
  # It simply runs and yields the grandchild's event.
  child = Workflow(
      name='child_agent',
      edges=[(START, grandchild)],
  )

  # Loop runs child agent. Max iterations = 2.
  loop_agent = LoopAgent(
      name=f'{request.function.__name__}_loop_agent',
      sub_agents=[child],
      max_iterations=2,
  )

  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, loop_agent, resumable=True
  )
  events = [e async for e in loop_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)

  # Check for loop iteration events to confirm it ran more than once.
  # simplify_resumable_app_events returns (author, content/state) tuples.
  # We expect the loop to run twice because the escalation is from 'grandchild',
  # but the loop agent only listens to 'child'.

  # Count "Hello, async grandchild_agent!" occurrences
  hello_counts = sum(
      1
      for e in simplified_events
      if e == ('grandchild_agent', 'Hello, async grandchild_agent!')
  )
  assert hello_counts == 2
