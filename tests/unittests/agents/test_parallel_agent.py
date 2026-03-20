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

"""Tests for the ParallelAgent."""

import asyncio
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import START
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import WorkflowAgentState
from google.genai import types
import pytest
from typing_extensions import override

from ..workflow import testing_utils

END_OF_AGENT = testing_utils.END_OF_AGENT


class _TestingAgent(BaseAgent):

  delay: float = 0
  """The delay before the agent generates an event."""

  @override
  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self.delay > 0:
      await asyncio.sleep(self.delay)
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            parts=[types.Part(text=f'Hello, async {self.name}!')]
        ),
    )


class _TestingAgentWithException(_TestingAgent):
  """Mock agent for testing."""

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
    raise Exception('Mock exception')


class _TestingAgentInfiniteEvents(_TestingAgent):
  """Mock agent for testing."""

  @override
  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[Event, None]:
    while True:
      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=types.Content(
              parts=[types.Part(text=f'Hello, async {self.name}!')]
          ),
      )
      # Yield control to allow other tasks to run
      await asyncio.sleep(0.01)


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
  agent1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, resumable=resumable
  )
  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  simplified_events = testing_utils.simplify_resumable_app_events(events)

  if resumable:
    # Check for presence of events from both agents
    content_events = [e for e in events if e.content]
    assert len(content_events) == 2
    authors = {e.author for e in content_events}
    assert authors == {agent1.name, agent2.name}

    # Verify final state
    last_event = events[-1]
    assert last_event.author == parallel_agent.name
    assert last_event.actions.end_of_agent

  else:
    assert len(events) == 2
    authors = {e.author for e in events}
    assert authors == {agent1.name, agent2.name}


@pytest.mark.asyncio
@pytest.mark.parametrize('is_resumable', [True, False])
async def test_run_async_branches(
    request: pytest.FixtureRequest, is_resumable: bool
):
  agent1 = _TestingAgent(
      name=f'{request.function.__name__}_test_agent_1',
      delay=0.1,  # Small delay to encourage agent2/3 to start first, though not guaranteed
  )
  agent2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  agent3 = _TestingAgent(name=f'{request.function.__name__}_test_agent_3')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_sequential_agent',
      sub_agents=[agent2, agent3],
  )
  parallel_agent = ParallelAgent(
      name=f'{request.function.__name__}_test_parallel_agent',
      sub_agents=[
          sequential_agent,
          agent1,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, parallel_agent, resumable=is_resumable
  )
  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  # Collect all content events
  content_events = [e for e in events if e.content]
  authors = [e.author for e in content_events]

  # Agent 1, 2, 3 should all produce events
  assert len(content_events) == 3
  assert set(authors) == {agent1.name, agent2.name, agent3.name}

  # For sequential agent, agent2 must come before agent3
  # Find indices
  idx2 = authors.index(agent2.name)
  idx3 = authors.index(agent3.name)
  assert idx2 < idx3

  if is_resumable:
    last_event = events[-1]
    assert last_event.author == parallel_agent.name
    assert last_event.actions.end_of_agent


@pytest.mark.asyncio
async def test_resume_async_branches():
  agent1 = _TestingAgent(
      name='test_resume_async_branches_test_agent_1', delay=0.1
  )
  agent2 = _TestingAgent(name='test_resume_async_branches_test_agent_2')
  agent3 = _TestingAgent(name='test_resume_async_branches_test_agent_3')
  sequential_agent = SequentialAgent(
      name='test_resume_async_branches_test_sequential_agent',
      sub_agents=[agent2, agent3],
  )
  parallel_agent = ParallelAgent(
      name='test_resume_async_branches_test_parallel_agent',
      sub_agents=[
          sequential_agent,
          agent1,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      'test_resume_async_branches', parallel_agent, resumable=True
  )

  # Setup state:
  # ParallelAgent: START done. SequentialAgent RUNNING. Agent1 DONE.
  # SequentialAgent: START done. Agent2 DONE. Agent3 PENDING.

  parent_ctx.agent_states[parallel_agent.name] = WorkflowAgentState(
      nodes={
          START.name: NodeState(status=NodeStatus.COMPLETED),
          agent1.name: NodeState(status=NodeStatus.COMPLETED),
          sequential_agent.name: NodeState(status=NodeStatus.RUNNING),
      }
  )

  parent_ctx.agent_states[f'{parallel_agent.name}/{sequential_agent.name}'] = (
      WorkflowAgentState(
          nodes={
              START.name: NodeState(status=NodeStatus.COMPLETED),
              agent2.name: NodeState(status=NodeStatus.COMPLETED),
              agent3.name: NodeState(status=NodeStatus.PENDING),
          }
      )
  )

  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  # Expect: Agent3 event. Agent1 should NOT emit. Agent2 should NOT emit.
  content_events = [e for e in events if e.content]
  assert len(content_events) == 1
  assert content_events[0].author == agent3.name

  # Verify final success
  assert events[-1].author == parallel_agent.name
  assert events[-1].actions.end_of_agent


@pytest.mark.asyncio
async def test_stop_agent_if_sub_agent_fails():
  # This test is to verify that the parallel agent and subagents will all stop
  # processing and throw exception to top level runner in case of exception.
  agent1 = _TestingAgentWithException(
      name='test_stop_agent_if_sub_agent_fails_test_agent_1'
  )
  agent2 = _TestingAgentInfiniteEvents(
      name='test_stop_agent_if_sub_agent_fails_test_agent_2'
  )
  parallel_agent = ParallelAgent(
      name='test_stop_agent_if_sub_agent_fails_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      'test_stop_agent_if_sub_agent_fails', parallel_agent
  )

  agen = parallel_agent.run_async(parent_ctx)
  # We expect to receive an exception from one of subagents.
  # The exception should be propagated to root agent.
  with pytest.raises(Exception, match='Mock exception'):
    async for _ in agen:
      pass


@pytest.mark.asyncio
async def test_resume_async():
  agent1 = _TestingAgent(name='test_resume_async_test_agent_1')
  agent2 = _TestingAgent(name='test_resume_async_test_agent_2')
  parallel_agent = ParallelAgent(
      name='test_resume_async_test_parallel_agent',
      sub_agents=[
          agent1,
          agent2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      'test_resume_async', parallel_agent, resumable=True
  )

  # Simulate state where agent1 is completed but agent2 is pending
  # Note: In Workflow, 'START' node would also be COMPLETED.
  parent_ctx.agent_states[parallel_agent.name] = WorkflowAgentState(
      nodes={
          START.name: NodeState(status=NodeStatus.COMPLETED),
          agent1.name: NodeState(status=NodeStatus.COMPLETED),
          agent2.name: NodeState(status=NodeStatus.PENDING),
      },
  )

  events = [e async for e in parallel_agent.run_async(parent_ctx)]

  # We expect event from agent2, but NOT from agent1
  content_events = [e for e in events if e.content]
  assert len(content_events) == 1
  assert content_events[0].author == agent2.name

  # Verify final success
  assert events[-1].actions.end_of_agent
