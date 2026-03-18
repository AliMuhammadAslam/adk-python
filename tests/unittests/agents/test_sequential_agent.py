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

"""Testings for the SequentialAgent."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import START
from google.adk.workflow._execution_state import NodeStatus
from google.adk.workflow._workflow import NodeState
from google.adk.workflow._workflow import WorkflowAgentState
from google.genai import types
import pytest
from typing_extensions import override

from ..workflow import testing_utils


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
async def test_run_async(request: pytest.FixtureRequest):
  agent_1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent_2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_agent',
      sub_agents=[
          agent_1,
          agent_2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, sequential_agent
  )
  events = [e async for e in sequential_agent.run_async(parent_ctx)]

  assert len(events) == 2
  assert events[0].author == agent_1.name
  assert events[1].author == agent_2.name
  assert events[0].content.parts[0].text == f'Hello, async {agent_1.name}!'
  assert events[1].content.parts[0].text == f'Hello, async {agent_2.name}!'


@pytest.mark.asyncio
async def test_run_async_with_resumability(request: pytest.FixtureRequest):
  agent_1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent_2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_agent',
      sub_agents=[
          agent_1,
          agent_2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, sequential_agent, resumable=True
  )
  events = [e async for e in sequential_agent.run_async(parent_ctx)]
  simplified_events = testing_utils.simplify_resumable_app_events(events)

  assert simplified_events == [
      (
          sequential_agent.name,
          {
              'node_states': {
                  agent_1.name: NodeStatus.RUNNING.value,
              }
          },
      ),
      (agent_1.name, f'Hello, async {agent_1.name}!'),
      (
          sequential_agent.name,
          {
              'node_states': {
                  agent_1.name: NodeStatus.COMPLETED.value,
                  agent_2.name: NodeStatus.RUNNING.value,
              }
          },
      ),
      (agent_2.name, f'Hello, async {agent_2.name}!'),
      (
          sequential_agent.name,
          {
              'node_states': {
                  agent_1.name: NodeStatus.COMPLETED.value,
                  agent_2.name: NodeStatus.COMPLETED.value,
              }
          },
      ),
      (sequential_agent.name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_resume_async(request: pytest.FixtureRequest):
  agent_1 = _TestingAgent(name=f'{request.function.__name__}_test_agent_1')
  agent_2 = _TestingAgent(name=f'{request.function.__name__}_test_agent_2')
  sequential_agent = SequentialAgent(
      name=f'{request.function.__name__}_test_agent',
      sub_agents=[
          agent_1,
          agent_2,
      ],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, sequential_agent, resumable=True
  )
  parent_ctx.agent_states[sequential_agent.name] = WorkflowAgentState(
      nodes={
          START.name: NodeState(status=NodeStatus.COMPLETED),
          agent_1.name: NodeState(status=NodeStatus.COMPLETED),
          agent_2.name: NodeState(status=NodeStatus.PENDING),
      }
  )

  events = [e async for e in sequential_agent.run_async(parent_ctx)]

  # 4 events:
  # 1. SequentialAgent checkpoint event for agent 2 running
  # 2. Agent 2 event
  # 3. SequentialAgent checkpoint event for agent 2 completion
  # 4. SequentialAgent final checkpoint event
  assert len(events) == 4
  assert events[0].author == sequential_agent.name
  nodes_0 = events[0].actions.agent_state['nodes']
  assert nodes_0[START.name]['status'] == NodeStatus.COMPLETED.value
  assert nodes_0[agent_1.name]['status'] == NodeStatus.COMPLETED.value
  assert nodes_0[agent_2.name]['status'] == NodeStatus.RUNNING.value

  assert events[1].author == agent_2.name
  assert events[1].content.parts[0].text == f'Hello, async {agent_2.name}!'

  assert events[2].author == sequential_agent.name
  assert not events[2].actions.end_of_agent
  nodes_2 = events[2].actions.agent_state['nodes']
  assert nodes_2[START.name]['status'] == NodeStatus.COMPLETED.value
  assert nodes_2[agent_1.name]['status'] == NodeStatus.COMPLETED.value
  assert nodes_2[agent_2.name]['status'] == NodeStatus.COMPLETED.value

  assert events[3].author == sequential_agent.name
  assert events[3].actions.end_of_agent
