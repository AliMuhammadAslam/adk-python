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

"""Testings for the Workflow with agent nodes."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.events.event import Event as AdkEvent
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import simplify_events_with_node


class SimpleAgent(BaseAgent):
  """A simple agent for testing."""

  message: str = ''

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[AdkEvent, None]:
    """Yields a single event with a message."""
    yield AdkEvent(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text=self.message)]),
    )


@pytest.mark.asyncio
async def test_run_async_with_agent_nodes(request: pytest.FixtureRequest):
  """Tests running a workflow with BaseAgent instances as nodes."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  agent_b = SimpleAgent(name='AgentB', message='World')
  agent = Workflow(
      name='wf_with_agents',
      edges=[
          (START, agent_a),
          (agent_a, agent_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('AgentA', 'Hello'),
      ('AgentB', 'World'),
  ]


@pytest.mark.asyncio
async def test_run_async_with_agent_node_piping_data(
    request: pytest.FixtureRequest,
):
  """Tests that Event data from an agent node is piped to the next node."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='wf_with_agent_piping',
      edges=[
          (START, agent_a),
          (agent_a, node_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  _ = [e async for e in agent.run_async(ctx)]

  # AgentNode does not record content as node output, so the next node
  # receives None as input.
  assert node_b.received_inputs == [None]
