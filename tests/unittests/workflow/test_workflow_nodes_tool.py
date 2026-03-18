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

"""Testings for the Workflow with tools."""

from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._workflow_graph import WorkflowGraph
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node


def _func_a() -> dict[str, str]:
  """Returns a value from function A."""
  return {'val': 'Hello'}


def _func_b() -> str:
  """Returns a value from function B."""
  return 'world'


@pytest.mark.asyncio
async def test_run_async_with_function_tools(request: pytest.FixtureRequest):
  tool_a = FunctionTool(_func_a)
  tool_b = FunctionTool(_func_b)
  agent = Workflow(
      name='wf_with_tools',
      edges=[
          (START, tool_a),
          (tool_a, tool_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('wf_with_tools', {'node_name': '_func_a', 'output': {'val': 'Hello'}}),
      ('wf_with_tools', {'node_name': '_func_b', 'output': 'world'}),
  ]
