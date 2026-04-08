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

import copy
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm._call_llm_node import CallLlmResult
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import JoinNode
from google.adk.workflow._workflow_class import Workflow
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import find_function_call_event
from .workflow_testing_utils import RequestInputNode
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return None


@pytest.mark.asyncio
async def test_nested_workflow_as_node(request: pytest.FixtureRequest):
  """Tests that a Workflow can be used as a node in another Workflow."""

  async def nested_func(node_input: types.Content):
    return 'I am nested'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_func)],
  )

  async def output_func(node_input: str):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/nested_func',
          {
              'node_name': 'nested_func',
              'output': 'I am nested',
          },
      ),
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_join_node(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow with JoinNode works correctly."""

  async def nested_node_a():
    return {'a': 1}

  async def nested_node_b():
    return {'b': 2}

  async def nested_node_c():
    return {'c': 3}

  nested_join_node = JoinNode(name='nested_join')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', nested_node_a),
          ('START', nested_node_c),
          (nested_node_a, nested_join_node),
          (nested_node_c, nested_node_b),
          (nested_node_b, nested_join_node),
      ],
  )

  async def output_func(node_input: dict):
    return (
        'Joined output:'
        f' a={node_input["nested_node_a"]["a"]},'
        f' b={node_input["nested_node_b"]["b"]}'
    )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert sorted(simplified_events[0:2], key=lambda x: x[1]['node_name']) == [
      (
          'outer_agent/nested_agent/nested_node_a',
          {'node_name': 'nested_node_a', 'output': {'a': 1}},
      ),
      (
          'outer_agent/nested_agent/nested_node_c',
          {'node_name': 'nested_node_c', 'output': {'c': 3}},
      ),
  ]
  assert simplified_events[2:] == [
      (
          'outer_agent/nested_agent/nested_node_b',
          {'node_name': 'nested_node_b', 'output': {'b': 2}},
      ),
      (
          'outer_agent/nested_agent/nested_join',
          {
              'node_name': 'nested_join',
              'output': {'nested_node_a': {'a': 1}, 'nested_node_b': {'b': 2}},
          },
      ),
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'Joined output: a=1, b=2',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_updates_state_outer_reads(
    request: pytest.FixtureRequest,
):
  """Tests that outer workflow can read state updated by nested workflow."""

  async def nested_state_updater(ctx: Context):
    yield Event(
        state={'my_key': 'my_value'},
    )
    yield 'nested agent finished'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_state_updater)],
  )

  def outer_state_reader(my_key: str, node_input: str):
    return f'Nested agent output: {node_input}, state value: {my_key}'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, outer_state_reader)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/nested_state_updater',
          {
              'node_name': 'nested_state_updater',
              'output': 'nested agent finished',
          },
      ),
      (
          'outer_agent/outer_state_reader',
          {
              'node_name': 'outer_state_reader',
              'output': (
                  'Nested agent output: nested agent finished, state value:'
                  ' my_value'
              ),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_intermediate_nodes(
    request: pytest.FixtureRequest,
):
  """Tests that only the final output of a nested workflow is passed to the outer workflow."""

  node_a = TestingNode(name='NodeA', output='Inner Intermediate')
  node_b = TestingNode(name='NodeB', output='Inner Final')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', node_a),
          (node_a, node_b),
      ],
  )

  output_node = InputCapturingNode(name='OutputNode')

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_node),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert output_node.received_inputs == ['Inner Final']

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/NodeA',
          {'node_name': 'NodeA', 'output': 'Inner Intermediate'},
      ),
      (
          'outer_agent/nested_agent/NodeB',
          {'node_name': 'NodeB', 'output': 'Inner Final'},
      ),
      (
          'outer_agent/OutputNode',
          {'node_name': 'OutputNode', 'output': {'received': 'Inner Final'}},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_hitl(request: pytest.FixtureRequest):
  """Tests that a nested Workflow with HITL works correctly."""
  # Given: A nested workflow with an LLM agent that calls a long running tool
  llm_agent = LlmAgent(
      name='llm_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tool'),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func(node_input: Any):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then: It should yield a FunctionCall event and wait for tool execution
  ev = find_function_call_event(events1, 'long_running_tool_func')
  assert ev is not None
  function_call_id = ev.content.parts[0].function_call.id

  simplified_events1 = simplify_events_with_node(events1, use_node_path=True)
  assert simplified_events1 == [
      (
          'outer_agent/nested_agent/llm_agent',
          types.Part.from_function_call(name='long_running_tool_func', args={}),
      ),
  ]

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  simplified_events2 = simplify_events_with_node(events2, use_node_path=True)
  assert simplified_events2 == [
      ('outer_agent/nested_agent/llm_agent', 'LLM response after tool'),
      (
          'outer_agent/output_func',
          {'node_name': 'output_func', 'output': 'I am outer'},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_with_request_input_event_hitl(
    request: pytest.FixtureRequest,
):
  """Nested workflow correctly propagates RequestInput and resumes.

  Setup: outer_agent -> nested_agent -> node_hitl.
  Act:
    - Run 1: start workflow, node_hitl yields RequestInput.
    - Run 2: resume with user response.
  Assert:
    - Run 1: returns RequestInput event.
    - Run 2: node_hitl receives input and completes.
  """

  class NodeHitl(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='node_hitl')

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      if resume_input := ctx.resume_inputs.get('request_input'):
        yield f'Resumed with user input: {resume_input}'
      else:
        yield RequestInput(
            message='requesting input via RequestInputEvent',
            interrupt_id='request_input',
        )

  # Given: A nested workflow where the inner node requests input
  node_hitl_instance = NodeHitl()

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', node_hitl_instance)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then: It should yield a RequestInput event
  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id == 'request_input'

  simplified_events1 = simplify_events_with_node(events1, use_node_path=True)
  assert simplified_events1 == [
      (
          'outer_agent/nested_agent/node_hitl',
          testing_utils.simplify_content(copy.deepcopy(req_events[0].content)),
      ),
  ]

  # When: Resuming with user response
  hitl_response = types.Content(
      parts=[
          create_request_input_response(
              interrupt_id, {'response': 'user input for hitl'}
          )
      ],
      role='user',
  )

  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  # Then: It should complete and pass output to the outer workflow
  simplified_events2 = simplify_events_with_node(events2, use_node_path=True)
  assert simplified_events2 == [
      (
          'outer_agent/nested_agent/node_hitl',
          {
              'node_name': 'node_hitl',
              'output': (
                  "Resumed with user input: {'response': 'user input for hitl'}"
              ),
          },
      ),
      (
          'outer_agent/output_func',
          {'node_name': 'output_func', 'output': 'I am outer'},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_agent_with_request_input_piped_to_next_node(
    request: pytest.FixtureRequest,
):
  """Tests that user response to RequestInput in nested agent is piped to next node."""
  ask_user = RequestInputNode(
      name='ask_user',
      message='Please provide input',
  )
  capture_node = InputCapturingNode(name='capture_node')

  sub_agent = Workflow(
      name='sub_agent',
      edges=[
          ('START', ask_user),
          (ask_user, capture_node),
      ],
  )

  root_agent = Workflow(
      name='root_agent',
      edges=[('START', sub_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id
  hitl_response_payload = {'response': 'user input for hitl'}
  hitl_response = types.Content(
      parts=[
          create_request_input_response(interrupt_id, hitl_response_payload)
      ],
      role='user',
  )

  invocation_id = events1[0].invocation_id
  await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  assert capture_node.received_inputs == [hitl_response_payload]


@pytest.mark.asyncio
async def test_nested_workflow_chain_input_propagation(
    request: pytest.FixtureRequest,
):

  async def create_output_a():
    return 'output of A'

  nested_agent_a = Workflow(
      name='nested_agent_a',
      edges=[('START', create_output_a)],
  )

  capture_node_b = InputCapturingNode(name='capture_node_b')
  nested_agent_b = Workflow(
      name='nested_agent_b',
      edges=[('START', capture_node_b)],
  )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent_a),
          (nested_agent_a, nested_agent_b),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('hello'))

  assert capture_node_b.received_inputs == ['output of A']


@pytest.mark.asyncio
async def test_nested_workflow_with_tool_calls(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow works correctly with two tool calls."""
  tool_call_count = 0

  def simple_tool() -> str:
    nonlocal tool_call_count
    tool_call_count += 1
    return f'Tool output {tool_call_count}'

  # Given: A nested workflow where the inner node calls a tool
  llm_agent = LlmAgent(
      name='llm_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='simple_tool',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tools'),
          ]
      ),
      tools=[simple_tool],
  )

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_func),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  # Then: It should call the tool and yield events
  assert tool_call_count == 1

  simplified_events = simplify_events_with_node(events, use_node_path=True)

  # Extract the dynamically generated function_call_id from events.
  ev = find_function_call_event(events, 'simple_tool')
  assert ev is not None
  function_call_id = ev.content.parts[0].function_call.id

  assert simplified_events == [
      (
          'outer_agent/nested_agent/llm_agent',
          types.Part.from_function_call(name='simple_tool', args={}),
      ),
      (
          'outer_agent/nested_agent/llm_agent',
          types.Part(
              function_response=types.FunctionResponse(
                  name='simple_tool',
                  response={'result': 'Tool output 1'},
              )
          ),
      ),
      ('outer_agent/nested_agent/llm_agent', 'LLM response after tools'),
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_three_level_nested_workflow(request: pytest.FixtureRequest):
  """Tests a 3-level nested Workflow structure."""

  async def inner_func():
    return 'I am inner'

  inner_agent = Workflow(
      name='inner_agent',
      edges=[('START', inner_func)],
  )

  async def middle_func():
    return 'I am middle'

  middle_agent = Workflow(
      name='middle_agent',
      edges=[('START', inner_agent), (inner_agent, middle_func)],
  )

  async def outer_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', middle_agent), (middle_agent, outer_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/middle_agent/inner_agent/inner_func',
          {
              'node_name': 'inner_func',
              'output': 'I am inner',
          },
      ),
      (
          'outer_agent/middle_agent/middle_func',
          {
              'node_name': 'middle_func',
              'output': 'I am middle',
          },
      ),
      (
          'outer_agent/outer_func',
          {
              'node_name': 'outer_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_duplicate_grandchild_workflow_names(
    request: pytest.FixtureRequest,
):
  """Tests that grandchild workflow agents with same name can coexist."""

  # Given: A workflow hierarchy where grandchild workflows have the same name
  async def grandchild_func():
    return 'I am grandchild'

  grandchild_agent = Workflow(
      name='grandchild',
      edges=[('START', grandchild_func)],
  )

  child1_agent = Workflow(
      name='child1',
      edges=[('START', copy.deepcopy(grandchild_agent))],
  )

  child2_agent = Workflow(
      name='child2',
      edges=[('START', copy.deepcopy(grandchild_agent))],
  )

  root_agent = Workflow(
      name='root',
      edges=[('START', child1_agent), (child1_agent, child2_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  # When: Starting the workflow
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  # Then: It should execute both grandchildren successfully
  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'root/child1/grandchild/grandchild_func',
          {'node_name': 'grandchild_func', 'output': 'I am grandchild'},
      ),
      (
          'root/child2/grandchild/grandchild_func',
          {'node_name': 'grandchild_func', 'output': 'I am grandchild'},
      ),
  ]


@pytest.mark.asyncio
async def test_duplicate_name_in_ancestral_path(
    request: pytest.FixtureRequest,
):
  """Tests that agent with same name can exist in ancestral path (A->B->A)."""

  async def func_a():
    return 'I am A'

  agent_a_child = Workflow(
      name='A',
      edges=[('START', func_a)],
  )

  agent_b = Workflow(
      name='B',
      edges=[('START', agent_a_child)],
  )

  agent_a_root = Workflow(
      name='A',
      edges=[('START', agent_b)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=agent_a_root,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'A/B/A/func_a',
          {'node_name': 'func_a', 'output': 'I am A'},
      ),
  ]
