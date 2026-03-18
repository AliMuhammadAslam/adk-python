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

"""Unit tests for call_llm node function in isolation.

Tests call ``call_llm()`` directly with a hand-built WorkflowContext,
without going through the full _SingleLlmAgent + InMemoryRunner pipeline.
This ensures the node's output contract (yielded events and routing
data) is verified independently.

End-to-end pipeline tests are in test_single_llm_agent.py.
"""

from __future__ import annotations

from google.adk.agents.llm._call_llm_node import call_llm
from google.adk.agents.llm._call_llm_node import CallLlmResult
from google.adk.agents.llm._single_llm_agent import _convert_node_input_to_json
from google.adk.agents.llm._single_llm_agent import _SingleLlmAgent
from google.adk.events.event import Event as AdkEvent
from google.adk.events.event import Event as WorkflowEvent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
from pydantic import BaseModel
import pytest

from tests.unittests.workflow import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(mock_model, tools=None, **kwargs):
  """Create a _SingleLlmAgent wired to a MockModel."""
  return _SingleLlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **kwargs,
  )


async def _collect_events(node_fn, ctx):
  """Collect all events yielded by a node function."""
  events = []
  async for event in node_fn(ctx=ctx):
    events.append(event)
  return events


def _route_events(events):
  """Filter to WorkflowEvents that carry a route."""
  return [e for e in events if isinstance(e, WorkflowEvent) and e.actions.route]


def _adk_events(events):
  """Filter to AdkEvents that are not routing WorkflowEvents."""
  return [
      e
      for e in events
      if isinstance(e, AdkEvent)
      and not (isinstance(e, WorkflowEvent) and e.actions.route)
  ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallLlm:
  """Tests for call_llm node function in isolation."""

  @pytest.mark.asyncio
  async def test_text_response_yields_no_route(self):
    """Text-only LLM response yields AdkEvent, no route event."""
    mock_model = testing_utils.MockModel.create(responses=['Hello, world!'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    events = await _collect_events(call_llm, ctx)

    adk = _adk_events(events)
    routes = _route_events(events)
    assert len(adk) >= 1
    assert adk[-1].content.parts[0].text.strip() == 'Hello, world!'
    assert routes == []

  @pytest.mark.asyncio
  async def test_function_call_yields_route_event(self):
    """Function call yields a route event to execute_tools."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[add])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Add 1+2'
    )

    events = await _collect_events(call_llm, ctx)

    routes = _route_events(events)
    assert len(routes) == 1
    assert routes[0].actions.route == 'execute_tools'

  @pytest.mark.asyncio
  async def test_function_call_data_contract(self):
    """Route event data is a CallLlmResult with FunctionCall objects.

    This verifies the inter-node contract: call_llm emits a
    ``CallLlmResult`` containing the function calls.
    """

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[add])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Add 1+2'
    )

    events = await _collect_events(call_llm, ctx)

    routes = _route_events(events)
    assert len(routes) == 1
    data = routes[0].output
    assert isinstance(data, CallLlmResult)
    assert len(data.function_calls) == 1
    assert data.function_calls[0].name == 'add'
    assert data.function_calls[0].args == {'x': 1, 'y': 2}

  @pytest.mark.asyncio
  async def test_long_running_tool_suppresses_function_call_event(self):
    """Long-running tool function_call is suppressed; route event still emitted.

    When the LLM response contains a call to a long-running tool,
    call_llm should NOT yield the finalized AdkEvent (which would
    duplicate the function_call that execute_tools already emits as an
    interrupt event).  The route event with CallLlmResult must still
    be yielded so execute_tools can process it.
    """

    def long_op():
      """A long-running operation."""
      return None

    lr_tool = LongRunningFunctionTool(func=long_op)
    fc = types.Part.from_function_call(name='long_op', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[lr_tool])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Run long op'
    )

    events = await _collect_events(call_llm, ctx)

    # No AdkEvent with function_call content should be yielded.
    adk = _adk_events(events)
    fc_adk = [e for e in adk if e.content and e.get_function_calls()]
    assert (
        fc_adk == []
    ), 'call_llm should suppress function_call AdkEvent for long-running tools'

    # Route event must still be emitted.
    routes = _route_events(events)
    assert len(routes) == 1
    assert routes[0].actions.route == 'execute_tools'
    data = routes[0].output
    assert isinstance(data, CallLlmResult)
    assert len(data.function_calls) == 1
    assert data.function_calls[0].name == 'long_op'

  @pytest.mark.asyncio
  async def test_regular_tool_still_yields_function_call_event(self):
    """Non-long-running tool function_call is NOT suppressed."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[add])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Add 1+2'
    )

    events = await _collect_events(call_llm, ctx)

    # Regular tool should yield the function_call AdkEvent.
    adk = _adk_events(events)
    fc_adk = [e for e in adk if e.content and e.get_function_calls()]
    assert (
        len(fc_adk) == 1
    ), 'call_llm should yield function_call AdkEvent for regular tools'

    # Route event must also be emitted.
    routes = _route_events(events)
    assert len(routes) == 1

  @pytest.mark.asyncio
  async def test_multiple_function_calls_serialized(self):
    """Multiple function calls are all serialized in data."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    fc_add = types.Part.from_function_call(name='add', args={'x': 2, 'y': 3})
    fc_mul = types.Part.from_function_call(
        name='multiply', args={'x': 4, 'y': 5}
    )
    mock_model = testing_utils.MockModel.create(responses=[[fc_add, fc_mul]])
    agent = _make_agent(mock_model, tools=[add, multiply])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Compute'
    )

    events = await _collect_events(call_llm, ctx)

    routes = _route_events(events)
    assert len(routes) == 1
    data = routes[0].output
    assert isinstance(data, CallLlmResult)
    assert len(data.function_calls) == 2
    names = {fc.name for fc in data.function_calls}
    assert names == {'add', 'multiply'}

  @pytest.mark.asyncio
  async def test_before_model_callback_short_circuits(self):
    """before_model_callback intercepts; no route, model not called."""

    def before_cb(callback_context, llm_request):
      return LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Intercepted!')],
          )
      )

    mock_model = testing_utils.MockModel.create(
        responses=['Should not reach here.']
    )
    agent = _make_agent(mock_model, before_model_callback=before_cb)
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='test'
    )

    events = await _collect_events(call_llm, ctx)

    assert len(mock_model.requests) == 0
    routes = _route_events(events)
    assert routes == []

  @pytest.mark.asyncio
  async def test_after_model_callback_modifies_response(self):
    """after_model_callback modifies the yielded response."""

    def after_cb(callback_context, llm_response):
      return LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Modified!')],
          )
      )

    mock_model = testing_utils.MockModel.create(responses=['Original.'])
    agent = _make_agent(mock_model, after_model_callback=after_cb)
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='test'
    )

    events = await _collect_events(call_llm, ctx)

    adk = _adk_events(events)
    assert len(adk) >= 1
    assert adk[-1].content.parts[0].text.strip() == 'Modified!'

  @pytest.mark.asyncio
  async def test_agent_name_label(self):
    """call_llm sets adk_agent_name label on the LLM request."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='test'
    )

    await _collect_events(call_llm, ctx)

    assert len(mock_model.requests) == 1
    labels = mock_model.requests[0].config.labels
    assert labels.get('adk_agent_name') == 'test_agent'


class TestConvertNodeInputToJson:
  """Tests for _convert_node_input_to_json conversion utility."""

  def test_string_returned_as_is(self):
    assert _convert_node_input_to_json('hello') == 'hello'

  def test_int(self):
    assert _convert_node_input_to_json(42) == '42'

  def test_float(self):
    assert _convert_node_input_to_json(3.14) == '3.14'

  def test_bool_true(self):
    assert _convert_node_input_to_json(True) == 'true'

  def test_bool_false(self):
    assert _convert_node_input_to_json(False) == 'false'

  def test_dict(self):
    result = _convert_node_input_to_json({'key': 'value'})
    assert result == '{"key": "value"}'

  def test_list(self):
    result = _convert_node_input_to_json([1, 2, 3])
    assert result == '[1, 2, 3]'

  def test_pydantic_model(self):

    class MyModel(BaseModel):
      name: str
      count: int

    model = MyModel(name='test', count=5)
    result = _convert_node_input_to_json(model)
    assert '"name":"test"' in result or '"name": "test"' in result
    assert '"count":5' in result or '"count": 5' in result

  def test_unsupported_type_raises_type_error(self):
    with pytest.raises(TypeError):
      _convert_node_input_to_json(object())


class TestCallLlmNodeInput:
  """Tests for node_input appended to session in _SingleLlmAgent.run()."""

  @pytest.mark.asyncio
  async def test_node_input_appended_to_session(self):
    """node_input is appended to session events as user content."""
    mock_model = testing_utils.MockModel.create(responses=['Got it.'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    async for _ in agent.run(ctx=ctx, node_input={'data': 'workflow_value'}):
      pass

    request = mock_model.requests[0]
    user_contents = [c for c in request.contents if c.role == 'user']
    texts = [p.text for c in user_contents for p in c.parts if p.text]
    assert any('workflow_value' in t for t in texts)

  @pytest.mark.asyncio
  async def test_node_input_none_not_appended(self):
    """When node_input is None, no extra event is added."""
    mock_model = testing_utils.MockModel.create(responses=['Got it.'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')
    events_before = len(ctx._invocation_context.session.events)

    async for _ in agent.run(ctx=ctx, node_input=None):
      pass

    # No extra user event was added to the session
    user_events = [
        e
        for e in ctx._invocation_context.session.events[: events_before + 1]
        if e.author == 'user'
    ]
    assert len(user_events) == 1

  @pytest.mark.asyncio
  async def test_node_input_string_appended_directly(self):
    """String node_input is appended as-is, not JSON-encoded."""
    mock_model = testing_utils.MockModel.create(responses=['Got it.'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    async for _ in agent.run(ctx=ctx, node_input='plain text input'):
      pass

    request = mock_model.requests[0]
    user_contents = [c for c in request.contents if c.role == 'user']
    texts = [p.text for c in user_contents for p in c.parts if p.text]
    assert 'plain text input' in texts

  @pytest.mark.asyncio
  async def test_node_input_content_appended_directly(self):
    """types.Content node_input is appended directly without conversion."""
    mock_model = testing_utils.MockModel.create(responses=['Got it.'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    content_input = types.Content(
        role='user',
        parts=[types.Part.from_text(text='direct content')],
    )
    async for _ in agent.run(ctx=ctx, node_input=content_input):
      pass

    request = mock_model.requests[0]
    user_contents = [c for c in request.contents if c.role == 'user']
    texts = [p.text for c in user_contents for p in c.parts if p.text]
    assert 'direct content' in texts
