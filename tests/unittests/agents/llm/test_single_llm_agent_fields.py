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

"""Unit tests for canonical_* fields in _SingleLlmAgent."""

from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm import _base_llm_agent
from google.adk.agents.llm._single_llm_agent import _SingleLlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.google_llm import Gemini
from google.adk.models.registry import LLMRegistry
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
import pytest

# BaseLlmAgent uses TYPE_CHECKING for ReadonlyContext and CallbackContext,
# so Pydantic needs model_rebuild() to resolve the forward references.
_ns = {
    **vars(_base_llm_agent),
    'ReadonlyContext': ReadonlyContext,
    'CallbackContext': CallbackContext,
}
_SingleLlmAgent.model_rebuild(_types_namespace=_ns)


def _noop():
  pass


_NOOP_NODE = FunctionNode(_noop, name='_noop')
_DUMMY_EDGES = [(START, _NOOP_NODE)]


def _make_agent(**kwargs) -> _SingleLlmAgent:
  """Creates a _SingleLlmAgent with dummy edges for testing."""
  kwargs.setdefault('edges', _DUMMY_EDGES)
  return _SingleLlmAgent(**kwargs)


async def _create_readonly_context(
    agent: _SingleLlmAgent, state: Optional[dict[str, Any]] = None
) -> ReadonlyContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ReadonlyContext(invocation_context)


# ------------------------------------------------------------------
# canonical_model
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ('default_model', 'expected_model_name', 'expected_model_type'),
    [
        (_SingleLlmAgent.DEFAULT_MODEL, _SingleLlmAgent.DEFAULT_MODEL, Gemini),
        ('gemini-2.0-flash', 'gemini-2.0-flash', Gemini),
    ],
)
def test_canonical_model_default_fallback(
    default_model, expected_model_name, expected_model_type
):
  original_default = _SingleLlmAgent._default_model
  _SingleLlmAgent.set_default_model(default_model)
  try:
    agent = _make_agent(name='test_agent')
    assert isinstance(agent.canonical_model, expected_model_type)
    assert agent.canonical_model.model == expected_model_name
  finally:
    _SingleLlmAgent.set_default_model(original_default)


def test_canonical_model_str():
  agent = _make_agent(name='test_agent', model='gemini-pro')

  assert agent.canonical_model.model == 'gemini-pro'


def test_canonical_model_llm():
  llm = LLMRegistry.new_llm('gemini-pro')
  agent = _make_agent(name='test_agent', model=llm)

  assert agent.canonical_model == llm


def test_canonical_model_inherit():
  # Workflow doesn't support sub_agents, so manually wire parent_agent.
  parent_agent = _make_agent(name='parent_agent', model='gemini-pro')
  sub_agent = _make_agent(name='sub_agent')
  sub_agent.parent_agent = parent_agent

  assert sub_agent.canonical_model == parent_agent.canonical_model


def test_canonical_model_inherit_deep_chain():
  # Workflow doesn't support sub_agents, so manually wire parent_agent.
  root = _make_agent(name='root', model='gemini-pro')
  child = _make_agent(name='child')
  child.parent_agent = root
  grandchild = _make_agent(name='grandchild')
  grandchild.parent_agent = child

  assert grandchild.canonical_model.model == 'gemini-pro'
  assert grandchild.canonical_model == root.canonical_model


def test_set_default_model_invalid_type():
  with pytest.raises(TypeError, match='must be a model name or BaseLlm'):
    _SingleLlmAgent.set_default_model(123)


def test_set_default_model_empty_string():
  with pytest.raises(ValueError, match='must be a non-empty string'):
    _SingleLlmAgent.set_default_model('')


# ------------------------------------------------------------------
# canonical_instruction
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canonical_instruction_str():
  agent = _make_agent(name='test_agent', instruction='instruction')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction'
  assert not bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_instruction_callable():
  def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = _make_agent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_instruction_async_callable():
  async def _instruction_provider(ctx: ReadonlyContext) -> str:
    return f'instruction: {ctx.state["state_var"]}'

  agent = _make_agent(name='test_agent', instruction=_instruction_provider)
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == 'instruction: state_value'
  assert bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_instruction_empty_str():
  agent = _make_agent(name='test_agent')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_instruction(ctx)
  )
  assert canonical_instruction == ''
  assert not bypass_state_injection


# ------------------------------------------------------------------
# canonical_global_instruction
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canonical_global_instruction_str():
  agent = _make_agent(
      name='test_agent', global_instruction='global instruction'
  )
  ctx = await _create_readonly_context(agent)

  with pytest.warns(DeprecationWarning, match='global_instruction'):
    canonical_instruction, bypass_state_injection = (
        await agent.canonical_global_instruction(ctx)
    )
  assert canonical_instruction == 'global instruction'
  assert not bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_global_instruction_callable():
  def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = _make_agent(
      name='test_agent',
      global_instruction=_global_instruction_provider,
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  with pytest.warns(DeprecationWarning, match='global_instruction'):
    canonical_global_instruction, bypass_state_injection = (
        await agent.canonical_global_instruction(ctx)
    )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_global_instruction_async_callable():
  async def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    return f'global instruction: {ctx.state["state_var"]}'

  agent = _make_agent(
      name='test_agent',
      global_instruction=_global_instruction_provider,
  )
  ctx = await _create_readonly_context(
      agent, state={'state_var': 'state_value'}
  )

  with pytest.warns(DeprecationWarning, match='global_instruction'):
    canonical_global_instruction, bypass_state_injection = (
        await agent.canonical_global_instruction(ctx)
    )
  assert canonical_global_instruction == 'global instruction: state_value'
  assert bypass_state_injection


@pytest.mark.asyncio
async def test_canonical_global_instruction_empty_no_warning():
  agent = _make_agent(name='test_agent')
  ctx = await _create_readonly_context(agent)

  canonical_instruction, bypass_state_injection = (
      await agent.canonical_global_instruction(ctx)
  )
  assert canonical_instruction == ''
  assert not bypass_state_injection


# ------------------------------------------------------------------
# canonical_tools
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canonical_tools_function():
  def _my_tool(sides: int) -> int:
    return sides

  agent = _make_agent(name='test_agent', model='gemini-pro', tools=[_my_tool])
  ctx = await _create_readonly_context(agent)
  tools = await agent.canonical_tools(ctx)

  assert len(tools) == 1
  assert tools[0].name == '_my_tool'
  assert tools[0].__class__.__name__ == 'FunctionTool'


@pytest.mark.asyncio
async def test_canonical_tools_empty():
  agent = _make_agent(name='test_agent', model='gemini-pro')
  ctx = await _create_readonly_context(agent)
  tools = await agent.canonical_tools(ctx)

  assert tools == []


@pytest.mark.asyncio
async def test_canonical_tools_multiple():
  def _tool_a() -> str:
    return 'a'

  def _tool_b() -> str:
    return 'b'

  agent = _make_agent(
      name='test_agent',
      model='gemini-pro',
      tools=[_tool_a, _tool_b],
  )
  ctx = await _create_readonly_context(agent)
  tools = await agent.canonical_tools(ctx)

  assert len(tools) == 2
  assert tools[0].name == '_tool_a'
  assert tools[1].name == '_tool_b'


# ------------------------------------------------------------------
# canonical_before_model_callbacks
# ------------------------------------------------------------------


def test_canonical_before_model_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_before_model_callbacks == []


def test_canonical_before_model_callbacks_single():
  def _cb(callback_context, llm_request):
    return None

  agent = _make_agent(name='test_agent', before_model_callback=_cb)
  assert agent.canonical_before_model_callbacks == [_cb]


def test_canonical_before_model_callbacks_list():
  def _cb1(callback_context, llm_request):
    return None

  def _cb2(callback_context, llm_request):
    return None

  agent = _make_agent(name='test_agent', before_model_callback=[_cb1, _cb2])
  assert agent.canonical_before_model_callbacks == [_cb1, _cb2]


# ------------------------------------------------------------------
# canonical_after_model_callbacks
# ------------------------------------------------------------------


def test_canonical_after_model_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_after_model_callbacks == []


def test_canonical_after_model_callbacks_single():
  def _cb(callback_context, llm_response):
    return None

  agent = _make_agent(name='test_agent', after_model_callback=_cb)
  assert agent.canonical_after_model_callbacks == [_cb]


def test_canonical_after_model_callbacks_list():
  def _cb1(callback_context, llm_response):
    return None

  def _cb2(callback_context, llm_response):
    return None

  agent = _make_agent(name='test_agent', after_model_callback=[_cb1, _cb2])
  assert agent.canonical_after_model_callbacks == [_cb1, _cb2]


# ------------------------------------------------------------------
# canonical_on_model_error_callbacks
# ------------------------------------------------------------------


def test_canonical_on_model_error_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_on_model_error_callbacks == []


def test_canonical_on_model_error_callbacks_single():
  def _cb(callback_context, llm_request, error):
    return None

  agent = _make_agent(name='test_agent', on_model_error_callback=_cb)
  assert agent.canonical_on_model_error_callbacks == [_cb]


def test_canonical_on_model_error_callbacks_list():
  def _cb1(callback_context, llm_request, error):
    return None

  def _cb2(callback_context, llm_request, error):
    return None

  agent = _make_agent(name='test_agent', on_model_error_callback=[_cb1, _cb2])
  assert agent.canonical_on_model_error_callbacks == [_cb1, _cb2]


# ------------------------------------------------------------------
# canonical_before_tool_callbacks
# ------------------------------------------------------------------


def test_canonical_before_tool_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_before_tool_callbacks == []


def test_canonical_before_tool_callbacks_single():
  def _cb(tool, args, tool_context):
    return None

  agent = _make_agent(name='test_agent', before_tool_callback=_cb)
  assert agent.canonical_before_tool_callbacks == [_cb]


def test_canonical_before_tool_callbacks_list():
  def _cb1(tool, args, tool_context):
    return None

  def _cb2(tool, args, tool_context):
    return None

  agent = _make_agent(name='test_agent', before_tool_callback=[_cb1, _cb2])
  assert agent.canonical_before_tool_callbacks == [_cb1, _cb2]


# ------------------------------------------------------------------
# canonical_after_tool_callbacks
# ------------------------------------------------------------------


def test_canonical_after_tool_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_after_tool_callbacks == []


def test_canonical_after_tool_callbacks_single():
  def _cb(tool, args, tool_context, result):
    return None

  agent = _make_agent(name='test_agent', after_tool_callback=_cb)
  assert agent.canonical_after_tool_callbacks == [_cb]


def test_canonical_after_tool_callbacks_list():
  def _cb1(tool, args, tool_context, result):
    return None

  def _cb2(tool, args, tool_context, result):
    return None

  agent = _make_agent(name='test_agent', after_tool_callback=[_cb1, _cb2])
  assert agent.canonical_after_tool_callbacks == [_cb1, _cb2]


# ------------------------------------------------------------------
# canonical_on_tool_error_callbacks
# ------------------------------------------------------------------


def test_canonical_on_tool_error_callbacks_none():
  agent = _make_agent(name='test_agent')
  assert agent.canonical_on_tool_error_callbacks == []


def test_canonical_on_tool_error_callbacks_single():
  def _cb(tool, args, tool_context, error):
    return None

  agent = _make_agent(name='test_agent', on_tool_error_callback=_cb)
  assert agent.canonical_on_tool_error_callbacks == [_cb]


def test_canonical_on_tool_error_callbacks_list():
  def _cb1(tool, args, tool_context, error):
    return None

  def _cb2(tool, args, tool_context, error):
    return None

  agent = _make_agent(name='test_agent', on_tool_error_callback=[_cb1, _cb2])
  assert agent.canonical_on_tool_error_callbacks == [_cb1, _cb2]
