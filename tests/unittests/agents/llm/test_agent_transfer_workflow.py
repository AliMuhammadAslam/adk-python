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

"""Unit tests for workflow agents/agent_transfer.py."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm._agent_transfer import inject_transfer_tools
from google.adk.agents.llm._transfer_target_info import _TransferTargetInfo
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


async def _create_invocation_context() -> InvocationContext:
  agent = LlmAgent(name='test_agent')
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )


@pytest.mark.asyncio
async def test_inject_transfer_tools_empty_targets():
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  await inject_transfer_tools(inv_ctx, llm_request, [])

  # No instructions or tools should be added.
  assert not llm_request.config.system_instruction
  assert not llm_request.config.tools


@pytest.mark.asyncio
async def test_inject_transfer_tools_single_target():
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  targets = [
      _TransferTargetInfo(name='agent_a', description='Handles topic A'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  # Should have transfer instructions in system_instruction.
  si = llm_request.config.system_instruction
  assert 'agent_a' in si
  assert 'Handles topic A' in si
  assert 'transfer_to_agent' in si

  # Should have the transfer_to_agent tool declaration.
  assert llm_request.config.tools
  tool = llm_request.config.tools[0]
  decls = tool.function_declarations
  assert any(d.name == 'transfer_to_agent' for d in decls)

  # Should have the tool in tools_dict.
  assert 'transfer_to_agent' in llm_request.tools_dict


@pytest.mark.asyncio
async def test_inject_transfer_tools_multiple_targets():
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  targets = [
      _TransferTargetInfo(name='agent_b', description='Handles topic B'),
      _TransferTargetInfo(name='agent_a', description='Handles topic A'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  si = llm_request.config.system_instruction
  # Both agents should appear in instructions.
  assert 'agent_a' in si
  assert 'agent_b' in si
  assert 'Handles topic A' in si
  assert 'Handles topic B' in si

  # Agent names in the NOTE should be sorted alphabetically.
  assert '`agent_a`, `agent_b`' in si


@pytest.mark.asyncio
async def test_inject_transfer_tools_enum_constraint():
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  targets = [
      _TransferTargetInfo(name='agent_x'),
      _TransferTargetInfo(name='agent_y'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  # The transfer_to_agent tool should have enum constraints on agent_name.
  tool = llm_request.config.tools[0]
  decl = next(
      d for d in tool.function_declarations if d.name == 'transfer_to_agent'
  )
  agent_name_param = decl.parameters.properties['agent_name']
  assert set(agent_name_param.enum) == {'agent_x', 'agent_y'}


@pytest.mark.asyncio
async def test_inject_transfer_tools_preserves_existing_instructions():
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()
  llm_request.config.system_instruction = 'You are a helpful assistant.'

  targets = [
      _TransferTargetInfo(name='agent_a', description='Handles topic A'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  si = llm_request.config.system_instruction
  # Original instruction should be preserved.
  assert si.startswith('You are a helpful assistant.')
  # Transfer instructions should be appended.
  assert 'agent_a' in si


@pytest.mark.asyncio
async def test_inject_transfer_tools_default_agent_single():
  """With a single target, it should be the default fallback agent."""
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  targets = [
      _TransferTargetInfo(name='agent_a', description='Handles topic A'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  si = llm_request.config.system_instruction
  # Should NOT contain the parent-agent-specific suffix.
  assert 'parent agent' not in si
  # Should contain a default-agent fallback using the first target.
  assert 'transfer to `agent_a`' in si


@pytest.mark.asyncio
async def test_inject_transfer_tools_default_agent_multiple():
  """With multiple targets, the first target is the default fallback."""
  inv_ctx = await _create_invocation_context()
  llm_request = LlmRequest()

  targets = [
      _TransferTargetInfo(name='agent_b', description='Handles topic B'),
      _TransferTargetInfo(name='agent_a', description='Handles topic A'),
  ]
  await inject_transfer_tools(inv_ctx, llm_request, targets)

  si = llm_request.config.system_instruction
  # Default fallback should use the first target (agent_b), not sorted.
  assert 'transfer to `agent_b`' in si
