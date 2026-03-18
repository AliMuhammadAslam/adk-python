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

"""Agent transfer injection for workflow agents."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from ...tools.tool_context import ToolContext
from ...tools.transfer_to_agent_tool import TransferToAgentTool

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ..invocation_context import InvocationContext
  from ._transfer_target_info import _TransferTargetInfo


def _build_target_agents_info(target_agent: Any) -> str:
  # TODO: Refactor the annotation of the parameters
  return f"""
Agent name: {target_agent.name}
Agent description: {target_agent.description}
"""


line_break = '\n'


def _build_transfer_instruction_body(
    tool_name: str,
    target_agents: list[Any],
) -> str:
  """Build the core transfer instruction text.

  TODO: Refactor the annotation of the parameters

  This is the agent-tree-agnostic portion of transfer instructions. It
  works with any objects having ``.name`` and ``.description`` attributes

  Args:
    tool_name: The name of the transfer tool (e.g. 'transfer_to_agent').
    target_agents: Objects with ``.name`` and ``.description``.

  Returns:
    Instruction text for the LLM about agent transfers.
  """
  available_agent_names = [t.name for t in target_agents]
  available_agent_names.sort()
  formatted_agent_names = ', '.join(
      f'`{name}`' for name in available_agent_names
  )

  return f"""
You have a list of other agents to transfer to:

{line_break.join([
    _build_target_agents_info(target_agent) for target_agent in target_agents
])}

If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `{tool_name}` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `{tool_name}` function are
{formatted_agent_names}.
"""


async def inject_transfer_tools(
    invocation_context: 'InvocationContext',
    llm_request: 'LlmRequest',
    transfer_targets: list['_TransferTargetInfo'],
) -> None:
  """Inject transfer-to-agent tool and instructions into an LLM request.

  This is the workflow-agent counterpart of
  ``_AgentTransferLlmRequestProcessor``. Instead of discovering targets
  from the agent tree, it receives pre-computed ``_TransferTargetInfo``
  objects (built by ``_Mesh._build_workflow_context``).

  The first element of ``transfer_targets`` is treated as the default
  fallback agent: when the LLM cannot handle the request and is unsure
  which agent to transfer to, it is instructed to transfer to this
  default agent.

  Args:
    invocation_context: The current invocation context.
    llm_request: The LLM request to augment with transfer tool and
      instructions.
    transfer_targets: Pre-computed transfer target metadata. Each object
      has ``.name`` and ``.description`` attributes. The first element
      is used as the default fallback agent.
  """
  if not transfer_targets:
    return

  transfer_to_agent_tool = TransferToAgentTool(
      agent_names=[t.name for t in transfer_targets]
  )

  si = _build_transfer_instruction_body(
      transfer_to_agent_tool.name,
      transfer_targets,
  )
  si += (
      '\nIf you cannot handle the request and are unsure which agent'
      f' to transfer to, transfer to `{transfer_targets[0].name}`.\n'
  )
  llm_request.append_instructions([si])

  tool_context = ToolContext(invocation_context)
  await transfer_to_agent_tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )
