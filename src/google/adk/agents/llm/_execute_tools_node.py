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

"""execute_tools: workflow node function that executes function calls."""

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.genai import types

from . import _output_schema_processor
from ...auth.auth_preprocessor import _store_auth_and_collect_resume_targets
from ...auth.auth_tool import AuthConfig
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.base_tool import BaseTool
from ..context import Context
from ._call_llm_node import CallLlmResult
from ._functions import generate_auth_event
from ._functions import generate_request_confirmation_event
from ._functions import get_long_running_function_calls
from ._functions import handle_function_call_list_async
from ._reasoning import _process_agent_tools
from ._request_confirmation import _parse_tool_confirmation
from ._request_confirmation import _resolve_confirmation_targets

logger = logging.getLogger('google_adk.' + __name__)


async def _process_auth_resume(
    ctx: Context,
    invocation_context: Any,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> Optional[Event]:
  """Handle auth resume when execute_tools is re-run after an interrupt.

  Delegates credential storage and target resolution to the shared
  ``_store_auth_and_collect_resume_targets`` helper from
  ``auth_preprocessor``, then re-executes only the tools that needed
  auth.

  Returns:
    The function response event, or None if the resume inputs did not
    correspond to auth responses.
  """
  # Pre-filter: try to validate each resume input as AuthConfig.
  # Non-auth inputs (e.g. confirmation responses) will fail validation
  # and are skipped.
  auth_responses: dict[str, Any] = {}
  for fc_id, response in ctx.resume_inputs.items():
    try:
      AuthConfig.model_validate(response)
      auth_responses[fc_id] = response
    except Exception:
      continue

  if not auth_responses:
    return None

  events = invocation_context.session.events
  # _store_auth_and_collect_resume_targets internally matches IDs
  # against adk_request_credential function calls in events, so
  # any false positives from validation are filtered out there.
  tools_to_resume = await _store_auth_and_collect_resume_targets(
      events,
      set(auth_responses.keys()),
      auth_responses,
      invocation_context.session.state,
  )
  if not tools_to_resume:
    return None

  return await handle_function_call_list_async(
      invocation_context,
      function_calls,
      tools_dict,
      filters=tools_to_resume,
  )


async def _process_confirmation_resume(
    ctx: Context,
    invocation_context: Any,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> Optional[Event]:
  """Handle confirmation resume when execute_tools is re-run.

  Parses ``ToolConfirmation`` from each resume input using the shared
  ``_parse_tool_confirmation`` helper, resolves original function calls
  via ``_resolve_confirmation_targets``, then re-executes with the
  confirmation dict.

  Returns:
    The function response event, or None if the resume inputs did not
    correspond to confirmation responses.
  """
  from ...tools.tool_confirmation import ToolConfirmation

  events = invocation_context.session.events
  confirmation_fc_ids = set(ctx.resume_inputs.keys())

  # Parse ToolConfirmation from each resume input.
  confirmations_by_fc_id: dict[str, ToolConfirmation] = {}
  for fc_id, response in ctx.resume_inputs.items():
    try:
      confirmations_by_fc_id[fc_id] = _parse_tool_confirmation(response)
    except Exception:
      continue

  if not confirmations_by_fc_id:
    return None

  # Resolve original function calls.
  tool_confirmation_dict, original_fcs_dict = _resolve_confirmation_targets(
      events, confirmation_fc_ids, confirmations_by_fc_id
  )

  if not tool_confirmation_dict:
    return None

  return await handle_function_call_list_async(
      invocation_context,
      list(original_fcs_dict.values()),
      tools_dict,
      filters=set(tool_confirmation_dict.keys()),
      tool_confirmation_dict=tool_confirmation_dict,
  )


async def _process_long_running_resume(
    ctx: Context,
    invocation_context: Any,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> Optional[Event]:
  """Handle long-running tool resume when execute_tools is re-run.

  Builds function_response events from resume_inputs keyed by
  long-running tool function_call IDs.

  Returns:
    The function response event, or None if the resume inputs did not
    correspond to long-running tool responses.
  """
  long_running_tool_ids = get_long_running_function_calls(
      function_calls, tools_dict
  )
  long_running_responses = {
      fc_id: response
      for fc_id, response in ctx.resume_inputs.items()
      if fc_id in long_running_tool_ids
  }
  if not long_running_responses:
    return None

  parts = []
  for fc_id, response in long_running_responses.items():
    original_fc = next((fc for fc in function_calls if fc.id == fc_id), None)
    if original_fc:
      parts.append(
          types.Part(
              function_response=types.FunctionResponse(
                  name=original_fc.name,
                  id=fc_id,
                  response=response,
              )
          )
      )

  if not parts:
    return None

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      content=types.Content(parts=parts, role='user'),
  )


def _long_running_interrupt_event(
    invocation_context: Any,
    function_calls: list[types.FunctionCall],
    pending_ids: set[str],
) -> Event:
  """Creates an interrupt event for pending long-running tool calls."""
  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      content=types.Content(
          role='model',
          parts=[
              types.Part(function_call=fc)
              for fc in function_calls
              if fc.id in pending_ids
          ],
      ),
      long_running_tool_ids=pending_ids,
  )


CONTINUE_ROUTE = 'continue'
"""Route label that signals the reason-act loop should continue."""


def _continue_event() -> Event:
  """Creates a route event to continue the call_llm ↔ execute_tools loop."""
  event = Event()
  event.actions.route = CONTINUE_ROUTE
  return event


async def execute_tools(
    ctx: Context, node_input: CallLlmResult
) -> AsyncGenerator[Any, None]:
  """Workflow node that executes function calls from an LLM response.

  Receives a ``CallLlmResult`` containing ``FunctionCall`` objects
  and reuses the ``tools_dict`` resolved by ``call_llm`` (passed via
  session temp state). Falls back to rebuilding ``tools_dict`` on
  resume after checkpoint. Executes the tools and yields the function
  response event.

  If the function response contains a ``transfer_to_agent`` action,
  the workflow terminates (no route back). Otherwise the ``'continue'``
  route triggers ``call_llm`` for the next reason-act iteration.
  """
  invocation_context = ctx.get_invocation_context()

  function_calls = list(node_input.function_calls)

  # Reuse tools_dict from call_llm via temp state to guarantee the
  # same resolved tools are used for dispatch. Temp state is not
  # persisted, so on resume after checkpoint we fall back to
  # rebuilding (same behavior as the original flow on resume).
  tools_dict = ctx.state.get('temp:tools_dict')
  if not tools_dict:
    llm_request = LlmRequest()
    await _process_agent_tools(invocation_context, llm_request)
    tools_dict = llm_request.tools_dict

  # --- Resume handling ---
  # When this node is re-run after an auth/confirmation interrupt
  # (rerun_on_resume=True), ctx.resume_inputs contains the user's
  # response keyed by interrupt ID. Both handlers self-filter via
  # schema validation (AuthConfig / ToolConfirmation), so they can
  # run unconditionally — this handles the mixed case where a single
  # LLM response triggered both auth and confirmation.
  if ctx.resume_inputs:
    auth_event = await _process_auth_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    conf_event = await _process_confirmation_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    lr_event = await _process_long_running_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    response_events = [e for e in [auth_event, conf_event, lr_event] if e]
    if response_events:
      for response_event in response_events:
        yield response_event.model_copy()
      if not any(e.actions.transfer_to_agent for e in response_events):
        yield _continue_event()
      return

  # Detect long-running tools before execution.
  long_running_tool_ids = get_long_running_function_calls(
      function_calls, tools_dict
  )

  # Execute function calls.
  function_response_event = await handle_function_call_list_async(
      invocation_context, function_calls, tools_dict
  )

  if not function_response_event:
    if long_running_tool_ids:
      yield _long_running_interrupt_event(
          invocation_context,
          function_calls,
          long_running_tool_ids,
      )
    return

  # Generate auth event if any tool requested credentials.
  auth_event = generate_auth_event(invocation_context, function_response_event)
  if auth_event:
    yield auth_event.model_copy()

  # Generate tool confirmation event if any tool requested confirmation.
  confirmation_event = generate_request_confirmation_event(
      invocation_context, function_calls, function_response_event
  )
  if confirmation_event:
    yield confirmation_event.model_copy()

  # Auth/confirmation events are interrupts — yield the function
  # response for persistence but do not route back to call_llm.
  if auth_event or confirmation_event:
    yield function_response_event.model_copy()
    return

  # When call_llm suppresses the finalized event for long-running
  # tools, the function_call Parts are missing from the session.
  # Yield them here (without long_running_tool_ids) so content
  # reconstruction works when the reason-act loop continues back
  # to call_llm (e.g. tool returned a pending response).
  if long_running_tool_ids:
    function_call_event = Event(
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        content=types.Content(
            role='model',
            parts=[types.Part(function_call=fc) for fc in function_calls],
        ),
    )
    yield function_call_event

  # Yield the function response event.
  yield function_response_event.model_copy()

  # Check for pending long-running tools that returned None (mixed case:
  # some tools completed, some long-running returned no response).
  if long_running_tool_ids:
    responded_ids = set()
    if function_response_event.content:
      for part in function_response_event.content.parts:
        if part.function_response and part.function_response.id:
          responded_ids.add(part.function_response.id)
    pending_ids = long_running_tool_ids - responded_ids
    if pending_ids:
      yield _long_running_interrupt_event(
          invocation_context,
          function_calls,
          pending_ids,
      )
      return

  # Check for structured output (set_model_response tool).
  json_response = _output_schema_processor.get_structured_model_response(
      function_response_event
  )
  if json_response:
    final_event = _output_schema_processor.create_final_model_response_event(
        invocation_context, json_response
    )
    yield final_event.model_copy()

  # Check for agent transfer.
  transfer_to_agent = function_response_event.actions.transfer_to_agent
  if transfer_to_agent:
    # Terminate the workflow. The function_response_event already
    # yielded above carries actions.transfer_to_agent for _Mesh to see.
    return

  # Check for task delegation.
  if function_response_event.actions.request_task:
    return

  # Check for task completion.
  if function_response_event.actions.finish_task:
    return

  # Check for skip_summarization (e.g. exit_loop sets this). In the old
  # flow, is_final_response() returning True would break the reason-act
  # while-loop. Without a 'continue' route, the workflow won't route
  # back to call_llm.
  if function_response_event.actions.skip_summarization:
    return

  # Route back to call_llm for the next reason-act iteration.
  yield _continue_event()
