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

"""LLM reasoning helpers: model callbacks, toolset auth, and processors."""

from __future__ import annotations

import inspect
import logging
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from . import _basic
from . import _code_execution
from . import _compaction
from . import _contents
from . import _context_cache_processor
from . import _identity
from . import _instructions
from . import _interactions_processor
from . import _nl_planning
from . import _output_schema_processor
from . import _request_confirmation
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_handler import AuthHandler
from ...auth.auth_tool import AuthConfig
from ...auth.credential_manager import CredentialManager
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry import tracing
from ...tools.base_toolset import BaseToolset
from ...tools.tool_context import ToolContext
from ...utils.context_utils import Aclosing
from ._functions import build_auth_request_event
from ._functions import get_long_running_function_calls
from ._functions import populate_client_function_call_id
from ._functions import REQUEST_EUC_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)

# Prefix used by toolset auth credential IDs
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'


def _finalize_model_response_event(
    llm_request: LlmRequest,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> Event:
  """Finalize and build the model response event from LLM response.

  Merges the LLM response data into the model response event and
  populates function call IDs and long-running tool information.

  Args:
    llm_request: The original LLM request.
    llm_response: The LLM response from the model.
    model_response_event: The base event to populate.

  Returns:
    The finalized Event with LLM response data merged in.
  """
  finalized_event = Event.model_validate({
      **model_response_event.model_dump(exclude_none=True),
      **llm_response.model_dump(exclude_none=True),
  })

  if finalized_event.content:
    function_calls = finalized_event.get_function_calls()
    if function_calls:
      populate_client_function_call_id(finalized_event)
      finalized_event.long_running_tool_ids = get_long_running_function_calls(
          function_calls, llm_request.tools_dict
      )

  return finalized_event


async def _resolve_toolset_auth(
    invocation_context: InvocationContext,
    agent: LlmAgent,
) -> AsyncGenerator[Event, None]:
  """Resolves authentication for toolsets before tool listing.

  For each toolset with auth configured via get_auth_config():
  - If credential is available, populate auth_config.exchanged_auth_credential
  - If credential is not available, yield auth request event and interrupt

  Args:
    invocation_context: The invocation context.
    agent: The LLM agent.

  Yields:
    Auth request events if any toolset needs authentication.
  """
  if not agent.tools:
    return

  pending_auth_requests: dict[str, AuthConfig] = {}
  callback_context = CallbackContext(invocation_context)

  for tool_union in agent.tools:
    if not isinstance(tool_union, BaseToolset):
      continue

    auth_config = tool_union.get_auth_config()
    if not auth_config:
      continue

    try:
      credential = await CredentialManager(auth_config).get_auth_credential(
          callback_context
      )
    except ValueError as e:
      # Validation errors from CredentialManager should be logged but not
      # block the flow - the toolset may still work without auth
      logger.warning(
          'Failed to get auth credential for toolset %s: %s',
          type(tool_union).__name__,
          e,
      )
      credential = None

    if credential:
      # Populate in-place for toolset to use in get_tools()
      auth_config.exchanged_auth_credential = credential
    else:
      # Need auth - will interrupt
      toolset_id = (
          f'{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}{type(tool_union).__name__}'
      )
      pending_auth_requests[toolset_id] = auth_config

  if not pending_auth_requests:
    return

  # Build auth requests dict with generated auth requests
  auth_requests = {
      credential_id: AuthHandler(auth_config).generate_auth_request()
      for credential_id, auth_config in pending_auth_requests.items()
  }

  # Yield event with auth requests using the shared helper
  yield build_auth_request_event(
      invocation_context,
      auth_requests,
      author=agent.name,
  )

  # Interrupt invocation
  invocation_context.end_invocation = True


async def _handle_before_model_callback(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
) -> Optional[LlmResponse]:
  """Runs before-model callbacks (plugins then agent callbacks).

  Args:
    invocation_context: The invocation context.
    llm_request: The LLM request being built.
    model_response_event: The model response event for callback context.

  Returns:
    An LlmResponse if a callback short-circuits the LLM call, else None.
  """
  agent = invocation_context.agent

  callback_context = CallbackContext(
      invocation_context, event_actions=model_response_event.actions
  )

  # First run callbacks from the plugins.
  callback_response = (
      await invocation_context.plugin_manager.run_before_model_callback(
          callback_context=callback_context,
          llm_request=llm_request,
      )
  )
  if callback_response:
    return callback_response

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  if not agent.canonical_before_model_callbacks:
    return
  for callback in agent.canonical_before_model_callbacks:
    callback_response = callback(
        callback_context=callback_context, llm_request=llm_request
    )
    if inspect.isawaitable(callback_response):
      callback_response = await callback_response
    if callback_response:
      return callback_response


async def _handle_after_model_callback(
    invocation_context: InvocationContext,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> Optional[LlmResponse]:
  """Runs after-model callbacks (plugins then agent callbacks).

  Also handles grounding metadata injection when google_search_agent is
  among the agent's tools.

  Args:
    invocation_context: The invocation context.
    llm_response: The LLM response to process.
    model_response_event: The model response event for callback context.

  Returns:
    An altered LlmResponse if a callback modifies it, else None.
  """
  agent = invocation_context.agent

  # Add grounding metadata to the response if needed.
  # TODO(b/448114567): Remove this function once the workaround is no longer needed.
  async def _maybe_add_grounding_metadata(
      response: Optional[LlmResponse] = None,
  ) -> Optional[LlmResponse]:
    readonly_context = ReadonlyContext(invocation_context)
    if (tools := invocation_context.canonical_tools_cache) is None:
      tools = await agent.canonical_tools(readonly_context)
      invocation_context.canonical_tools_cache = tools

    if not any(tool.name == 'google_search_agent' for tool in tools):
      return response
    ground_metadata = invocation_context.session.state.get(
        'temp:_adk_grounding_metadata', None
    )
    if not ground_metadata:
      return response

    if not response:
      response = llm_response
    response.grounding_metadata = ground_metadata
    return response

  callback_context = CallbackContext(
      invocation_context, event_actions=model_response_event.actions
  )

  # First run callbacks from the plugins.
  callback_response = (
      await invocation_context.plugin_manager.run_after_model_callback(
          callback_context=CallbackContext(invocation_context),
          llm_response=llm_response,
      )
  )
  if callback_response:
    return await _maybe_add_grounding_metadata(callback_response)

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  if not agent.canonical_after_model_callbacks:
    return await _maybe_add_grounding_metadata()
  for callback in agent.canonical_after_model_callbacks:
    callback_response = callback(
        callback_context=callback_context, llm_response=llm_response
    )
    if inspect.isawaitable(callback_response):
      callback_response = await callback_response
    if callback_response:
      return await _maybe_add_grounding_metadata(callback_response)
  return await _maybe_add_grounding_metadata()


async def _run_and_handle_error(
    response_generator: AsyncGenerator[LlmResponse, None],
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
) -> AsyncGenerator[LlmResponse, None]:
  """Wraps an LLM response generator with error callback handling.

  Runs the response generator within a tracing span. If an error occurs,
  runs on-model-error callbacks (plugins then agent callbacks). If a
  callback returns a response, that response is yielded instead of
  re-raising the error.

  Args:
    response_generator: The async generator producing LLM responses.
    invocation_context: The invocation context.
    llm_request: The LLM request.
    model_response_event: The model response event.

  Yields:
    LlmResponse objects from the generator.

  Raises:
    The original model error if no error callback handles it.
  """
  agent = invocation_context.agent
  if not hasattr(agent, 'canonical_on_model_error_callbacks'):
    raise TypeError(
        'Expected agent to have canonical_on_model_error_callbacks'
        f' attribute, but got {type(agent)}'
    )

  async def _run_on_model_error_callbacks(
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    error_response = (
        await invocation_context.plugin_manager.run_on_model_error_callback(
            callback_context=callback_context,
            llm_request=llm_request,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    for callback in agent.canonical_on_model_error_callbacks:
      error_response = callback(
          callback_context=callback_context,
          llm_request=llm_request,
          error=error,
      )
      if inspect.isawaitable(error_response):
        error_response = await error_response
      if error_response is not None:
        return error_response

    return None

  try:
    async with Aclosing(response_generator) as agen:
      async with tracing.use_inference_span(
          llm_request,
          invocation_context,
          model_response_event,
      ) as gc_span:
        async for llm_response in agen:
          if gc_span:
            tracing.trace_inference_result(
                gc_span,
                llm_response,
            )
          yield llm_response
  except Exception as model_error:
    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )
    error_response = await _run_on_model_error_callbacks(
        callback_context=callback_context,
        llm_request=llm_request,
        error=model_error,
    )
    if error_response is not None:
      yield error_response
    else:
      raise model_error


async def _process_agent_tools(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Process the agent's tools and populate ``llm_request.tools_dict``.

  Iterates over the agent's ``tools`` list, converts each tool union
  (callable, BaseTool, or BaseToolset) into resolved ``BaseTool``
  instances, and calls ``process_llm_request`` on each to register
  tool declarations in the request.

  After this function returns, ``llm_request.tools_dict`` maps tool
  names to ``BaseTool`` instances ready for function call dispatch.

  Args:
    invocation_context: The invocation context (``agent`` is read
      from ``invocation_context.agent``).
    llm_request: The LLM request to populate with tool declarations.
  """
  agent = invocation_context.agent
  if not hasattr(agent, 'tools') or not agent.tools:
    return

  multiple_tools = len(agent.tools) > 1
  model = agent.canonical_model
  for tool_union in agent.tools:
    tool_context = ToolContext(invocation_context)

    # If it's a toolset, process it first
    if isinstance(tool_union, BaseToolset):
      await tool_union.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

    from ...agents.llm_agent import _convert_tool_union_to_tools

    # Then process all tools from this tool union
    tools = await _convert_tool_union_to_tools(
        tool_union,
        ReadonlyContext(invocation_context),
        model,
        multiple_tools,
    )
    for tool in tools:
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )


def _create_request_processors():
  """Create the standard request processor list for an LLM agent."""
  from ...auth import auth_preprocessor

  return [
      _basic.request_processor,
      auth_preprocessor.request_processor,
      _request_confirmation.request_processor,
      _instructions.request_processor,
      _identity.request_processor,
      # Compaction should run before contents so compacted events are
      # reflected in the model request context.
      _compaction.request_processor,
      _contents.request_processor,
      # Context cache processor sets up cache config and finds
      # existing cache metadata.
      _context_cache_processor.request_processor,
      # Interactions processor extracts previous_interaction_id for
      # stateful conversations via the Interactions API.
      _interactions_processor.request_processor,
      # Some implementations of NL Planning mark planning contents
      # as thoughts in the post processor.  Since these need to be
      # unmarked, NL Planning should be after contents.
      _nl_planning.request_processor,
      # Code execution should be after the contents as it mutates
      # the contents to optimize data files.
      _code_execution.request_processor,
      # Output schema processor adds system instruction and
      # set_model_response when both output_schema and tools are
      # present.
      _output_schema_processor.request_processor,
  ]


def _create_response_processors():
  """Create the standard response processor list for an LLM agent."""
  return [
      _nl_planning.response_processor,
      _code_execution.response_processor,
  ]
