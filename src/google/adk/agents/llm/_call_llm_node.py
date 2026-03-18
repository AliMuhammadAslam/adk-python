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

"""call_llm: workflow node function that encapsulates one LLM call cycle."""

from __future__ import annotations

import datetime
import logging
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel

from . import _basic
from . import _code_execution
from . import _compaction
from . import _context_cache_processor
from . import _identity
from . import _instructions
from . import _interactions_processor
from . import _nl_planning
from . import _output_schema_processor
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...telemetry.tracing import trace_call_llm
from ...telemetry.tracing import tracer
from ...utils.context_utils import Aclosing
from ..context import Context
from ..run_config import StreamingMode
from ._agent_transfer import inject_transfer_tools
from ._reasoning import _create_response_processors
from ._reasoning import _finalize_model_response_event
from ._reasoning import _handle_after_model_callback
from ._reasoning import _handle_before_model_callback
from ._reasoning import _process_agent_tools
from ._reasoning import _resolve_toolset_auth
from ._reasoning import _run_and_handle_error
from .task import _task_contents_processor

logger = logging.getLogger('google_adk.' + __name__)

_EXECUTE_TOOLS_ROUTE = 'execute_tools'
_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'


def _create_workflow_request_processors():
  """Request processors for the workflow-based _SingleLlmAgent.

  Same as ``single_flow._create_request_processors`` but excludes
  ``auth_preprocessor`` and ``request_confirmation`` processors.
  In the workflow flow, auth and confirmation resume is handled
  natively by the ``execute_tools`` node via ``rerun_on_resume=True``.
  """
  return [
      _basic.request_processor,
      _instructions.request_processor,
      _identity.request_processor,
      _compaction.request_processor,
      _task_contents_processor.request_processor,
      _context_cache_processor.request_processor,
      _interactions_processor.request_processor,
      _nl_planning.request_processor,
      _code_execution.request_processor,
      _output_schema_processor.request_processor,
  ]


class CallLlmResult(BaseModel):
  """Data passed from call_llm to execute_tools."""

  function_calls: list[types.FunctionCall]


async def call_llm(ctx: Context) -> AsyncGenerator[Any, None]:
  """Workflow node that encapsulates one LLM call cycle.

  Builds an ``LlmRequest``, runs request processors, calls the LLM,
  runs response processors, and yields the model response event.

  If the LLM response contains function calls, routes to
  ``execute_tools``. Otherwise the workflow ends (no route).
  """
  invocation_context = ctx.get_invocation_context()
  agent = invocation_context.agent

  llm_request = LlmRequest()

  # --- Run request processors ---
  # TODO: Remove Event conversion once workflow Event is merged
  # back into the base ADK Event.
  request_processors = _create_workflow_request_processors()
  for processor in request_processors:
    async with Aclosing(
        processor.run_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        copied = event.model_copy()
        copied.node_info.path = ctx.node_path
        yield copied

  # --- Resolve toolset authentication ---
  async with Aclosing(_resolve_toolset_auth(invocation_context, agent)) as agen:
    async for event in agen:
      copied = event.model_copy()
      copied.node_info.path = ctx.node_path
      yield copied
  if invocation_context.end_invocation:
    return

  # --- Process tool unions ---
  await _process_agent_tools(invocation_context, llm_request)

  # Store tools_dict in temp state so execute_tools can reuse the exact
  # same resolved tools. Temp state (prefix "temp:") is not persisted,
  # avoiding serialization issues with non-JSON-serializable BaseTool
  # instances. On resume after checkpoint, temp state is empty and
  # execute_tools falls back to rebuilding (same as the original flow).
  ctx.state['temp:tools_dict'] = llm_request.tools_dict

  # --- Inject transfer targets from WorkflowContext ---
  transfer_targets = ctx.transfer_targets
  if transfer_targets:
    await inject_transfer_tools(
        invocation_context, llm_request, transfer_targets
    )

  # --- Call the LLM ---
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      branch=invocation_context.branch,
  )

  # Before-model callback
  if response := await _handle_before_model_callback(
      invocation_context, llm_request, model_response_event
  ):
    finalized = _finalize_model_response_event(
        llm_request, response, model_response_event
    )
    copied = finalized.model_copy()
    copied.node_info.path = ctx.node_path
    yield copied
    return

  # Config setup
  llm_request.config = llm_request.config or types.GenerateContentConfig()
  llm_request.config.labels = llm_request.config.labels or {}
  if _ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
    llm_request.config.labels[_ADK_AGENT_NAME_LABEL_KEY] = agent.name

  # LLM call
  llm = agent.canonical_model
  invocation_context.increment_llm_call_count()
  responses_generator = llm.generate_content_async(
      llm_request,
      stream=invocation_context.run_config.streaming_mode == StreamingMode.SSE,
  )

  response_processors = _create_response_processors()

  async with Aclosing(
      _run_and_handle_error(
          responses_generator,
          invocation_context,
          llm_request,
          model_response_event,
      )
  ) as agen:
    async for llm_response in agen:
      with tracer.start_as_current_span('call_llm') as span:
        trace_call_llm(
            invocation_context,
            model_response_event.id,
            llm_request,
            llm_response,
            span,
        )

      # After-model callback
      if altered := await _handle_after_model_callback(
          invocation_context, llm_response, model_response_event
      ):
        llm_response = altered

      # --- Run response processors ---
      for processor in response_processors:
        async with Aclosing(
            processor.run_async(invocation_context, llm_response)
        ) as resp_agen:
          async for event in resp_agen:
            copied = event.model_copy()
            copied.node_info.path = ctx.node_path
            yield copied

      # Skip empty responses (needed for code executor to loop).
      if (
          not llm_response.content
          and not llm_response.error_code
          and not llm_response.interrupted
      ):
        continue

      # --- Finalize the model response event ---
      finalized_event = _finalize_model_response_event(
          llm_request, llm_response, model_response_event
      )
      # Long-running tool handling belongs in execute_tools, not
      # call_llm. Clear to prevent call_llm from being interrupted
      # by should_pause_invocation in node_runner.
      has_long_running = bool(finalized_event.long_running_tool_ids)
      finalized_event.long_running_tool_ids = None
      # Suppress the finalized event when it contains long-running
      # tool calls. The function_call data is already passed to
      # execute_tools via CallLlmResult, which yields the proper
      # interrupt event with long_running_tool_ids set.
      if not has_long_running:
        yield finalized_event

      # Update the mutable event id for next iteration (streaming).
      model_response_event.id = Event.new_id()
      model_response_event.timestamp = datetime.datetime.now().timestamp()

      # --- Route decision ---
      function_calls = finalized_event.get_function_calls()
      if function_calls and not finalized_event.partial:
        yield Event(
            route=_EXECUTE_TOOLS_ROUTE,
            output=CallLlmResult(
                function_calls=function_calls,
            ),
        )
