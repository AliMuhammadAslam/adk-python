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

from __future__ import annotations

"""Utilities for ADK workflows."""

from collections.abc import Mapping
import json
from typing import Any

from google.genai import types

from ...events.event import Event
from ...events.request_input import RequestInput
from ...utils._schema_utils import schema_to_json_schema

REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'

_RESULT_KEY = 'result'
"""Key used to wrap non-dict values in a FunctionResponse dict."""


def wrap_response(value: Any) -> dict[str, Any]:
  """Wraps a value into a dict suitable for FunctionResponse.response.

  If the value is already a dict, returns it as-is.
  Otherwise wraps as ``{"result": value}``.
  """
  if isinstance(value, dict):
    return value
  return {_RESULT_KEY: value}


def unwrap_response(data: Any) -> Any:
  """Unwraps a FunctionResponse dict to the original value.

  If ``data`` is a dict with exactly one key ``"result"``, extracts the
  value.  String values are JSON-parsed when possible (the web frontend
  wraps user text as ``{"result": text}`` without parsing).

  Otherwise returns ``data`` unchanged.
  """
  if isinstance(data, dict) and len(data) == 1 and _RESULT_KEY in data:
    value = data[_RESULT_KEY]
    if isinstance(value, str):
      try:
        value = json.loads(value)
      except (json.JSONDecodeError, ValueError):
        pass
    return value
  return data


def create_request_input_event(request_input: RequestInput) -> Event:
  """Creates a RequestInput event from a RequestInput object."""
  args = request_input.model_dump(exclude={'response_schema'})
  args['response_schema'] = (
      schema_to_json_schema(request_input.response_schema)
      if request_input.response_schema is not None
      else None
  )
  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                      args=args,
                      id=request_input.interrupt_id,
                  )
              )
          ]
      ),
      long_running_tool_ids=[request_input.interrupt_id],
  )


def has_request_input_function_call(event: Event) -> bool:
  """Checks if an event contains a `request_input` function call."""
  if not (event.content and event.content.parts):
    return False
  return any(
      p.function_call
      and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
      for p in event.content.parts
  )


def create_request_input_response(
    interrupt_id: str,
    response: Mapping[str, Any],
) -> types.Part:
  """Creates a FunctionResponse part in response to a `request_input` function call.

  Args:
    interrupt_id: The interrupt_id from an event containing a `request_input`
      function call.
    response: The response data to send back.

  Returns:
    A types.Part containing the FunctionResponse.
  """
  return types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id,
          name=REQUEST_INPUT_FUNCTION_CALL_NAME,
          response=response,
      )
  )


def get_request_input_interrupt_ids(event: Event) -> list[str]:
  """Extracts interrupt_ids from an event containing `request_input` function calls."""
  interrupt_ids = []
  if not event.content or not event.content.parts:
    return interrupt_ids
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    ):
      interrupt_ids.append(part.function_call.id)
  return interrupt_ids
