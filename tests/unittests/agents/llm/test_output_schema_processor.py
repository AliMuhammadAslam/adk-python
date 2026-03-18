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

"""Tests for output schema processor functionality."""

from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from pydantic import BaseModel
from pydantic import Field
import pytest


class PersonSchema(BaseModel):
  """Test schema for structured output."""

  name: str = Field(description="A person's name")
  age: int = Field(description="A person's age")
  city: str = Field(description='The city they live in')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Searched for: {query}'


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


@pytest.mark.asyncio
async def test_output_schema_with_tools_validation_removed():
  """Test that LlmAgent now allows output_schema with tools."""
  # This should not raise an error anymore
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  assert agent.output_schema == PersonSchema
  assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_output_schema_with_sub_agents():
  """Test that LlmAgent now allows output_schema with sub_agents."""
  sub_agent = LlmAgent(
      name='sub_agent',
      model='gemini-1.5-flash',
  )
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      sub_agents=[sub_agent],
  )

  assert agent.output_schema == PersonSchema
  assert len(agent.sub_agents) == 1


@pytest.mark.asyncio
async def test_basic_processor_skips_output_schema_with_tools():
  """Test that basic processor doesn't set output_schema when tools are present."""
  from google.adk.agents.llm._basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should not have set response_schema since agent has tools
  assert llm_request.config.response_schema is None
  assert llm_request.config.response_mime_type != 'application/json'


@pytest.mark.asyncio
async def test_basic_processor_sets_output_schema_without_tools():
  """Test that basic processor still sets output_schema when no tools are present."""
  from google.adk.agents.llm._basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[],  # No tools
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should have set response_schema since agent has no tools
  assert llm_request.config.response_schema == PersonSchema
  assert llm_request.config.response_mime_type == 'application/json'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'output_schema_with_tools_allowed',
    [
        False,
        True,
    ],
)
async def test_output_schema_request_processor(
    output_schema_with_tools_allowed, mocker
):
  """Test that output schema processor adds set_model_response tool."""
  from google.adk.agents.llm._output_schema_processor import _OutputSchemaRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _OutputSchemaRequestProcessor()

  can_use_output_schema_with_tools = mocker.patch(
      'google.adk.agents.llm._output_schema_processor.can_use_output_schema_with_tools',
      mock.MagicMock(return_value=output_schema_with_tools_allowed),
  )

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  if not output_schema_with_tools_allowed:
    # Should have added set_model_response tool if output schema with tools is
    # allowed
    assert 'set_model_response' in llm_request.tools_dict
    # Should have added instruction about using set_model_response
    assert 'set_model_response' in llm_request.config.system_instruction
  else:
    # Should skip modifying LlmRequest
    assert not llm_request.tools_dict
    assert not llm_request.config.system_instruction

  # Should have checked if output schema can be used with tools
  can_use_output_schema_with_tools.assert_called_once_with(
      agent.canonical_model
  )


@pytest.mark.asyncio
async def test_set_model_response_tool():
  """Test the set_model_response tool functionality."""
  from google.adk.tools.set_model_response_tool import SetModelResponseTool
  from google.adk.tools.tool_context import ToolContext

  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Call the tool with valid data
  result = await tool.run_async(
      args={'name': 'John Doe', 'age': 30, 'city': 'New York'},
      tool_context=tool_context,
  )

  # Verify the tool returns dict directly
  assert result is not None
  assert result['name'] == 'John Doe'
  assert result['age'] == 30
  assert result['city'] == 'New York'


@pytest.mark.asyncio
async def test_output_schema_helper_functions():
  """Test the helper functions for handling set_model_response."""
  from google.adk.agents.llm._output_schema_processor import create_final_model_response_event
  from google.adk.agents.llm._output_schema_processor import get_structured_model_response
  from google.adk.events.event import Event
  from google.genai import types

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  # Test get_structured_model_response with a function response event
  test_dict = {'name': 'Jane Smith', 'age': 25, 'city': 'Los Angeles'}
  test_json = '{"name": "Jane Smith", "age": 25, "city": "Los Angeles"}'

  # Create a function response event with set_model_response
  function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='set_model_response', response=test_dict
                  )
              )
          ],
      ),
  )

  # Test get_structured_model_response function
  extracted_json = get_structured_model_response(function_response_event)
  assert extracted_json == test_json

  # Test create_final_model_response_event function
  final_event = create_final_model_response_event(invocation_context, test_json)
  assert final_event.author == 'test_agent'
  assert final_event.invocation_id == invocation_context.invocation_id
  assert final_event.branch == invocation_context.branch
  assert final_event.content.role == 'model'
  assert final_event.content.parts[0].text == test_json

  # Test get_structured_model_response with non-set_model_response function
  other_function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='other_tool', response={'result': 'other response'}
                  )
              )
          ],
      ),
  )

  extracted_json = get_structured_model_response(other_function_response_event)
  assert extracted_json is None


@pytest.mark.asyncio
async def test_get_structured_model_response_with_non_ascii():
  """Test get_structured_model_response with non-ASCII characters."""
  from google.adk.agents.llm._output_schema_processor import get_structured_model_response
  from google.adk.events.event import Event
  from google.genai import types

  # Test with a dictionary containing non-ASCII characters
  test_dict = {'city': 'São Paulo'}
  expected_json = '{"city": "São Paulo"}'

  # Create a function response event
  function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='set_model_response', response=test_dict
                  )
              )
          ],
      ),
  )

  # Get the structured response
  extracted_json = get_structured_model_response(function_response_event)

  # Assert that the output is the expected JSON string without escaped characters
  assert extracted_json == expected_json


@pytest.mark.asyncio
async def test_get_structured_model_response_with_wrapped_result():
  """Test get_structured_model_response with wrapped list result.

  When a tool returns a non-dict (e.g., list), it gets wrapped as
  {'result': [...]}.  This test ensures we correctly unwrap the result.
  """
  from google.adk.agents.llm._output_schema_processor import get_structured_model_response
  from google.adk.events.event import Event
  from google.genai import types

  # Simulate a list result wrapped by ADK's functions.py
  wrapped_response = {
      'result': [
          {'name': 'Alice', 'age': 30},
          {'name': 'Bob', 'age': 25},
      ]
  }
  expected_json = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'

  # Create a function response event with wrapped result
  function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='set_model_response', response=wrapped_response
                  )
              )
          ],
      ),
  )

  # Get the structured response
  extracted_json = get_structured_model_response(function_response_event)

  # Should extract the unwrapped list, not the wrapped dict
  assert extracted_json == expected_json
