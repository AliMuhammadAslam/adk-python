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

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import click
from google.genai import types
from pydantic import BaseModel

from ..agents.llm_agent import LlmAgent
from ..apps.app import App
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ..utils.env_utils import is_env_enabled
from .service_registry import load_services_module
from .utils import envs
from .utils.agent_loader import AgentLoader
from .utils.service_factory import create_artifact_service_from_options
from .utils.service_factory import create_memory_service_from_options
from .utils.service_factory import create_session_service_from_options


class InputFile(BaseModel):
  state: dict[str, object]
  queries: list[str]


async def run_input_file(
    app_name: str,
    user_id: str,
    agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
    input_path: str,
    memory_service: Optional[BaseMemoryService] = None,
) -> Session:
  app = (
      agent_or_app
      if isinstance(agent_or_app, App)
      else App(name=app_name, root_agent=agent_or_app)
  )
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      credential_service=credential_service,
  )
  with open(input_path, 'r', encoding='utf-8') as f:
    input_file = InputFile.model_validate_json(f.read())
  input_file.state['_time'] = datetime.now().isoformat()

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, state=input_file.state
  )
  for query in input_file.queries:
    click.echo(f'[user]: {query}')
    content = types.Content(role='user', parts=[types.Part(text=query)])
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        )
    ) as agen:
      async for event in agen:
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')
  return session


_REQUEST_INPUT = 'adk_request_input'
_REQUEST_CONFIRMATION = 'adk_request_confirmation'


def _collect_pending_function_calls(
    events: list[Event],
) -> list[tuple[str, str, dict[str, Any]]]:
  """Collects pending HITL function calls from events.

  Returns a list of (function_call_id, function_name, args) tuples
  for function calls that need user input.
  """
  pending = []
  for event in events:
    lr_ids = getattr(event, 'long_running_tool_ids', None)
    if not lr_ids:
      continue
    content = getattr(event, 'content', None)
    if not content or not content.parts:
      continue
    for part in content.parts:
      fc = part.function_call
      if fc and fc.id in lr_ids:
        pending.append((fc.id, fc.name, fc.args or {}))
  return pending


def _prompt_for_function_call(
    fc_id: str, fc_name: str, args: dict[str, Any]
) -> types.Content:
  """Prompts the user for a HITL function call and returns the response."""
  if fc_name == _REQUEST_INPUT:
    message = args.get('message') or 'Input requested'
    schema = args.get('response_schema')
    click.echo(f'[HITL input] {message}')
    if schema:
      click.echo(f'  Schema: {json.dumps(schema)}')
  elif fc_name == _REQUEST_CONFIRMATION:
    tool_confirmation = args.get('toolConfirmation', {})
    hint = tool_confirmation.get('hint', '')
    original_fc = args.get('originalFunctionCall', {})
    original_name = original_fc.get('name', 'unknown')
    click.echo(f'[HITL confirm] {hint or f"Confirm {original_name}?"}')
    click.echo('  Type "yes" to confirm, anything else to reject.')
  else:
    click.echo(f'[HITL] Waiting for input for {fc_name}({args})')

  user_input = input('[user]: ')

  # Build the FunctionResponse.
  if fc_name == _REQUEST_CONFIRMATION:
    confirmed = user_input.strip().lower() in ('yes', 'y')
    response = {'confirmed': confirmed}
  else:
    # Try to parse as JSON, fall back to wrapping as {"result": value}.
    try:
      parsed = json.loads(user_input)
      response = parsed if isinstance(parsed, dict) else {'result': parsed}
    except (json.JSONDecodeError, ValueError):
      response = {'result': user_input}

  return types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=fc_id,
                  name=fc_name,
                  response=response,
              )
          )
      ],
  )


async def run_interactively(
    root_agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session: Session,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
    memory_service: Optional[BaseMemoryService] = None,
) -> None:
  app = (
      root_agent_or_app
      if isinstance(root_agent_or_app, App)
      else App(name=session.app_name, root_agent=root_agent_or_app)
  )
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      credential_service=credential_service,
  )

  next_message = None
  resume_invocation_id = None
  while True:
    if next_message is None:
      query = input('[user]: ')
      if not query or not query.strip():
        continue
      if query == 'exit':
        break
      next_message = types.Content(role='user', parts=[types.Part(text=query)])

    collected_events = []
    invocation_id = None
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=next_message,
            invocation_id=resume_invocation_id,
        )
    ) as agen:
      async for event in agen:
        collected_events.append(event)
        if getattr(event, 'invocation_id', None):
          invocation_id = event.invocation_id
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')

    next_message = None
    resume_invocation_id = None

    # Check for pending HITL function calls that need user input.
    pending = _collect_pending_function_calls(collected_events)
    if pending:
      # Handle each pending function call. If there are multiple,
      # collect all responses into a single Content with multiple parts.
      parts = []
      for fc_id, fc_name, args in pending:
        response_content = _prompt_for_function_call(fc_id, fc_name, args)
        parts.extend(response_content.parts)
      next_message = types.Content(role='user', parts=parts)
      resume_invocation_id = invocation_id

  await runner.close()


async def run_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    input_file: Optional[str] = None,
    saved_session_file: Optional[str] = None,
    save_session: bool,
    session_id: Optional[str] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = True,
) -> None:
  """Runs an interactive CLI for a certain agent.

  Args:
    agent_parent_dir: str, the absolute path of the parent folder of the agent
      folder.
    agent_folder_name: str, the name of the agent folder.
    input_file: Optional[str], the absolute path to the json file that contains
      the initial session state and user queries, exclusive with
      saved_session_file.
    saved_session_file: Optional[str], the absolute path to the json file that
      contains a previously saved session, exclusive with input_file.
    save_session: bool, whether to save the session on exit.
    session_id: Optional[str], the session ID to save the session to on exit.
    session_service_uri: Optional[str], custom session service URI.
    artifact_service_uri: Optional[str], custom artifact service URI.
    memory_service_uri: Optional[str], custom memory service URI.
    use_local_storage: bool, whether to use local .adk storage by default.
  """
  agent_parent_path = Path(agent_parent_dir).resolve()
  agent_root = agent_parent_path / agent_folder_name
  load_services_module(str(agent_root))
  user_id = 'test_user'

  agents_dir = str(agent_parent_path)
  agent_loader = AgentLoader(agents_dir=agents_dir)
  agent_or_app = agent_loader.load_agent(agent_folder_name)
  session_app_name = (
      agent_or_app.name if isinstance(agent_or_app, App) else agent_folder_name
  )
  app_name_to_dir = None
  if isinstance(agent_or_app, App) and agent_or_app.name != agent_folder_name:
    app_name_to_dir = {agent_or_app.name: agent_folder_name}

  if not is_env_enabled('ADK_DISABLE_LOAD_DOTENV'):
    envs.load_dotenv_for_agent(agent_folder_name, agents_dir)

  # Create session and artifact services using factory functions.
  # Sessions persist under <agents_dir>/<agent>/.adk/session.db when enabled.
  session_service = create_session_service_from_options(
      base_dir=agent_parent_path,
      session_service_uri=session_service_uri,
      app_name_to_dir=app_name_to_dir,
      use_local_storage=use_local_storage,
  )

  artifact_service = create_artifact_service_from_options(
      base_dir=agent_root,
      artifact_service_uri=artifact_service_uri,
      use_local_storage=use_local_storage,
  )
  memory_service = create_memory_service_from_options(
      base_dir=agent_parent_path,
      memory_service_uri=memory_service_uri,
  )

  credential_service = InMemoryCredentialService()

  # Helper function for printing events
  def _print_event(event) -> None:
    content = event.content
    if not content or not content.parts:
      return
    text_parts = [part.text for part in content.parts if part.text]
    if not text_parts:
      return
    author = event.author or 'system'
    click.echo(f'[{author}]: {"".join(text_parts)}')

  if input_file:
    session = await run_input_file(
        app_name=session_app_name,
        user_id=user_id,
        agent_or_app=agent_or_app,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
        credential_service=credential_service,
        input_path=input_file,
    )
  elif saved_session_file:
    # Load the saved session from file
    with open(saved_session_file, 'r', encoding='utf-8') as f:
      loaded_session = Session.model_validate_json(f.read())

    # Create a new session in the service, copying state from the file
    session = await session_service.create_session(
        app_name=session_app_name,
        user_id=user_id,
        state=loaded_session.state if loaded_session else None,
    )

    # Append events from the file to the new session and display them
    if loaded_session:
      for event in loaded_session.events:
        await session_service.append_event(session, event)
        _print_event(event)

    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
        memory_service=memory_service,
    )
  else:
    session = await session_service.create_session(
        app_name=session_app_name, user_id=user_id
    )
    click.echo(f'Running agent {agent_or_app.name}, type exit to exit.')
    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
        memory_service=memory_service,
    )

  if save_session:
    session_id = session_id or input('Session ID to save: ')
    session_path = agent_root / f'{session_id}.session.json'

    # Fetch the session again to get all the details.
    session = await session_service.get_session(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    session_path.write_text(
        session.model_dump_json(indent=2, exclude_none=True, by_alias=True),
        encoding='utf-8',
    )

    print('Session saved to', session_path)
