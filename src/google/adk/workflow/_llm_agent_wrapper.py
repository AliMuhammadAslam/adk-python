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

"""Wrapper that adapts an LlmAgent for use as a workflow graph node.

- Sets a branch for content isolation (single_turn mode only)
- Converts node_input to user content (single_turn mode only)
- Re-emits finish_task output so the outer node_runner can route it
"""

from __future__ import annotations

import json
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.context import Context
from ..agents.llm_agent import LlmAgent
from ..events.event import Event
from ._base_node import BaseNode


def _node_input_to_content(node_input: Any) -> types.Content:
  """Converts node_input to a user Content for the LLM agent."""
  if isinstance(node_input, types.Content):
    return node_input
  if isinstance(node_input, str):
    text = node_input
  elif isinstance(node_input, BaseModel):
    text = node_input.model_dump_json()
  elif isinstance(node_input, (dict, list)):
    text = json.dumps(node_input)
  else:
    text = str(node_input)
  return types.Content(role='user', parts=[types.Part(text=text)])


class _LlmAgentWrapper(BaseNode):
  """Adapts a task/single_turn LlmAgent for use as a workflow graph node.

  Output handling by mode:
    single_turn (leaf, no sub_agents): Bypasses Mesh by running
      _SingleLlmAgent directly. Output is extracted via
      LlmAgent._maybe_save_output_to_state and emitted as a
      separate Event before END_OF_AGENT.
    single_turn (with sub_agents): Runs the full LlmAgent which
      handles output internally via run_node_impl(). The wrapper
      only suppresses output on interrupt.
    task: The wrapper intercepts finish_task actions and re-emits the
      output as a separate Event, since the original finish_task event
      carries the output inside actions, not in event.output.
  """

  agent: LlmAgent = Field(...)
  rerun_on_resume: bool = Field(default=True)
  _single: Any = PrivateAttr(default=None)

  @model_validator(mode='before')
  @classmethod
  def _set_defaults(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if data.get('name') is None and 'agent' in data:
        data['name'] = getattr(data['agent'], 'name', '')
    return data

  @model_validator(mode='after')
  def _validate_and_default_mode(self) -> _LlmAgentWrapper:
    """Defaults unset mode to single_turn; rejects unsupported modes."""
    if self.agent.mode not in ('task', 'single_turn'):
      if 'mode' not in self.agent.model_fields_set:
        self.agent._update_mode('single_turn')
      else:
        raise ValueError(
            f'LlmAgentWrapper only supports task and single_turn mode,'
            f" but agent '{self.agent.name}' has"
            f" mode='{self.agent.mode}'."
        )
    if self.agent.mode == 'task':
      self.wait_for_output = True

    # For leaf single_turn agents, use _SingleLlmAgent directly,
    # bypassing the _Mesh orchestration layer.
    if self.agent.mode == 'single_turn' and not self.agent.sub_agents:
      from ..agents.llm._single_llm_agent import _SingleLlmAgent

      self._single = _SingleLlmAgent.from_base_llm_agent(self.agent)

    return self

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> _LlmAgentWrapper:
    """Propagates name updates to the inner agent.

    When _ParallelWorker schedules dynamic nodes, each worker gets a
    unique name (e.g. 'agent__0'). The inner agent must also receive
    this name so that events carry the correct author.
    """
    copied = super().model_copy(update=update, deep=deep)
    if update and 'name' in update:
      copied.agent = copied.agent.model_copy(update={'name': update['name']})
      if copied._single is not None:
        copied._single = copied._single.model_copy(
            update={'name': update['name']}
        )
    return copied

  def _validate_input(self, node_input: Any) -> None:
    """Validates node_input against the agent's input_schema if set."""
    if not self.agent.input_schema or node_input is None:
      return
    if isinstance(node_input, dict):
      self.agent.input_schema.model_validate(node_input)
    elif isinstance(node_input, BaseModel):
      self.agent.input_schema.model_validate(node_input.model_dump())

  def _prepare_input(
      self, ctx: Context, node_input: Any
  ) -> tuple[Context, Any]:
    """Prepares the agent context and input based on mode.

    Single_turn agents get content isolation via a branch so parallel
    agents don't see each other's events. Task agents skip branching
    because HITL user messages are appended without a branch.
    """
    if self.agent.mode != 'single_turn':
      return ctx, node_input

    node_path = ctx.node_path or ''
    branch = f'node:{node_path}.{self.agent.name}'
    agent_input = (
        _node_input_to_content(node_input) if node_input is not None else None
    )
    ic = ctx._invocation_context.model_copy(
        update={'branch': branch},
    )
    agent_ctx = Context(
        invocation_context=ic,
        node_path=ctx.node_path,
        execution_id=ctx.execution_id,
    )
    return agent_ctx, agent_input

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the wrapped agent and translates output for downstream nodes."""
    self._validate_input(node_input)
    agent_ctx, agent_input = self._prepare_input(ctx, node_input)

    inner = self._single if self._single is not None else self.agent

    # When the agent has parallel_worker=True, call run_node_impl()
    # directly to bypass Node.run()'s internal parallel logic.
    if self.agent.parallel_worker:
      run_iter = inner.run_node_impl(ctx=agent_ctx, node_input=agent_input)
    else:
      run_iter = inner.run(ctx=agent_ctx, node_input=agent_input)

    if self.agent.mode == 'single_turn':
      if self._single is not None:
        # Leaf agent bypass: since we skip LlmAgent.run_node_impl(),
        # replicate its output handling here.
        # _maybe_save_output_to_state applies output_schema/output_key
        # and clears content on the final response. We emit the output
        # as a pathless Event before END_OF_AGENT.
        node_path = agent_ctx.node_path or ''
        single_output = None
        async for event in run_iter:
          if isinstance(event, Event):
            output_before = event.output
            self.agent._maybe_save_output_to_state(event, node_path)
            if event.output is not None and output_before is None:
              single_output = event.output
          if (
              single_output is not None
              and isinstance(event, Event)
              and event.actions
              and event.actions.end_of_agent
          ):
            yield Event(output=single_output)
            single_output = None
          yield event
        if single_output is not None:
          yield Event(output=single_output)
      else:
        # Agent with sub_agents: LlmAgent.run_node_impl() handles
        # output internally. Suppress output when interrupted to
        # avoid mixed output/interrupt errors in node_runner.
        interrupted = False
        async for event in run_iter:
          if isinstance(event, Event) and event.long_running_tool_ids:
            interrupted = True
          if (
              interrupted
              and isinstance(event, Event)
              and event.output is not None
              and not event.node_info.path
          ):
            continue
          yield event
    else:
      # Task mode: finish_task output is inside event.actions, not
      # event.output. Intercept it and re-emit as a proper output event.
      finish_task_output = None
      async for event in run_iter:
        yield event
        if (
            isinstance(event, Event)
            and event.actions
            and event.actions.finish_task
        ):
          finish_task_output = Event(
              output=event.actions.finish_task.get('output')
          )

      if finish_task_output:
        yield finish_task_output
