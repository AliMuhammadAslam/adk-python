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

"""Parallel agent implementation using workflow graph."""

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar

from typing_extensions import override

from ..events.event import Event
from ..workflow import START
from ..workflow import Workflow
from .base_agent_config import BaseAgentConfig
from .context import Context
from .parallel_agent_config import ParallelAgentConfig

logger = logging.getLogger('google_adk.' + __name__)


class ParallelAgent(Workflow):
  """A shell agent that runs its sub-agents in parallel using workflow graph."""

  config_type: ClassVar[type[BaseAgentConfig]] = ParallelAgentConfig
  """The config type for this agent."""

  @override
  def model_post_init(self, context: Any) -> None:
    if self.sub_agents:
      if self.graph is not None or self.edges:
        raise ValueError(
            'ParallelAgent constructs its graph internally and does not'
            " accept 'graph' or 'edges' arguments."
        )

      # Create edges from START to all sub-agents
      for sub_agent in self.sub_agents:
        self.edges.append((START, sub_agent))

    super().model_post_init(context)

  @override
  async def _run_live_impl(self, ctx: Context) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for ParallelAgent.')
    yield  # AsyncGenerator requires having at least one yield statement
