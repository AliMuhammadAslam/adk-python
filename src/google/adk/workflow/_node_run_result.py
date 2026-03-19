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

"""Structured result from NodeRunner. Runtime-only, not persisted."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from ._definitions import RouteValue


@dataclass
class NodeRunResult:
  """Structured result from NodeRunner.run().

  Used internally by orchestrators (WorkflowNode, MeshNode, LlmAgent)
  that need output, route, and interrupt info. User-facing ctx.run_node()
  returns just the output.
  """

  output: Any | None = None
  """The node's output value. None if no output Event was yielded."""

  route: RouteValue | list[RouteValue] | None = None
  """Route value for conditional edge matching. None if no routing."""

  interrupt_ids: list[str] = field(default_factory=list)
  """IDs of interrupts. Empty means not interrupted."""
