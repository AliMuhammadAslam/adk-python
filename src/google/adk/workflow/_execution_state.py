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

from enum import Enum
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class NodeStatus(Enum):
  """The status of a node in the workflow graph."""

  INACTIVE = 0
  """The node is not ready to be executed."""

  PENDING = 1
  """The node is ready to be executed."""

  RUNNING = 2
  """The node is being executed."""

  COMPLETED = 3
  """The node has been executed successfully."""

  WAITING = 4
  """The node is waiting (e.g. for a user response or re-trigger)."""

  FAILED = 5
  """The node has failed."""

  CANCELLED = 6
  """The node has been cancelled."""


class NodeState(BaseModel):
  """State of a node in the workflow."""

  model_config = ConfigDict(extra='ignore', ser_json_bytes='base64')

  status: NodeStatus = NodeStatus.INACTIVE
  """The execution status of the node."""

  input: Any = None
  """The input provided to the node."""

  triggered_by: Optional[str] = None
  """The node that triggered the current node."""

  retry_count: int = Field(default=0, exclude_if=lambda v: v == 0)
  """The retry count number for this node execution. 0 means this is the default first execution."""

  interrupts: list[str] = Field(default_factory=list)
  """The interrupt ids that are pending to be resolved."""

  resume_inputs: dict[str, Any] = Field(default_factory=dict)
  """The responses for resuming the node, keyed by interrupt id."""

  execution_id: str | None = None
  """The execution id of this node execution."""

  parent_execution_id: Optional[str] = None
  """The execution id of the parent node which dynamically
  scheduled this node execution."""

  source_node_name: Optional[str] = None
  """The original node definition which was dynamically scheduled."""
