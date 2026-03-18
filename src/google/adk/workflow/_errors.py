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

"""Errors raised by the workflow framework."""


class NodeInterruptedError(BaseException):
  """Raised when a node execution is interrupted.

  This should only be raised by ADK and should not be caught by the user.
  """


class NodeTimeoutError(Exception):
  """Raised when a node exceeds its configured timeout.

  This is a regular ``Exception`` (not ``BaseException``) so it is
  compatible with ``retry_config`` — a timed-out node can be retried.
  """

  def __init__(self, node_name: str, timeout: float) -> None:
    self.node_name = node_name
    self.timeout = timeout
    super().__init__(f"Node '{node_name}' timed out after {timeout} seconds.")
