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

"""Type definitions for building workflow graphs."""

from typing import Callable
from typing import Literal

from ..tools.base_tool import BaseTool
from ._base_node import BaseNode

RouteValue = bool | int | str
"""Type alias for valid routing values used in conditional graph edges."""

NodeLike = BaseNode | BaseTool | Callable | Literal['START']
"""Type alias for objects that can be converted to a workflow node."""

RoutingMap = dict[RouteValue, NodeLike | tuple[NodeLike, ...]]
"""A mapping from route values to destination nodes.

Syntactic sugar for declaring multiple routed edges from a single source.
Values can be a single node or a tuple of nodes (fan-out).

Examples::

    {"route_a": node_a, "route_b": node_b}
    {"route_x": (node_a, node_b)}  # fan-out: both triggered
"""
