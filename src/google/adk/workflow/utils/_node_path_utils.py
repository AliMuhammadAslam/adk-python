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

from .._node_path_builder import _NodePathBuilder

"""Node path utilities.

Node paths are slash-separated strings that uniquely identify a node within a
hierarchical workflow. Each component of the path represents the name of a
Workflow or a leaf node.

Example:
  node1 = BaseNode(name="node1")
  node2 = BaseNode(name="node2")

  workflow_a = Workflow(name="workflow_a", edges=[('START', node1)])
  workflow_b = Workflow(name="workflow_b", edges=[('START', node2)])

  root_agent = Workflow(
      name="root_agent",
      edges=[
          ('START', workflow_a, workflow_b),
      ],
  )

Node paths:
  root_agent: "root_agent@1"
  workflow_a: "root_agent@1/workflow_a@1"
  workflow_b: "root_agent@1/workflow_b@1"
  node1: "root_agent@1/workflow_a@1/node1@1"
  node2: "root_agent@1/workflow_b@1/node2@1"
"""


def get_node_name_from_path(path: str) -> str:
  """Returns the node name from a full node path.

  Args:
    path: The path to extract the node name from.

  Returns:
    The node name.
  """
  return _NodePathBuilder.from_string(path).node_name


def get_parent_path(path: str) -> str:
  """Returns the parent path from a full node path.

  Args:
    path: The node path.

  Returns:
    The parent path.
  """
  parent = _NodePathBuilder.from_string(path).parent
  return str(parent) if parent else ''


def join_paths(parent: str | None, child: str) -> str:
  """Joins a parent path and a child name.

  Args:
    parent: The parent path.
    child: The child name.

  Returns:
    The joined path.
  """
  if not parent:
    return child
  return str(_NodePathBuilder.from_string(parent).append(child))


def is_direct_child(parent_path: str | None, child_path: str | None) -> bool:
  """Checks if the child path is a direct child of the parent path.

  Example: is_direct_child('wf@1', 'wf@1/nodeA@1') → True
           is_direct_child('wf@1', 'wf@1/inner@1/nodeA@1') → False

  Args:
    parent_path: The parent node path.
    child_path: The child node path.

  Returns:
    True if child_path is a direct child of parent_path.
  """
  if not child_path:
    return False
  return _NodePathBuilder.from_string(child_path).is_direct_child_of(
      _NodePathBuilder.from_string(parent_path)
  )


def direct_child_name(parent_path: str, descendant_path: str) -> str:
  """Extracts the first-level child name from a descendant path.

  Example: direct_child_name('wf@1', 'wf@1/inner@1/nodeA@1') → 'inner@1'
  """
  child = _NodePathBuilder.from_string(parent_path).get_direct_child(
      _NodePathBuilder.from_string(descendant_path)
  )
  return child.node_name if child else ''


def is_descendant(ancestor_path: str, descendant_path: str | None) -> bool:
  """Checks if the descendant path is a descendant of the ancestor path.

  Args:
    ancestor_path: The ancestor node path.
    descendant_path: The descendant node path.

  Returns:
    True if descendant_path is a descendant of ancestor_path.
  """
  if not descendant_path:
    return False
  return _NodePathBuilder.from_string(descendant_path).is_descendant_of(
      _NodePathBuilder.from_string(ancestor_path)
  )
