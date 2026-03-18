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

"""Utility functions for visualizing agent graphs."""

from __future__ import annotations

from typing import Any

import graphviz

from ...workflow._execution_state import NodeStatus


def plot_workflow_graph(
    app_info: dict[str, Any],
    agent_state: dict[str, Any] = {},
    format: str = "svg",
) -> str | bytes:
  """Plots the workflow graph with node statuses."""
  root_agent = app_info.get("root_agent", {})
  graph = root_agent.get("graph", {})

  if not graph:
    root_name = root_agent.get("name", "root_agent")
    sub_agents = root_agent.get("sub_agents", [])
    if not sub_agents:
      return "" if format in ("svg", "dot") else b""

    nodes = [{"name": root_name}]
    edges = []

    def _traverse_sub_agents(agent_dict, parent_name):
      for sub in agent_dict.get("sub_agents", []):
        sub_name = sub.get("name")
        if sub_name:
          nodes.append({"name": sub_name})
          edges.append({
              "from_node": {"name": parent_name},
              "to_node": {"name": sub_name},
          })
          _traverse_sub_agents(sub, sub_name)

    _traverse_sub_agents(root_agent, root_name)
    graph = {"nodes": nodes, "edges": edges}

  nodes_state = agent_state.get("nodes", {})
  dot = graphviz.Digraph(comment="Workflow Visualization")

  dot.attr(
      "graph",
      bgcolor="#F8FAFC",
      pad="0.5",
      nodesep="0.5",
      ranksep="0.8",
      fontname="Helvetica",
      splines="spline",
  )

  dot.attr(
      "node",
      shape="rect",
      style="rounded,filled",
      fillcolor="#FFFFFF",
      color="#94A3B8",
      penwidth="1.5",
      fontname="Helvetica",
      fontcolor="#0F172A",
      fontsize="12",
      margin="0.25,0.15",
  )

  dot.attr(
      "edge",
      color="#64748B",
      penwidth="1.2",
      fontname="Helvetica",
      fontcolor="#475569",
      fontsize="10",
      arrowhead="vee",
      arrowsize="0.7",
  )

  # Add nodes
  nodes = graph.get("nodes", [])
  for node in nodes:
    node_name = node.get("name")
    if not node_name or node_name == "__START__":
      continue

    node_data = nodes_state.get(node_name, {})
    status_val = node_data.get("status", NodeStatus.INACTIVE.value)
    if isinstance(status_val, NodeStatus):
      status = status_val
    else:
      try:
        status = NodeStatus(status_val)
      except (ValueError, KeyError):
        status = NodeStatus.INACTIVE

    fillcolor = "#FFFFFF"  # Default
    if status == NodeStatus.COMPLETED:
      fillcolor = "#69CB87"  # updated light green
    elif status == NodeStatus.RUNNING:
      fillcolor = "#e8b589"
    elif status == NodeStatus.FAILED:
      fillcolor = "salmon"
    elif status == NodeStatus.INACTIVE:
      fillcolor = "#FFFFFF"  # Let it match global style
    elif status == NodeStatus.WAITING:
      fillcolor = "#d2a6e0"
    elif status == NodeStatus.CANCELLED:
      fillcolor = "lightgray"

    dot.node(node_name, node_name, style="rounded,filled", fillcolor=fillcolor)

  # Add edges
  edges = graph.get("edges", [])
  for edge in edges:
    from_node_obj = edge.get("from_node", {})
    to_node_obj = edge.get("to_node", {})

    from_node = from_node_obj.get("name")
    to_node = to_node_obj.get("name")

    if from_node == "__START__":
      dot.node(
          "__START__",
          "START",
          shape="circle",
          style="filled",
          fillcolor="#3B82F6",
          color="#2563EB",
          fontcolor="#FFFFFF",
          fontname="Helvetica-Bold",
          width="0.9",
          fixedsize="true",
      )

    if from_node and to_node:
      label = f"  {edge.get('route')}" if edge.get("route") else ""
      dot.edge(from_node, to_node, label=label)

  if format == "dot":
    return dot.source
  if format == "svg":
    return dot.pipe(format="svg").decode("utf-8")
  return dot.pipe(format=format)
