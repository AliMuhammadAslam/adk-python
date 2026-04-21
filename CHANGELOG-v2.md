# Changelog

## [2.0.0-beta.1](https://github.com/google/adk-python/compare/v2.0.0-beta.0...v2.0.0-beta.1) (2026-04-21)

### Highlights

*   **Transition to Beta**: Updated documentation to reflect the project's move to Beta phase.
*   **Workflow Engine Refactoring**: Significant cleanup and stabilization of the ADK 2.0 Workflow engine, including stateless JoinNode, standardized retry mechanics, and improved session rehydration.
*   **Security Fix**: Resolved a potential RCE vulnerability related to nested YAML configurations.
*   **Documentation & Style**: Modularized the ADK style guide and added new topics.


## [2.0.0-alpha.3](https://github.com/google/adk-python/compare/v2.0.0-alpha.2...v2.0.0-alpha.3) (2026-04-09)

### Features

* **Workflow Orchestration:** Added Workflow(BaseNode) graph orchestration implementation, support for lazy scan deduplication and resume for dynamic nodes, partial resume for nested workflows, and state/artifact delta bundling onto yielded events.
* **CLI and Web UI:** Added support for Workflow graph visualization in web UI, improved graph readability with distinct icons and shapes, and active node rendering in event graph.
* **Documentation:** Added reference documentation for development (skills like adk-style, adk-git, and observability architecture).

## [2.0.0-alpha.1](https://github.com/google/adk-python/compare/v2.0.0-alpha.0...v2.0.0-alpha.1) (2026-03-18)

### Features

Introduces two major capabilities:
* Workflow runtime: graph-based execution engine for composing
  deterministic execution flows for agentic apps, with support for
  routing, fan-out/fan-in, loops, retry, state management, dynamic
  nodes, human-in-the-loop, and nested workflows
* Task API: structured agent-to-agent delegation with multi-turn
  task mode, single-turn controlled output, mixed delegation
  patterns, human-in-the-loop, and task agents as workflow nodes
