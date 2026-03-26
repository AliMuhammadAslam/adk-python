# ADK Architecture Reference

## BaseNode: `@final run()` + `_run_impl()`

Every node follows a two-method pattern:

- `run()` is `@final` — handles lifecycle (context setup, event
  emission, error handling). Never override.
- `_run_impl()` is the extension point — subclasses implement their
  logic here as an async generator that yields outputs.

```python
class MyNode(BaseNode):
    async def _run_impl(self, *, ctx, node_input):
        result = do_work(node_input)
        yield result  # becomes the node's output event
```

**Why this split:** `run()` guarantees consistent lifecycle behavior
(event creation, state flushing, error wrapping) regardless of what
the subclass does. The subclass only thinks about its domain logic.

## Runner vs NodeRunner vs Workflow

These three are deliberately separate:

- **Runner** = lifecycle orchestrator (InvocationContext, session,
  plugins, invocation boundaries)
- **NodeRunner** = task scheduler (asyncio tasks, node execution,
  completions)
- **Workflow** = graph engine (edges, traversal, node sequencing)

Merging Runner and NodeRunner would deadlock on nested workflows
(inner workflow's NodeRunner would block the outer's Runner).

## Event authoring

Nodes emit events by yielding from `_run_impl()`. The `run()` wrapper
normalizes yields: raw values become `Event(output=value)`, `None` is
skipped, `RequestInput` becomes an interrupt Event. Key fields:

- `event.output` carries the node's result value
- `event.actions.state_delta` carries state mutations
- `event.long_running_tool_ids` signals an interrupt to the caller
- `event.node_info.path` identifies the emitting node in the tree

## Checkpoint and resume lifecycle

HITL (Human-in-the-Loop) follows this pattern:

1. **Interrupt**: Node yields an event with `long_running_tool_ids`.
   Each ancestor propagates the interrupt upward via its own event.
2. **Persist**: Only the leaf node's interrupt event is persisted to
   session. Workflow-level `_adk_internal` events are NOT persisted.
3. **Resume**: User sends a `FunctionResponse` message. The Runner
   scans session events to find the matching `invocation_id`, then
   reconstructs node state from persisted events.
4. **Continue**: The interrupted node receives the FR and continues
   execution. Downstream nodes receive the resumed node's output.

## Async generator conventions

Nodes yield results via async generators:

```python
async def _run_impl(self, *, ctx, node_input):
    # Yield zero or more intermediate results
    yield intermediate

    # Yield exactly one final output (for most nodes)
    yield final_result
```

- A node that yields nothing produces no output event
- Most nodes yield exactly once (the output)
- Workflow nodes may yield multiple times (one per child completion)
