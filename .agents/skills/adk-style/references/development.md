# ADK Development Conventions

## Public API vs internal methods

### Public API

Public API is the contract with users. Prioritize clarity and
discoverability over implementation convenience.

- **Short, intuitive names.** A user reading the method name should
  immediately understand what it does without consulting the source.
- **Explicit docstrings.** Every public method must have a docstring
  that states what it does, what it returns, and any constraints.
  If the docstring doesn't make the expected usage obvious, clarify
  with the author and improve it before merging.
- **Stable signatures.** Changing a public method's parameters or
  return type is a breaking change. Design the signature carefully
  upfront.

```python
# Good — concise name, clear docstring
def run_async(self, *, user_id, session_id, new_message):
    """Runs one invocation: processes the user message and yields events.

    Args:
      user_id: Identifies the user for session lookup.
      session_id: The session to append events to.
      new_message: The user's input content.

    Yields:
      Event objects as the agent processes the message.
    """

# Bad — vague name, no docstring
def process(self, uid, sid, msg):
    ...
```

### Internal methods

Internal methods (`_`-prefixed) serve the implementation. Prioritize
readability and maintainability over brevity.

- **Self-explanatory names.** Longer names are fine — prefer
  `_resolve_invocation_id_from_fr` over `_resolve_id`. The name
  should make the method's purpose clear without reading its body.
- **Short method bodies.** If a method is getting long, extract
  logical steps into well-named helper methods. Each method should
  do one thing.
- **Flat control flow.** Prefer early returns, `continue`, and
  guard clauses over deeply nested `if/else/try` blocks. Aim for
  no more than 2-3 levels of indentation in the method body.
- **Respect class contracts.** Before using or setting a field, read
  the class definition to understand its intended lifecycle. For
  example, `Event.id` is set by the session service — callers
  should never assign it directly.

```python
# Good — self-explanatory name, flat control flow
def _find_interrupt_event_in_session(self, session, fc_id):
    for event in reversed(session.events):
      if not event.long_running_tool_ids:
        continue
      if fc_id in event.long_running_tool_ids:
        return event
    return None

# Bad — cryptic name, deep nesting
def _find(self, s, fid):
    for e in s.events:
      if e.long_running_tool_ids:
        if fid in e.long_running_tool_ids:
          if e.node_info:
            return e
    return None
```

```python
# Good — respects class contract
event = Event(...)
# event.id is assigned by session service on append
session_service.append_event(session, event)

# Bad — bypasses intended lifecycle
event = Event(...)
event.id = 'my-custom-id'  # id is managed by session service
```

## Comments and docstrings

- Explain *why*, not *what* — the code should be self-documenting
- Don't reference RFCs or design docs in source code
- No double-quoted type hints when `from __future__ import annotations`
  is present — use bare type names

## File organization

- One class per file in `workflow/`
- Private modules prefixed with `_` (e.g., `_base_node.py`)
- Public API exported through `__init__.py`

## Imports

- **Source code** (`src/`): Use relative imports.
  `from ..agents.llm_agent import LlmAgent`
- **Tests** (`tests/`): Use absolute imports.
  `from google.adk.agents.llm_agent import LlmAgent`
- Import from the module file, not from `__init__.py`.
  `from ..agents.llm_agent import LlmAgent` (not `from ..agents import LlmAgent`)

### TYPE_CHECKING imports

Use `TYPE_CHECKING` for imports needed only by type hints to avoid
circular imports at runtime:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.invocation_context import InvocationContext
```

This works because `from __future__ import annotations` makes all
annotations strings (deferred evaluation), so the import is never
needed at runtime.

## Pydantic patterns

ADK models use Pydantic v2. Key patterns:

```python
class MyModel(BaseModel):
    # Public fields with defaults
    name: str
    timeout: int = Field(default=30, ge=1)

    # Private attributes (not serialized, not in schema)
    _cache: dict = PrivateAttr(default_factory=dict)

    # Post-init logic
    def model_post_init(self, __context):
        self._cache = {}
```

- Use `Field()` for validation, defaults, and descriptions
- Use `PrivateAttr()` for internal state that shouldn't be serialized
- Use `model_post_init()` instead of `__init__` for setup logic
- Prefer `model_dump()` over `dict()` (Pydantic v2)

## File headers

Every source file must have:

1. Apache 2.0 license header
2. `from __future__ import annotations`
3. Standard library imports, then third-party, then relative

## Formatting

- 2-space indentation (never tabs)
- 80-character line limit
- `pyink` formatter (Google-style)
- `isort` with Google profile for import sorting
- Enforced automatically by pre-commit hooks (isort, pyink,
  addlicense, mdformat). Install with `pre-commit install`.

```bash
# Format only staged files (runs automatically on commit)
pre-commit run

# Format all changed files (staged + unstaged)
pre-commit run --files $(git diff --name-only HEAD)

# Format all files in the repo
pre-commit run --all-files
```
