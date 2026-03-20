---
name: adk-style
description: ADK development style guide — architecture patterns, testing best practices, Python idioms, and codebase conventions. Use when writing code, tests, or reviewing PRs for the ADK project. Triggers on "write tests", "best practice", "code style", "how should I", "convention", "pattern", "review code".
---

# ADK Development Style Guide

## Testing

### Core Principles

- **Test through the public interface** — call what users call, assert what users see.
- **Test behavior, not implementation** — verify outcomes (outputs, side effects, errors), not internal mechanics.
- **Refactor-proof** — if an internal refactor preserves the same behavior, all tests should still pass.

### Rules

### 1. Test names describe the behavior, not the mechanism

```python
# Good — describes what the caller observes
def test_empty_queue_returns_none():
def test_retry_stops_after_max_attempts():
def test_missing_key_raises_key_error():

# Bad — describes implementation details
def test_deque_popleft_called():
def test_retry_counter_incremented():
def test_dict_getitem_raises():
```

### 2. One-line docstring describes the expected behavior

```python
# Good — what the caller observes
"""Getting from an empty cache returns the default value."""
"""Tasks added after shutdown are rejected."""
"""Serializing a circular reference raises ValueError."""

# Bad — restates the implementation
"""LRUCache._store.get returns sentinel when key missing."""
"""ThreadPool._accept_tasks flag checked in submit()."""
"""json.dumps calls _check_circular on each dict."""
```

### 3. Each test covers one behavior

If a test checks multiple unrelated behaviors, split it. If you can't
describe the test in one sentence, it's testing too much.

```python
# Bad — tests capacity AND eviction AND default in one test
def test_cache_behavior():
    assert cache.size == 0
    assert cache.get('x') is None
    cache.put('a', 1)
    assert cache.size == 1

# Good — split into focused tests
def test_new_cache_is_empty():
    """A freshly created cache has no entries."""

def test_cache_evicts_oldest_when_full():
    """Adding to a full cache removes the least recently used entry."""
```

### 4. Don't test internal state

```python
# Bad — reaches into private attributes
assert pool._workers[0].is_alive
assert parser._state == 'HEADER'
assert isinstance(router._handler, _FastHandler)

# Good — tests through the public interface
assert pool.active_count == 1
assert parser.parse('data') == expected
assert router.route('/api') == handler
```

### 5. Use real components, mock only boundaries

ADK tests should use real implementations as much as possible
instead of mocking.

- Mock external dependencies: LLM APIs, cloud services, session stores
- Use real ADK components: BaseNode subclasses, Event, Context
- Mock InvocationContext when testing NodeRunner (it's a boundary)

### 6. Test fixtures should be minimal

Define the simplest possible setup that triggers the behavior:

```python
# Good — minimal fixture, one purpose
def make_user(role='viewer'):
    return User(name='test', email='t@t.com', role=role)

# Bad — kitchen-sink fixture with unrelated setup
def make_full_test_env():
    db = create_database()
    user = create_user_with_billing()
    setup_notifications()
    ...
```

### 7. Assertions tell a story

```python
# Good — reads like a specification
assert queue.size == 0
assert config.get('timeout') == 30
assert response.status_code == 404

# Bad — overly defensive, tests framework behavior
assert isinstance(queue, Queue)
assert hasattr(config, 'get')
assert len(response.headers) > 0
```

### Test Structure Template

```python
"""Tests for <ComponentName>.

Verifies that <component> correctly <high-level behavior>.
"""

# --- Fixtures (minimal, one purpose each) ---

def _make_service():
    ...

# --- Tests (one behavior per test) ---

def test_<behavior_description>():
    """<One sentence: what the system does from the outside.>"""
    # Arrange
    service = _make_service()

    # Act
    result = service.do_something(input)

    # Assert
    assert result == expected
```

---

## Architecture Patterns

TODO: BaseNode `@final run()` + `_run_impl()` pattern, NodeRunner,
orchestrator contract, event_author, checkpoint/resume lifecycle.

---

## Python Idioms

TODO: Pydantic patterns (Field, PrivateAttr, model_post_init),
`from __future__ import annotations`, TYPE_CHECKING imports,
async generator conventions.

---

## Codebase Conventions

See the project style guide in the repo root for full details. Key points:
- Relative imports in `src/`, absolute imports in `tests/`
- 2-space indent, 80-char lines, pyink formatter
- `from __future__ import annotations` in every source file
- One class per file in `_engine/` and `workflow/`
