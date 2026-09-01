# Python conventions

Opinions specific to this project. For anything not listed here, follow the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and match the surrounding code.

## Environment

`uv` is the only package manager.

- `uv add <pkg>` / `uv remove <pkg>` — never hand-edit `[project.dependencies]`
- `uv add --dev <pkg>` for test and tooling dependencies
- `uv run <cmd>` for everything: tests, linters, scripts. Never activate a venv, never call bare `python` or `pip`.
- `uv sync` after pulling
- `uv.lock` is committed

## Imports

Prefer `from x import y` over `import x`, including for individual classes and functions.

> This deliberately overrides Google §2.2.4, which restricts `from` imports to modules. Where the two conflict, this file wins.

- Absolute paths only — no relative imports (`from .sibling import ...`)
- One import per line, except `typing` and `collections.abc`, which may combine
- Grouped `__future__` → stdlib → third-party → first-party, sorted within each group
- `import x as y` only for established abbreviations (`import numpy as np`)

## Docstrings

Google style, with three hard rules.

**1. The prose is at most two lines.** A summary line, plus at most one more line of context. Anything longer belongs in a comment beside the code it explains, in the module docstring, or in `docs/`. This is the rule most likely to be violated — check it explicitly before finishing a file.

**2. The only permitted sections are `Args:`, `Returns:` (or `Yields:`), `Raises:`, and `Attributes:`.** `Note:` is forbidden. So are `Example:`, `Examples:`, `Todo:`, `See Also:`, and any other invented heading. If something feels like it needs a `Note:`, it is either a code comment or it doesn't need saying.

**3. A docstring that restates the signature doesn't earn its place.** Omit it. Methods decorated `@override` need no docstring unless they materially change the base contract.

```python
def resolve_manifest(root: Path, *, strict: bool = False) -> Manifest:
    """Loads and validates the manifest under root.

    Overlay files in the same directory are merged in lexical order.

    Args:
        root: Directory containing manifest.toml.
        strict: If True, unknown keys raise instead of being dropped.

    Returns:
        The merged manifest with defaults applied.

    Raises:
        ManifestError: If the manifest is missing or fails validation.
    """
```

Not this:

```python
def resolve_manifest(root: Path, *, strict: bool = False) -> Manifest:
    """Loads and validates the manifest under root.

    This function walks the given directory looking for a manifest.toml file.
    Once found, it parses the TOML and applies the schema. It then looks for
    any overlay files, which are additional TOML files that override keys in
    the base manifest, and merges them in lexical order so that later files
    take precedence over earlier ones.

    Note:
        The strict flag was added in v2 and changes error behaviour.

    Example:
        >>> resolve_manifest(Path("./cfg"))
    """
```

Module docstrings follow the same two-line limit: one summary line, one optional line of context.

## Types and errors

- Annotate public APIs. `X | None`, never `Optional[X]`.
- Prefer `collections.abc.Sequence` / `Mapping` over `list` / `dict` in signatures.
- Raise specific exceptions. No bare `except:`, and no `except Exception:` unless re-raising.
- No mutable default arguments.
- f-strings everywhere except logging calls, which use lazy `%` formatting.
- Line length 88. Set it in `pyproject.toml` so it isn't a discussion.

## Verification

These must pass before any work item is marked done:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Adjust to match the project's actual tooling. Keep the list short enough that it gets run every time.
