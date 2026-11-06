"""Every ```python block in docs/*.md runs (blocks of one file share a namespace, in order)."""
import os
import re

import pytest

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
FILES = sorted(os.path.relpath(os.path.join(d, f), DOCS) for d, _, fs in os.walk(DOCS) for f in fs if f.endswith(".md"))


@pytest.mark.parametrize("name", FILES)
def test_doc_code_runs(name, tmp_path, monkeypatch):
    from cl_toolkit import heads, merging, routing, strategies
    text = open(os.path.join(DOCS, name)).read()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    if not blocks:
        pytest.skip("no python blocks")
    monkeypatch.chdir(tmp_path)                        # anything a block writes lands here
    # the guides register their examples by name; keep that out of the other tests
    registries = (strategies.STRATEGIES, merging.METHODS, heads.HEADS, routing.ROUTERS, routing.EXPERTS)
    snapshots = [dict(r) for r in registries]
    ns = {"__name__": "__docs__"}
    try:
        for i, code in enumerate(blocks):
            try:
                exec(compile(code, f"{name}[block {i}]", "exec"), ns)
            except Exception as e:                     # noqa: BLE001
                raise AssertionError(f"{name}, python block {i} failed:\n{code}") from e
    finally:
        for r, snap in zip(registries, snapshots):
            r.clear()
            r.update(snap)
