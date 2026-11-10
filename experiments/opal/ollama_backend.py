"""Local Ollama callable for OPAL; no OPAL source changes or paid services.

The AST gate reduces accidental access outside the optimizer contract. It is not
a security sandbox: OPAL executes accepted Python in the training process.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


FILES = {"updater_py": "updater.py", "analysis_py": "analysis.py"}
SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "At most 200 words: evidence, change or keep, and prediction."},
        "updater_py": {"type": ["string", "null"], "description": "Use null to KEEP the current updater; do not reprint unchanged code. Only for an actual optimizer change, return the COMPLETE module including top-level build(named_params, previous=None, spec=None)."},
        "analysis_py": {"type": ["string", "null"], "description": "Use null to KEEP the current analysis; do not reprint or repair existing upstream analysis. Only for an intentional diagnostic change, return the COMPLETE module including analyze(obs, memory)."},
    },
    "required": ["rationale", "updater_py", "analysis_py"],
    "additionalProperties": False,
}
BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "input", "breakpoint", "globals",
    "locals", "vars", "dir", "help", "print", "exit", "quit", "setattr",
    "delattr", "memoryview", "__import__", "__builtins__", "__loader__",
    "__spec__", "__file__", "__name__", "builtins", "os", "sys", "gc",
    "inspect", "pathlib", "subprocess", "socket", "requests", "urllib",
    "pickle", "marshal", "importlib", "ctypes", "multiprocessing",
    "threading", "asyncio", "getattr", "hasattr",
}
BLOCKED_ATTRS = {
    "load", "save", "load_state_dict_from_url", "from_file", "storage",
    "untyped_storage", "set_default_device", "set_default_dtype",
    "set_default_tensor_type", "set_num_threads", "set_num_interop_threads",
    "manual_seed", "seed", "initial_seed", "set_rng_state", "get_rng_state",
    "use_deterministic_algorithms", "set_grad_enabled", "compile", "jit",
    "hub", "distributed", "multiprocessing", "serialization", "package",
    "profiler", "backends", "cuda", "mps", "xpu", "classes", "ops",
    "utils", "data", "numpy", "frombuffer", "write", "read", "format",
    "format_map", "system", "popen", "environ", "modules", "eval", "exec",
}
TORCH_TOP = {
    "optim", "nn", "no_grad", "Tensor", "is_tensor", "tensor", "as_tensor",
    "zeros_like", "ones_like", "full_like", "empty_like", "zeros", "ones",
    "full", "empty", "eye", "arange", "stack", "cat", "vstack", "hstack",
    "lerp", "sign", "norm", "linalg", "clamp", "clamp_min", "clamp_max",
    "maximum", "minimum", "where", "sqrt", "rsqrt", "square", "pow",
    "exp", "log", "log1p", "isfinite", "isnan", "isinf", "abs", "mean",
    "sum", "amax", "amin", "min", "max", "quantile", "finfo", "float32",
    "float64", "float16", "bfloat16", "int64", "bool", "device", "div",
    "mul", "add", "sub", "dot", "einsum", "matmul", "bmm", "all", "any",
}
BUILTIN_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "super",
    "property", "staticmethod", "classmethod", "Exception", "ValueError",
    "TypeError", "RuntimeError", "NotImplementedError", "AssertionError",
}


def validate_source(source: str, filename: str) -> None:
    """Reject invalid code and obvious filesystem/network/process capabilities."""
    if not isinstance(source, str) or not source.strip() or len(source) > 40000:
        raise ValueError(f"{filename}: expected nonempty Python source of at most 40000 characters")
    tree = ast.parse(source, filename=filename)
    imported, torch_aliases = set(), set()
    definitions = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"torch", "math", "copy"}:
                    raise ValueError(f"{filename}: only imports torch, math, copy are allowed")
                imported.add(alias.asname or alias.name)
                if alias.name == "torch":
                    torch_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in {"math", "copy", "torch", "torch.optim"}:
                raise ValueError(f"{filename}: unsupported from-import {node.module}")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    raise ValueError(f"{filename}: wildcard/private imports are forbidden")
                if node.module == "torch" and alias.name not in TORCH_TOP:
                    raise ValueError(f"{filename}: unsupported torch import {alias.name}")
                if node.module == "copy" and alias.name not in {"copy", "deepcopy"}:
                    raise ValueError(f"{filename}: unsupported copy import")
                if node.module == "torch.optim" and alias.name not in {"AdamW", "Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "NAdam", "RAdam"}:
                    raise ValueError(f"{filename}: unsupported torch optimizer")
                imported.add(alias.asname or alias.name)
    allowed_calls = BUILTIN_CALLS | definitions | imported
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.While, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"{filename}: unsupported construct {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise ValueError(f"{filename}: forbidden name {node.id}")
        if isinstance(node, ast.Attribute):
            # Standard torch clipping is the sole allowed `.utils` path.
            clipping_utils = node.attr == "utils" and isinstance(node.value, ast.Attribute) and node.value.attr == "nn" and isinstance(node.value.value, ast.Name) and node.value.value.id in torch_aliases
            private_foreach = node.attr.startswith("_foreach_") and isinstance(node.value, ast.Name) and node.value.id in torch_aliases
            if (node.attr in BLOCKED_ATTRS and not clipping_utils) or (node.attr.startswith("_") and not private_foreach):
                raise ValueError(f"{filename}: forbidden attribute {node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases and node.attr not in TORCH_TOP and not private_foreach:
                raise ValueError(f"{filename}: unsupported torch operation {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_calls:
            raise ValueError(f"{filename}: unsupported direct call {node.func.id}")
        if isinstance(node, ast.FunctionDef) and node.name.startswith("__") and node.name != "__init__":
            raise ValueError(f"{filename}: unsupported special method {node.name}")
    required = "build" if filename == "updater.py" else "analyze"
    if not any(isinstance(n, ast.FunctionDef) and n.name == required for n in tree.body):
        raise ValueError(f"{filename}: missing top-level {required} function")
    compile(tree, filename, "exec")


def atomic_write(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path.name}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class OllamaBackend:
    """Use with opal.wrap(..., backend=OllamaBackend(record_root=...), summarize=False)."""

    def __init__(self, record_root, model="gemma4:e2b-qat-google", base_url="http://127.0.0.1:11434", timeout=180, retries=1, num_ctx=32768, num_predict=4096, think=False, temperature=0.2):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
            raise ValueError("This adapter only permits an unauthenticated local Ollama HTTP endpoint")
        if not 1 <= timeout <= 900 or not 0 <= retries <= 2:
            raise ValueError("timeout must be 1..900 seconds; retries must be 0..2")
        self.record_root = Path(record_root)
        self.record_root.mkdir(parents=True, exist_ok=True)
        self.model, self.url = model, base_url.rstrip("/") + "/api/chat"
        self.timeout, self.retries = timeout, retries
        self.num_ctx, self.num_predict = num_ctx, num_predict
        if not isinstance(think, bool) and think not in {'low', 'medium', 'high', 'max'}:
            raise ValueError('think must be a boolean or a supported reasoning level')
        self.think, self.temperature = think, float(temperature)
        self.history = []
        self.calls = 0
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, workspace, message, system):
        workspace = Path(workspace)
        self.calls += 1
        record = self.record_root / f"decision-{self.calls:03d}"
        record.mkdir(parents=True, exist_ok=False)
        current = {name: (workspace / name).read_text() for name in FILES.values()}
        source_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in current.items()}
        (record / "context_meta.json").write_text(json.dumps({"source_sha256_before": source_hashes, "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
        if sum(map(len, current.values())) > 45000:
            raise ValueError("Current source exceeds the local adapter context limit")
        instructions = (
            "You are the optimizer coding agent in OPAL. The user selected a local model. "
            "Return exactly one JSON object: rationale (string), updater_py (complete Python source or null), "
            "analysis_py (complete Python source or null). KEEP MEANS NULL: when keeping the optimizer, return "
            "updater_py: null; when keeping diagnostics, return analysis_py: null. Never reprint unchanged files. "
            "Keeping both is valid, and should produce {\"rationale\": \"brief reason\", \"updater_py\": null, \"analysis_py\": null}. "
            "Keep your rationale to at most 200 words. "
            "Do not claim edits or measurements that you did not make. Do not add files. "
            "The current source below is authoritative. Existing upstream code is already trusted and may stay "
            "unchanged, even if it uses APIs that are not allowed in newly generated code. Do not repair unchanged "
            "upstream diagnostics just to satisfy the new-code gate: return analysis_py: null. "
            "Preserve optimizer/scheduler state on changes. Only if you propose an actual new optimizer, return its "
            "complete module, including def build(named_params, previous=None, spec=None) at module level; "
            "an Updater class without build is invalid. "
            "The whole optimizer algorithm is yours; changes are optional and must follow the observed evidence. "
            "Only imports torch, math, copy (or from math/copy/torch/torch.optim) are allowed. "
            "No filesystem, network, processes, introspection, serialization, randomness, global framework settings, "
            "getattr/hasattr, private attributes, while loops, or parameter/gradient dtype/device changes. "
            "Use no_grad for updates and only plain checkpoint state. build must not mutate shared state. "
            "State rationale as observations, changed/kept, prediction, and fixed versus feedback rules.\n\n"
            + (system or "")[:26000]
            + "\n\nLOCAL ADAPTER RESPONSE CONTRACT: Return only one JSON object with rationale (<=200 words), "
            "updater_py (complete changed module or null), analysis_py (complete changed module or null). "
            "There are no editing tools: changed code must be in these JSON strings. KEEP = null, never reprint "
            "unchanged code. An actual optimizer proposal must include top-level build(named_params, previous=None, "
            "spec=None). Existing upstream diagnostics may remain unchanged despite the new-code restrictions. "
            "Keep example: {\"rationale\":\"...\",\"updater_py\":null,\"analysis_py\":null}"
        )
        history = json.dumps(self.history[-3:], ensure_ascii=False)[-7000:]
        content = "REMINDER: To KEEP any file, return null; do not reprint it. Keep rationale to <=200 words.\n\nRecent decisions:\n" + history + "\n\nCurrent updater.py:\n" + current["updater.py"] + "\n\nCurrent analysis.py:\n" + current["analysis.py"] + "\n\nLatest training feedback:\n" + message[-16000:] + "\n\nFINAL OUTPUT CHECK: KEEP = null. Do not repeat or repair unchanged upstream analysis. Any actual optimizer change must include the complete top-level build function. Return only the JSON object, with rationale <=200 words.\nKeep example: {\"rationale\":\"...\",\"updater_py\":null,\"analysis_py\":null}"
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": content}]
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            body = {"model": self.model, "messages": messages, "stream": False, "think": self.think, "format": SCHEMA, "options": {"temperature": self.temperature, "seed": 0, "num_ctx": self.num_ctx, "num_predict": self.num_predict}}
            (record / f"request-{attempt + 1}.json").write_text(json.dumps(body, indent=2))
            t0 = time.monotonic()
            response = None
            try:
                request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with self.http.open(request, timeout=self.timeout) as reply:
                    raw = reply.read(2000000)
                response = json.loads(raw)
                (record / f"response-{attempt + 1}.json").write_text(json.dumps(response, indent=2))
                text = response["message"]["content"]
                proposal = json.loads(text)
                if not isinstance(proposal, dict) or set(proposal) != set(SCHEMA["properties"]) or not isinstance(proposal["rationale"], str):
                    raise ValueError("Return exactly rationale (string), updater_py and analysis_py (source strings or null)")
                changes = {}
                validation_errors = []
                for key, filename in FILES.items():
                    if proposal[key] is None or proposal[key] == current[filename]:
                        continue
                    try:
                        validate_source(proposal[key], filename)
                        changes[filename] = proposal[key]
                    except (ValueError, SyntaxError) as exc:
                        validation_errors.append(str(exc))
                if validation_errors:
                    raise ValueError("All proposed-file errors: " + "; ".join(validation_errors) + ". For any unchanged file use null.")
                # All sources passed before either file changes. Restore originals on a write failure.
                try:
                    for filename, source in changes.items():
                        atomic_write(workspace / filename, source)
                except BaseException:
                    for filename in changes:
                        atomic_write(workspace / filename, current[filename])
                    raise
                result = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "model": self.model, "seconds": time.monotonic() - started, "attempts": attempt + 1, "changed": sorted(changes), "rationale": proposal["rationale"], "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count"), "cost_usd": 0.0, "accepted_by_adapter": True, "source_sha256_before": source_hashes, "source_sha256_after": {name: hashlib.sha256((workspace / name).read_bytes()).hexdigest() for name in current}}
                (record / "result.json").write_text(json.dumps(result, indent=2))
                self.history.append({"decision": self.calls, "changed": result["changed"], "rationale": proposal["rationale"][:2200]})
                return proposal["rationale"]
            except (ValueError, KeyError, TypeError, SyntaxError, OSError, urllib.error.URLError) as exc:
                failure = {"error": f"{type(exc).__name__}: {exc}", "attempt_seconds": time.monotonic() - t0, "accepted_by_adapter": False}
                (record / f"error-{attempt + 1}.json").write_text(json.dumps(failure, indent=2))
                if attempt == self.retries:
                    raise RuntimeError(f"Local Ollama decision {self.calls} failed after {attempt + 1} attempts: {exc}") from exc
                if response and isinstance(response.get("message", {}).get("content"), str):
                    messages += [{"role": "assistant", "content": response["message"]["content"][:18000]}, {"role": "user", "content": f"The proposal was rejected before applying any code: {exc}. Correct it, or keep files with null. Return the exact JSON schema."}]
                time.sleep(1)
