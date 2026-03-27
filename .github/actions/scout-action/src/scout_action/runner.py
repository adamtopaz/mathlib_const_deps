from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

HF_HUB_SPEC = "huggingface_hub==1.8.0"
DELETE_GLOB = "part-*.parquet"
TARGET_FLAGS = {
    "imports": "--imports",
    "library": "--library",
    "read": "--read",
}


class ActionError(RuntimeError):
    """Raised for user-facing action failures."""


@dataclass(frozen=True)
class Inputs:
    hf_repo_id: str
    hf_token: str
    command: str
    target_kind: str
    target_values: tuple[str, ...]
    project_root: Path
    parallel: int
    num_shards: int
    batch_rows: int
    config_json: str
    config: Any
    private: bool
    commit_message: str
    dry_run: bool


@dataclass(frozen=True)
class Outputs:
    data_dir: Path
    parquet_count: int
    dataset_url: str
    readme_path: Path
    scout_command: str


def parse_bool(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ActionError(f"invalid boolean value: {value!r}")



def parse_target_values(raw: str) -> tuple[str, ...]:
    values = tuple(line.strip() for line in raw.splitlines() if line.strip())
    if not values:
        raise ActionError("target_values must contain at least one non-empty line")
    return values



def resolve_project_root(project_root: str, *, cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    path = Path(project_root)
    if not path.is_absolute():
        path = base / path
    return path.resolve()



def ensure_lake_project(project_root: Path) -> None:
    if not project_root.exists():
        raise ActionError(f"project_root does not exist: {project_root}")
    if not project_root.is_dir():
        raise ActionError(f"project_root is not a directory: {project_root}")
    if not ((project_root / "lakefile.lean").exists() or (project_root / "lakefile.toml").exists()):
        raise ActionError(
            f"project_root does not look like a Lake project (missing lakefile.lean/toml): {project_root}"
        )



def ensure_lake_available() -> None:
    if shutil.which("lake") is None:
        raise ActionError(
            "`lake` was not found on PATH. Run your Lean CI/setup step before this action "
            "(for example, lean-action or an equivalent setup/build step)."
        )



def ensure_runtime_tools(*, dry_run: bool) -> None:
    if shutil.which("uv") is None:
        raise ActionError(
            "`uv` was not found on PATH. Install uv before invoking this action."
        )
    if not dry_run and shutil.which("uvx") is None:
        raise ActionError(
            "`uvx` was not found on PATH. Install uvx before invoking this action, "
            "or set dry_run=true to skip upload."
        )



def parse_json_config(config_json: str) -> Any:
    try:
        return json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ActionError(f"config_json is not valid JSON: {exc}") from exc



def require_int(name: str, raw: str, *, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ActionError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ActionError(f"{name} must be >= {minimum}, got {value}")
    return value



def default_commit_message(explicit: str, github_sha: str) -> str:
    value = explicit.strip()
    if value:
        return value
    short_sha = github_sha[:12] if github_sha else "unknown"
    return f"Update dataset from {short_sha}"



def load_inputs(env: Mapping[str, str], *, cwd: Path | None = None) -> Inputs:
    workspace_root = Path(env.get("GITHUB_WORKSPACE", "")).resolve() if env.get("GITHUB_WORKSPACE") else None
    project_root = resolve_project_root(
        env.get("SCOUT_ACTION_PROJECT_ROOT", "."),
        cwd=workspace_root or cwd,
    )
    ensure_lake_project(project_root)
    ensure_lake_available()

    dry_run = parse_bool(env.get("SCOUT_ACTION_DRY_RUN", "false"))
    ensure_runtime_tools(dry_run=dry_run)

    hf_repo_id = env.get("SCOUT_ACTION_HF_REPO_ID", "").strip()
    if not hf_repo_id:
        raise ActionError("hf_repo_id is required")

    command = env.get("SCOUT_ACTION_COMMAND", "").strip()
    if not command:
        raise ActionError("command is required")

    target_kind = env.get("SCOUT_ACTION_TARGET_KIND", "").strip()
    if target_kind not in TARGET_FLAGS:
        valid = ", ".join(sorted(TARGET_FLAGS))
        raise ActionError(f"target_kind must be one of: {valid}")

    target_values = parse_target_values(env.get("SCOUT_ACTION_TARGET_VALUES", ""))
    if target_kind == "library" and len(target_values) != 1:
        raise ActionError("target_kind=library requires exactly one target value")

    config_json = env.get("SCOUT_ACTION_CONFIG_JSON", "{}").strip() or "{}"
    config = parse_json_config(config_json)

    parallel = require_int("parallel", env.get("SCOUT_ACTION_PARALLEL", "1"))
    num_shards = require_int("num_shards", env.get("SCOUT_ACTION_NUM_SHARDS", "128"))
    batch_rows = require_int("batch_rows", env.get("SCOUT_ACTION_BATCH_ROWS", "1024"))
    private = parse_bool(env.get("SCOUT_ACTION_PRIVATE", "false"))

    hf_token = env.get("SCOUT_ACTION_HF_TOKEN", "").strip()
    if not hf_token and not dry_run:
        raise ActionError("hf_token is required unless dry_run=true")

    commit_message = default_commit_message(
        env.get("SCOUT_ACTION_COMMIT_MESSAGE", ""),
        env.get("GITHUB_SHA", ""),
    )

    return Inputs(
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        command=command,
        target_kind=target_kind,
        target_values=target_values,
        project_root=project_root,
        parallel=parallel,
        num_shards=num_shards,
        batch_rows=batch_rows,
        config_json=config_json,
        config=config,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )



def build_scout_command(inputs: Inputs, data_dir: Path) -> list[str]:
    cmd = [
        "lake",
        "run",
        "scout",
        "--command",
        inputs.command,
        "--parquet",
        "--dataDir",
        str(data_dir),
        "--cmdRoot",
        str(inputs.project_root),
        "--parallel",
        str(inputs.parallel),
        "--numShards",
        str(inputs.num_shards),
        "--batchRows",
        str(inputs.batch_rows),
        "--config",
        inputs.config_json,
        TARGET_FLAGS[inputs.target_kind],
        *inputs.target_values,
    ]
    return cmd



def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=capture_output,
            check=False,
        )
    except OSError as exc:
        rendered = shlex.join(list(cmd))
        raise ActionError(f"failed to execute command: {rendered} ({exc})") from exc



def run_scout(inputs: Inputs, data_dir: Path) -> list[Path]:
    cmd = build_scout_command(inputs, data_dir)
    print(f"Running Lean Scout: {shlex.join(cmd)}", flush=True)
    result = run_subprocess(cmd, cwd=inputs.project_root)
    if result.returncode != 0:
        raise ActionError(f"Lean Scout failed with exit code {result.returncode}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise ActionError(f"Lean Scout produced no parquet files in {data_dir}")
    return parquet_files



def try_load_schema(project_root: Path, command: str) -> tuple[Any | None, str | None]:
    cmd = ["lake", "exe", "lean_scout_schema", command]
    result = run_subprocess(cmd, cwd=project_root, capture_output=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        return None, message
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"schema output was not valid JSON: {exc}"



def title_from_repo_id(repo_id: str) -> str:
    slug = repo_id.split("/")[-1]
    words = [part for part in slug.replace("_", "-").split("-") if part]
    return " ".join(word.capitalize() for word in words) or repo_id



def format_json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)



def generate_readme(
    *,
    output_path: Path,
    inputs: Inputs,
    scout_command: Sequence[str],
    schema: Any | None,
    schema_error: str | None,
    source_repo: str,
    source_sha: str,
    generated_at: str,
) -> None:
    title = title_from_repo_id(inputs.hf_repo_id)
    target_lines = "\n".join(f"- `{value}`" for value in inputs.target_values)
    config_block = format_json_block(inputs.config)
    scout_command_block = shlex.join(list(scout_command))

    if schema is None:
        schema_section = (
            "Schema lookup was unavailable for this command.\n\n"
            f"Reason: `{schema_error or 'unknown error'}`"
        )
    else:
        schema_section = f"```json\n{format_json_block(schema)}\n```"

    content = f"""---
tags:
  - lean4
  - theorem-proving
  - lean-scout
---

# {title}

This dataset was generated with [lean_scout](https://github.com/mathlib-initiative/lean_scout)
from the GitHub repository `{source_repo}`.

## Source

- Source repository: `{source_repo}`
- Source commit: `{source_sha or 'unknown'}`
- Hugging Face dataset repo: `{inputs.hf_repo_id}`
- Generated at (UTC): `{generated_at}`

## Extraction details

- Extractor command: `{inputs.command}`
- Target kind: `{inputs.target_kind}`
- Private upload requested: `{str(inputs.private).lower()}`
- Dry run: `{str(inputs.dry_run).lower()}`

### Target values

{target_lines}

### Extractor config

```json
{config_block}
```

### Scout command

```bash
{scout_command_block}
```

## Schema

{schema_section}
"""
    output_path.write_text(content)



def build_upload_command(inputs: Inputs, data_dir: Path) -> list[str]:
    cmd = [
        "uvx",
        "--from",
        HF_HUB_SPEC,
        "hf",
        "upload",
        inputs.hf_repo_id,
        str(data_dir),
        ".",
        "--repo-type",
        "dataset",
        "--delete",
        DELETE_GLOB,
        "--commit-message",
        inputs.commit_message,
    ]
    if inputs.private:
        cmd.append("--private")
    return cmd



def upload_dataset(inputs: Inputs, data_dir: Path) -> None:
    cmd = build_upload_command(inputs, data_dir)
    print(f"Uploading dataset: {shlex.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["HF_TOKEN"] = inputs.hf_token
    result = run_subprocess(cmd, cwd=inputs.project_root, env=env)
    if result.returncode != 0:
        raise ActionError(f"Hugging Face upload failed with exit code {result.returncode}")



def dataset_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"



def write_outputs(outputs: Outputs, *, github_output: str | None) -> None:
    if not github_output:
        return
    output_path = Path(github_output)
    lines = [
        f"data_dir={outputs.data_dir}",
        f"parquet_count={outputs.parquet_count}",
        f"dataset_url={outputs.dataset_url}",
        f"readme_path={outputs.readme_path}",
        f"scout_command={outputs.scout_command}",
    ]
    with output_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")



def execute(inputs: Inputs) -> Outputs:
    data_dir = Path(tempfile.mkdtemp(prefix="scout-action-"))
    scout_cmd = build_scout_command(inputs, data_dir)
    parquet_files = run_scout(inputs, data_dir)

    schema, schema_error = try_load_schema(inputs.project_root, inputs.command)
    readme_path = data_dir / "README.md"
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    generate_readme(
        output_path=readme_path,
        inputs=inputs,
        scout_command=scout_cmd,
        schema=schema,
        schema_error=schema_error,
        source_repo=os.environ.get("GITHUB_REPOSITORY", "unknown"),
        source_sha=os.environ.get("GITHUB_SHA", "unknown"),
        generated_at=generated_at,
    )

    if inputs.dry_run:
        print("dry_run=true, skipping Hugging Face upload", flush=True)
    else:
        upload_dataset(inputs, data_dir)

    return Outputs(
        data_dir=data_dir,
        parquet_count=len(parquet_files),
        dataset_url=dataset_url(inputs.hf_repo_id),
        readme_path=readme_path,
        scout_command=shlex.join(scout_cmd),
    )



def main(argv: Sequence[str] | None = None) -> int:
    _ = argv
    try:
        inputs = load_inputs(os.environ)
        outputs = execute(inputs)
        write_outputs(outputs, github_output=os.environ.get("GITHUB_OUTPUT"))
        print(f"Done. Parquet files: {outputs.parquet_count}", flush=True)
        print(f"Output directory: {outputs.data_dir}", flush=True)
        print(f"Dataset URL: {outputs.dataset_url}", flush=True)
        return 0
    except ActionError as exc:
        print(f"::error::{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
