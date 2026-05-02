#!/usr/bin/env python3
"""Run a trace-enabled Fermi OpenCL sweep and summarize the useful fields."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import hashlib
import os
import pty
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROMPT = "Answer in one sentence: what is OpenCL?"
OPS = [
    "ADD",
    "MUL",
    "RMS_NORM",
    "MUL_MAT",
    "VIEW",
    "PERMUTE",
    "GET_ROWS",
    "SET_ROWS",
    "ROPE",
    "FLASH_ATTN_EXT",
    "GLU",
]
TRANSFER_OPS = [
    ("h2d", "NONE"),
    ("d2h", "RMS_NORM"),
    ("d2h", "MUL_MAT"),
    ("d2h", "VIEW"),
    ("d2h", "PERMUTE"),
]

TPS_RE = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt>[0-9.]+)\s*t/s\s*\|\s*Generation:\s*(?P<gen>[0-9.]+)\s*t/s\s*\]"
)
FINAL_RE = re.compile(
    r"legacy trace final summary graphs=(?P<graphs>\d+) nodes=(?P<nodes>\d+) "
    r"supports=\[queries=(?P<support_queries>\d+),accepted=(?P<support_accepted>\d+),rejected=(?P<support_rejected>\d+)\] "
    r"kernels=(?P<kernels>\d+) buffers=\[count=(?P<buffer_count>\d+),bytes=(?P<buffer_bytes>\d+)\] "
    r"tensors=(?P<tensors>\d+) transfers=\[h2d=(?P<h2d_count>\d+)/(?P<h2d_bytes>\d+)B,"
    r"d2h=(?P<d2h_count>\d+)/(?P<d2h_bytes>\d+)B\] "
    r"sync_other=(?:\[calls=(?P<sync_calls>\d+),waits=(?P<sync_waits>\d+),skipped=(?P<sync_skipped>\d+)\]|(?P<sync_legacy>\d+)) "
    r"finishes=(?P<finishes>\d+)"
)
OP_RE = re.compile(
    r"legacy trace op-summary op=(?P<op>\S+) "
    r"supports=\[accepted=(?P<accepted>\d+),rejected=(?P<rejected>\d+)\] "
    r"compute=\[nodes=(?P<nodes>\d+),failed=(?P<failed>\d+),kernels=(?P<kernels>\d+)\]"
)
TRANSFER_OP_RE = re.compile(
    r"legacy trace transfer-op-summary direction=(?P<direction>\S+) op=(?P<op>\S+) "
    r"count=(?P<count>\d+) bytes=(?P<bytes>\d+) avg=(?P<avg>\d+)"
)
FINISH_RE = re.compile(r"legacy trace finish-summary reason=(?P<reason>\S+) count=(?P<count>\d+)")
GPU_MEM_RE = re.compile(
    r"- GPUOpenCL .*?\|\s*(?P<total>\d+)\s*=\s*(?P<free>\d+)\s*\+\s*"
    r"\(\s*(?P<self_total>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*(?P<context>\d+)\s*\+\s*(?P<compute>\d+)\)\s*\+\s*(?P<unaccounted>\d+)"
)
HOST_MEM_RE = re.compile(
    r"- Host\s+\|\s*(?P<total>\d+)\s*=\s*(?P<model>\d+)\s*\+\s*(?P<context>\d+)\s*\+\s*(?P<compute>\d+)"
)
PLATFORM_RE = re.compile(r"ggml_opencl: selected platform: '(?P<value>.*)'")
DEVICE_RE = re.compile(r"ggml_opencl: device: '(?P<value>.*)'")
DRIVER_RE = re.compile(r"ggml_opencl: OpenCL driver: (?P<value>.*)")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NGL_LOG_RE = re.compile(r"opencl-ngl-(?P<ngl>\d+)\.log$")
NGL_CMD_RE = re.compile(r"(?:^|\s)-ngl\s+(?P<ngl>\d+)(?:\s|$)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_text(cmd: list[str], cwd: Path, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ngl(values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def int_fields(match: re.Match[str]) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {}
    for key, value in match.groupdict().items():
        if value is None:
            continue
        parsed[key] = int(value) if value.isdigit() else value
    return parsed


def parse_log(text: str) -> dict[str, object]:
    data: dict[str, object] = {}
    ops: dict[str, dict[str, int]] = {}
    transfers: dict[tuple[str, str], dict[str, int]] = {}
    finishes_by_reason: dict[str, int] = {}

    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if match := TPS_RE.search(line):
            data["prompt_tps"] = float(match.group("prompt"))
            data["generation_tps"] = float(match.group("gen"))
        if match := FINAL_RE.search(line):
            data.update(int_fields(match))
            if "sync_legacy" in data:
                data["sync_calls"] = data.pop("sync_legacy")
                data["sync_waits"] = ""
                data["sync_skipped"] = ""
        if match := OP_RE.search(line):
            ops[match.group("op")] = {
                "accepted": int(match.group("accepted")),
                "rejected": int(match.group("rejected")),
                "nodes": int(match.group("nodes")),
                "failed": int(match.group("failed")),
                "kernels": int(match.group("kernels")),
            }
        if match := TRANSFER_OP_RE.search(line):
            transfers[(match.group("direction"), match.group("op"))] = {
                "count": int(match.group("count")),
                "bytes": int(match.group("bytes")),
                "avg": int(match.group("avg")),
            }
        if match := FINISH_RE.search(line):
            finishes_by_reason[match.group("reason")] = int(match.group("count"))
        if match := GPU_MEM_RE.search(line):
            for key, value in int_fields(match).items():
                data[f"gpu_{key}_mib"] = value
        if match := HOST_MEM_RE.search(line):
            for key, value in int_fields(match).items():
                data[f"host_{key}_mib"] = value
        if match := PLATFORM_RE.search(line):
            data["opencl_platform"] = match.group("value")
        if match := DEVICE_RE.search(line):
            data["opencl_device"] = match.group("value")
        if match := DRIVER_RE.search(line):
            data["opencl_driver"] = match.group("value")

    data["ops"] = ops
    data["transfer_ops"] = transfers
    data["finishes_by_reason"] = finishes_by_reason
    data["throughput_parsed"] = "yes" if "generation_tps" in data else "no"
    data["completed"] = "yes" if "support_rejected" in data and "finishes" in data else "no"
    return data


def build_command(args: argparse.Namespace, ngl: int) -> list[str]:
    cmd = [
        str(args.binary),
        "-fit",
        "off",
        "--device",
        "GPUOpenCL",
        "-m",
        str(args.model),
        "-p",
        args.prompt,
        "-c",
        str(args.context),
        "-n",
        str(args.tokens),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "--single-turn",
        "--reasoning",
        "off",
        "-ngl",
        str(ngl),
    ]
    if args.nkvo:
        cmd.insert(-2, "-nkvo")
    cmd.extend(args.extra_arg)
    return cmd


def command_with_env(env_vars: dict[str, str], cmd: list[str]) -> str:
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_vars.items())
    rendered = shlex.join(cmd)
    return f"{env_prefix} {rendered}".strip()


def run_command(
    cmd: list[str],
    env_vars: dict[str, str],
    cwd: Path,
    log_path: Path,
    stream: bool,
    use_pty: bool,
) -> tuple[int, float]:
    if use_pty:
        return run_command_pty(cmd, env_vars, cwd, log_path, stream)
    return run_command_pipe(cmd, env_vars, cwd, log_path, stream)


def run_command_pipe(
    cmd: list[str],
    env_vars: dict[str, str],
    cwd: Path,
    log_path: Path,
    stream: bool,
) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(env_vars)
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if stream:
                print(line, end="", flush=True)
    return proc.wait(), time.monotonic() - started


def run_command_pty(
    cmd: list[str],
    env_vars: dict[str, str],
    cwd: Path,
    log_path: Path,
    stream: bool,
) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(env_vars)
    started = time.monotonic()
    pid, master_fd = pty.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            os.execvpe(cmd[0], cmd, env)
        except Exception as exc:
            os.write(2, f"exec failed: {exc}\n".encode("utf-8", errors="replace"))
            os._exit(127)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            log.write(text)
            log.flush()
            if stream:
                sys.stdout.write(text)
                sys.stdout.flush()

    _, status = os.waitpid(pid, 0)
    returncode = os.waitstatus_to_exitcode(status)
    os.close(master_fd)
    return returncode, time.monotonic() - started


def table_row(result: dict[str, object], op: str, field: str) -> object:
    ops = result.get("ops", {})
    if not isinstance(ops, dict):
        return ""
    value = ops.get(op, {})
    if isinstance(value, dict):
        return value.get(field, "")
    return ""


def transfer_field(result: dict[str, object], direction: str, op: str, field: str) -> object:
    transfers = result.get("transfer_ops", {})
    if not isinstance(transfers, dict):
        return ""
    value = transfers.get((direction, op), {})
    if isinstance(value, dict):
        return value.get(field, "")
    return ""


def finish_reason(result: dict[str, object], reason: str) -> object:
    finishes = result.get("finishes_by_reason", {})
    if isinstance(finishes, dict):
        return finishes.get(reason, "")
    return ""


def summary_columns() -> list[str]:
    cols = [
        "ngl",
        "returncode",
        "completed",
        "throughput_parsed",
        "prompt_tps",
        "generation_tps",
        "opencl_platform",
        "opencl_device",
        "opencl_driver",
        "gpu_total_mib",
        "gpu_free_mib",
        "gpu_model_mib",
        "gpu_context_mib",
        "gpu_compute_mib",
        "host_model_mib",
        "host_context_mib",
        "graphs",
        "nodes",
        "support_queries",
        "support_accepted",
        "support_rejected",
        "kernels",
        "buffer_count",
        "buffer_bytes",
        "tensors",
        "h2d_count",
        "h2d_bytes",
        "d2h_count",
        "d2h_bytes",
        "sync_calls",
        "sync_waits",
        "sync_skipped",
        "finishes",
        "finish_synchronize",
        "finish_buffer_clear",
    ]
    for op in OPS:
        cols.extend([f"{op}_accepted", f"{op}_rejected", f"{op}_nodes", f"{op}_kernels"])
    for direction, op in TRANSFER_OPS:
        cols.extend([f"{direction}_{op}_count", f"{direction}_{op}_bytes"])
    cols.extend(["elapsed_s", "log"])
    return cols


def result_row(result: dict[str, object]) -> dict[str, object]:
    row = {col: "" for col in summary_columns()}
    for col in row:
        if col in result:
            row[col] = result[col]
    row["finish_synchronize"] = finish_reason(result, "synchronize")
    row["finish_buffer_clear"] = finish_reason(result, "buffer-clear")
    for op in OPS:
        row[f"{op}_accepted"] = table_row(result, op, "accepted")
        row[f"{op}_rejected"] = table_row(result, op, "rejected")
        row[f"{op}_nodes"] = table_row(result, op, "nodes")
        row[f"{op}_kernels"] = table_row(result, op, "kernels")
    for direction, op in TRANSFER_OPS:
        row[f"{direction}_{op}_count"] = transfer_field(result, direction, op, "count")
        row[f"{direction}_{op}_bytes"] = transfer_field(result, direction, op, "bytes")
    return row


def write_tsv(path: Path, results: list[dict[str, object]]) -> None:
    cols = summary_columns()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result_row(result))


def md_value(value: object) -> str:
    return "" if value is None else str(value)


def write_summary_md(path: Path, metadata: dict[str, str], results: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fermi OpenCL Sweep Summary\n\n")
        f.write("## Metadata\n\n")
        for key, value in metadata.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## Results\n\n")
        f.write(
            "| `-ngl` | completed | throughput parsed | rc | prompt t/s | gen t/s | GPU model MiB | GPU context MiB | "
            "support rejects | kernels | H2D bytes | D2H bytes | finishes | log |\n"
        )
        f.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for result in results:
            f.write(
                "| "
                f"`{md_value(result.get('ngl'))}` | "
                f"{md_value(result.get('completed'))} | "
                f"{md_value(result.get('throughput_parsed'))} | "
                f"{md_value(result.get('returncode'))} | "
                f"{md_value(result.get('prompt_tps'))} | "
                f"{md_value(result.get('generation_tps'))} | "
                f"{md_value(result.get('gpu_model_mib'))} | "
                f"{md_value(result.get('gpu_context_mib'))} | "
                f"{md_value(result.get('support_rejected'))} | "
                f"{md_value(result.get('kernels'))} | "
                f"{md_value(result.get('h2d_bytes'))} | "
                f"{md_value(result.get('d2h_bytes'))} | "
                f"{md_value(result.get('finishes'))} | "
                f"`{md_value(result.get('log'))}` |\n"
            )

        f.write("\n## Op Coverage\n\n")
        f.write("| `-ngl` | " + " | ".join(f"`{op}` rejected/nodes" for op in OPS) + " |\n")
        f.write("| ---: | " + " | ".join("---:" for _ in OPS) + " |\n")
        for result in results:
            parts = []
            for op in OPS:
                rejected = table_row(result, op, "rejected")
                nodes = table_row(result, op, "nodes")
                parts.append(f"`{rejected}` / `{nodes}`")
            f.write(f"| `{md_value(result.get('ngl'))}` | " + " | ".join(parts) + " |\n")

        f.write("\n## Transfer Attribution\n\n")
        f.write("| `-ngl` | " + " | ".join(f"`{direction} {op}` count/bytes" for direction, op in TRANSFER_OPS) + " |\n")
        f.write("| ---: | " + " | ".join("---:" for _ in TRANSFER_OPS) + " |\n")
        for result in results:
            parts = []
            for direction, op in TRANSFER_OPS:
                count = transfer_field(result, direction, op, "count")
                nbytes = transfer_field(result, direction, op, "bytes")
                parts.append(f"`{count}` / `{nbytes}`")
            f.write(f"| `{md_value(result.get('ngl'))}` | " + " | ".join(parts) + " |\n")

        f.write("\n## Commands\n\n")
        for result in results:
            f.write(f"### `-ngl {md_value(result.get('ngl'))}`\n\n")
            f.write("```bash\n")
            f.write(str(result.get("command", "")).strip() + "\n")
            f.write("```\n\n")


def write_metadata(path: Path, metadata: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}={shlex.quote(value)}\n")


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parts = shlex.split(value)
        metadata[key] = parts[0] if parts else ""
    return metadata


def read_existing_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            try:
                rows[int(row.get("ngl", ""))] = row
            except ValueError:
                continue
    return rows


def read_commands(path: Path) -> dict[int, str]:
    commands: dict[int, str] = {}
    if not path.exists():
        return commands
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = NGL_CMD_RE.search(line)
        if match:
            commands[int(match.group("ngl"))] = line
    return commands


def summarize_existing_run(run_dir: Path) -> int:
    logs_dir = run_dir / "logs"
    metadata = read_metadata(run_dir / "metadata.env")
    old_rows = read_existing_rows(run_dir / "summary.tsv")
    commands = read_commands(run_dir / "commands.sh")
    results: list[dict[str, object]] = []

    for log_path in sorted(logs_dir.glob("opencl-ngl-*.log"), key=lambda p: int(NGL_LOG_RE.search(p.name).group("ngl")) if NGL_LOG_RE.search(p.name) else -1):
        match = NGL_LOG_RE.search(log_path.name)
        if not match:
            continue
        ngl = int(match.group("ngl"))
        text = log_path.read_text(encoding="utf-8", errors="replace")
        result = parse_log(text)
        old = old_rows.get(ngl, {})
        result.update(
            {
                "ngl": ngl,
                "returncode": old.get("returncode", ""),
                "elapsed_s": old.get("elapsed_s", ""),
                "log": str(log_path.relative_to(run_dir)),
                "command": commands.get(ngl, ""),
            }
        )
        if result.get("returncode") not in ("", "0", 0):
            result["completed"] = "no"
        results.append(result)

    write_tsv(run_dir / "summary.tsv", results)
    write_summary_md(run_dir / "summary.md", metadata, results)
    print(f"Regenerated {run_dir / 'summary.md'}")
    print(f"Regenerated {run_dir / 'summary.tsv'}")
    for result in results:
        print(
            "  -ngl {ngl}: completed={completed} throughput_parsed={throughput} gen={gen} tok/s "
            "rejects={rejects} d2h={d2h_count}/{d2h_bytes}B".format(
                ngl=result.get("ngl", ""),
                completed=result.get("completed", ""),
                throughput=result.get("throughput_parsed", ""),
                gen=result.get("generation_tps", ""),
                rejects=result.get("support_rejected", ""),
                d2h_count=result.get("d2h_count", ""),
                d2h_bytes=result.get("d2h_bytes", ""),
            )
        )
    return 0


def write_system_state(path: Path, root: Path) -> None:
    commands = [
        ("date", ["date", "--iso-8601=seconds"]),
        ("uname", ["uname", "-a"]),
        ("nvidia-smi", ["nvidia-smi"]),
        ("dkms status", ["dkms", "status"]),
        ("clinfo brief", ["clinfo"]),
    ]
    with path.open("w", encoding="utf-8") as f:
        for title, cmd in commands:
            f.write(f"## {title}\n\n")
            output = run_text(cmd, root, timeout=20)
            f.write(output if output else "<unavailable>")
            f.write("\n\n")


def make_metadata(args: argparse.Namespace, root: Path, run_dir: Path, ngl_values: list[int]) -> dict[str, str]:
    now = dt.datetime.now().astimezone()
    model = args.model if args.model.is_absolute() else root / args.model
    metadata = {
        "run_date": now.isoformat(timespec="seconds"),
        "timezone": now.tzname() or "",
        "host": socket.gethostname(),
        "repo": str(root),
        "run_dir": str(run_dir),
        "kernel": run_text(["uname", "-r"], root),
        "parent_commit": run_text(["git", "rev-parse", "--short", "HEAD"], root),
        "llama_cpp_commit": run_text(["git", "-C", "third_party/llama.cpp", "rev-parse", "--short", "HEAD"], root),
        "model_file": str(args.model),
        "model_sha256": sha256_file(model) if model.exists() and not args.no_hash else "",
        "prompt": args.prompt,
        "context_size": str(args.context),
        "generation_tokens": str(args.tokens),
        "batch": str(args.batch),
        "ubatch": str(args.ubatch),
        "ngl_values": ",".join(str(v) for v in ngl_values),
        "kv_offload": "disabled (-nkvo)" if args.nkvo else "enabled",
        "output_cpu": "false" if args.no_output_cpu else "true",
        "trace": "false" if args.no_trace else "true",
        "capture_pty": "false" if args.no_pty else "true",
    }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/llama.cpp-opencl-native/bin/llama-cli"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-0.6B-Q4_0.gguf"))
    parser.add_argument("--ngl", nargs="+", default=["2", "3", "4", "8", "16"], help="Layer counts, e.g. --ngl 2 3 4 or --ngl 2,3,4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--nkvo", action="store_true", help="Disable KV offload for comparison runs")
    parser.add_argument("--no-output-cpu", action="store_true", help="Do not set LLAMA_FERMI_OPENCL_OUTPUT_CPU=1")
    parser.add_argument("--no-trace", action="store_true", help="Do not set GGML_OPENCL_NVIDIA_LEGACY_TRACE=1")
    parser.add_argument("--no-hash", action="store_true", help="Skip hashing the model file")
    parser.add_argument("--no-stream", action="store_true", help="Do not stream llama-cli output to the terminal")
    parser.add_argument("--no-pty", action="store_true", help="Capture with a pipe instead of a pseudo-terminal")
    parser.add_argument("--summarize-existing", type=Path, help="Rebuild summary files from an existing run directory")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue to the next -ngl after a failed run")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and commands without executing llama-cli")
    parser.add_argument("--extra-arg", action="append", default=[], help="Append one extra llama-cli argument; repeat as needed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.summarize_existing:
        run_dir = args.summarize_existing
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        return summarize_existing_run(run_dir)

    if not args.binary.is_absolute():
        args.binary = root / args.binary
    if not args.model.is_absolute():
        args.model = root / args.model

    ngl_values = parse_ngl(args.ngl)
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S")
    run_dir = args.run_dir or root / "results" / "runs" / f"{timestamp}-opencl-fermi-sweep"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_vars: dict[str, str] = {"LC_ALL": "C"}
    if not args.no_trace:
        env_vars["GGML_OPENCL_NVIDIA_LEGACY_TRACE"] = "1"
    if not args.no_output_cpu:
        env_vars["LLAMA_FERMI_OPENCL_OUTPUT_CPU"] = "1"

    metadata = make_metadata(args, root, run_dir, ngl_values)
    write_metadata(run_dir / "metadata.env", metadata)
    write_system_state(run_dir / "system-state.txt", root)

    results: list[dict[str, object]] = []
    commands_path = run_dir / "commands.sh"
    with commands_path.open("w", encoding="utf-8") as commands_file:
        commands_file.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for ngl in ngl_values:
            cmd = build_command(args, ngl)
            rendered = command_with_env(env_vars, cmd)
            commands_file.write(rendered + "\n")
    commands_path.chmod(0o755)

    for ngl in ngl_values:
        log_path = logs_dir / f"opencl-ngl-{ngl}.log"
        cmd = build_command(args, ngl)
        rendered = command_with_env(env_vars, cmd)
        print(f"\n== -ngl {ngl} ==")
        print(rendered)

        if args.dry_run:
            result: dict[str, object] = {
                "ngl": ngl,
                "returncode": "",
                "completed": "dry-run",
                "elapsed_s": "",
                "log": str(log_path.relative_to(run_dir)),
                "command": rendered,
            }
        else:
            returncode, elapsed = run_command(cmd, env_vars, root, log_path, not args.no_stream, not args.no_pty)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            result = parse_log(text)
            result.update(
                {
                    "ngl": ngl,
                    "returncode": returncode,
                    "elapsed_s": f"{elapsed:.1f}",
                    "log": str(log_path.relative_to(run_dir)),
                    "command": rendered,
                }
            )
            if returncode != 0:
                result["completed"] = "no"
            if result.get("throughput_parsed") != "yes":
                print(f"warning: throughput line was not parsed for -ngl {ngl}; see {log_path}")

        results.append(result)
        write_tsv(run_dir / "summary.tsv", results)
        write_summary_md(run_dir / "summary.md", metadata, results)

        if not args.dry_run and result.get("returncode") not in (0, "0") and not args.continue_on_failure:
            print(f"Stopping after -ngl {ngl} failed with return code {result.get('returncode')}.")
            break

    print("\nWrote:")
    print(f"  {run_dir / 'summary.md'}")
    print(f"  {run_dir / 'summary.tsv'}")
    print(f"  {logs_dir}")

    if results:
        print("\nCompact summary:")
        for result in results:
            print(
                "  -ngl {ngl}: completed={completed} rc={rc} gen={gen} tok/s "
                "rejects={rejects} d2h={d2h_count}/{d2h_bytes}B kernels={kernels}".format(
                    ngl=result.get("ngl", ""),
                    completed=result.get("completed", ""),
                    rc=result.get("returncode", ""),
                    gen=result.get("generation_tps", ""),
                    rejects=result.get("support_rejected", ""),
                    d2h_count=result.get("d2h_count", ""),
                    d2h_bytes=result.get("d2h_bytes", ""),
                    kernels=result.get("kernels", ""),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
