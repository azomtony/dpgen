"""Standalone FP attempt wrapper: allocation identity, GPU filtering and evidence."""

import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HARDWARE_PATTERNS = (
    r"uncorrectable\s+(?:ecc|memory)\s+error",
    r"uncorrectable\s+ecc",
    r"cudaErrorECCUncorrectable",
    r"GPU\s+(?:has\s+)?fallen\s+off\s+the\s+bus",
    r"GPU\s+is\s+lost",
    r"double[- ]bit\s+ECC\s+error",
)


UNLIMITED_RETRY_CATEGORIES = ("hardware", "gpu_initialization")


def classify(text, scheduler_state=""):
    if re.split(r"[ +]", scheduler_state.upper())[0] == "NODE_FAIL":
        return "hardware", "Slurm reports NODE_FAIL"
    for pattern in HARDWARE_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            return "hardware", match.group(0)
    # A context-creation failure precedes the scientific calculation. Error 999
    # is not proof of a broken physical card (compute-mode/driver failures can
    # cause it too), so retry without adding any device to the fault registry.
    if re.search(
        r"\bcuCtxCreate(?:_v\d+)?\b[^\n]*(?:CUDA_ERROR_UNKNOWN|(?:error\s+|returned\s+)999\b)",
        text,
        re.I,
    ):
        return (
            "gpu_initialization",
            "CUDA context creation failed with error 999; physical GPU fault is unconfirmed",
        )
    lower = text.lower()
    if "out of memory" in lower or "out_of_memory" in lower or "oom-kill" in lower:
        return "resource", "Out of memory; not proof of faulty hardware"
    if "preempt" in lower or scheduler_state.startswith("PREEMPT"):
        return "preemption", "Scheduler preemption"
    if scheduler_state.startswith("TIMEOUT") or "time limit" in lower:
        return "timeout", "Wall-time limit"
    for phrase in (
        "command not found",
        "permission denied",
        "no such file or directory",
    ):
        if phrase in lower:
            return "configuration", phrase
    if "illegal memory access" in lower:
        return (
            "application",
            "CUDA illegal memory access; hardware fault is not established",
        )
    if "ediff" in lower or "convergen" in lower or "brmix" in lower or "zhegv" in lower:
        return "application", "Electronic convergence or application error"
    return "unknown", "No confirmed hardware evidence; inspect captured diagnostics"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def input_digest(root=".", names=None):
    digest = hashlib.sha256()
    for name in (
        names if names is not None else ("POSCAR", "INCAR", "POTCAR", "KPOINTS")
    ):
        path = Path(root) / name
        digest.update(name.encode())
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for item in paths:
            if item.is_file():
                if path.is_dir():
                    digest.update(str(item.relative_to(path)).encode())
                with item.open("rb") as stream:
                    for block in iter(lambda: stream.read(1048576), b""):
                        digest.update(block)
    return digest.hexdigest()


def gpu_inventory(run=subprocess.run):
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,mig.mode.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        rows = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) in (3, 4):
                    row = dict(index=parts[0], uuid=parts[1], pci_bus_id=parts[2])
                    if len(parts) == 4:
                        row["mig_mode"] = parts[3]
                    rows.append(row)
        return rows, result.stderr.strip() or (
            result.stdout.strip() if result.returncode else ""
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)


def pci_identity(value):
    """Normalize CUDA's four-digit and NVML's eight-digit PCI domains."""
    match = re.fullmatch(
        r"([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)", value.strip(), re.I
    )
    return tuple(int(part, 16) for part in match.groups()) if match else None


def cuda_visible_uuids(environment, inventory=None):
    """Resolve allocation-scoped CUDA ordinals through PCI IDs, never NVML indices."""
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if not visible or visible == "-1":
        return (
            None,
            "No allocation-scoped CUDA_VISIBLE_DEVICES; GPU identity is unknown",
        )
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens and all(token.startswith(("GPU-", "MIG-")) for token in tokens):
        return tokens, ""
    if any(not token.isdigit() for token in tokens):
        return (
            None,
            "Mixed or unsupported CUDA device mask; preserving the scheduler mask",
        )
    inventory = gpu_inventory()[0] if inventory is None else inventory
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    errors = []
    for library in candidates:
        if not library:
            continue
        try:
            runtime = ctypes.CDLL(library)
            runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            runtime.cudaGetDeviceCount.restype = ctypes.c_int
            runtime.cudaDeviceGetPCIBusId.argtypes = [
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
                ctypes.c_int,
            ]
            runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
            count = ctypes.c_int()
            status = runtime.cudaGetDeviceCount(ctypes.byref(count))
            if status:
                return (
                    None,
                    "uncorrectable ECC error"
                    if status == 214
                    else f"CUDA runtime enumeration failed with code {status}",
                )
            if count.value != len(tokens):
                return (
                    None,
                    "CUDA-visible device count differs from the scheduler mask; identity is unknown",
                )
            identities = []
            for device in range(count.value):
                bus = ctypes.create_string_buffer(32)
                status = runtime.cudaDeviceGetPCIBusId(bus, len(bus), device)
                if status:
                    return None, f"CUDA PCI bus query failed with code {status}"
                pci = pci_identity(bus.value.decode("ascii"))
                matches = [
                    row
                    for row in inventory
                    if pci is not None and pci_identity(row["pci_bus_id"]) == pci
                ]
                if len(matches) != 1:
                    return (
                        None,
                        "CUDA PCI bus ID has no unique match in the NVML inventory",
                    )
                row = matches[0]
                # A physical UUID would widen a numeric MIG allocation to the
                # entire card. Require a known non-MIG configuration for this
                # fallback; explicit MIG UUIDs are handled above.
                if row.get("mig_mode") not in (
                    "Disabled",
                    "N/A",
                    "[N/A]",
                    "Not Supported",
                ):
                    return (
                        None,
                        "PCI lookup cannot safely distinguish MIG instances; preserving the scheduler mask",
                    )
                identities.append(row["uuid"])
            if len(set(identities)) != len(identities):
                return (
                    None,
                    "CUDA devices map to a shared physical GPU; preserving the scheduler mask",
                )
            return identities, ""
        except (OSError, AttributeError) as exc:
            errors.append(f"{library}: {exc}")
    return (
        None,
        "CUDA runtime PCI query unavailable; preserving the scheduler mask. "
        + "; ".join(errors[-3:]),
    )


def allocated_gpus(inventory, environment, visible_uuids=None):
    """Resolve CUDA-visible UUIDs; Slurm/NVML numeric indices can differ."""
    if visible_uuids is None:
        visible = environment.get("CUDA_VISIBLE_DEVICES", "")
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if tokens and all(token.startswith(("GPU-", "MIG-")) for token in tokens):
            visible_uuids = tokens
        else:
            return []
    return [
        {
            "uuid": identity,
            **next((row for row in inventory if row["uuid"] == identity), {}),
        }
        for identity in visible_uuids
    ]


def choose_devices(allocated, bad_uuids):
    return [gpu["uuid"] for gpu in allocated if gpu["uuid"] not in set(bad_uuids)]


def run_attempt(
    config_file="fp-recovery.json", run=subprocess.run, popen=subprocess.Popen
):
    config = json.loads(Path(config_file).read_text())
    fingerprint = input_digest(names=config.get("input_files"))
    environment = dict(os.environ)
    inventory, inventory_error = gpu_inventory(run)
    visible_uuids, mapping_error = cuda_visible_uuids(environment, inventory)
    allocated = allocated_gpus(inventory, environment, visible_uuids)
    report = {
        "started": time.time(),
        "job_id": environment.get("SLURM_JOB_ID"),
        "partition": environment.get("SLURM_JOB_PARTITION"),
        "node": environment.get("SLURMD_NODENAME", environment.get("HOSTNAME")),
        "input_digest": fingerprint,
        "allocated_gpus": allocated,
        "gpu_inventory": inventory,
        "gpu_identity_source": "explicit_uuid_mask"
        if visible_uuids
        and all(
            token.strip().startswith(("GPU-", "MIG-"))
            for token in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        )
        else "cuda_pci_bus_id"
        if visible_uuids
        else "unresolved",
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": environment.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": environment.get("SLURM_STEP_GPUS"),
        "gpu_query_error": inventory_error,
        "gpu_mapping_error": mapping_error,
        "status": "running",
    }
    # The controller sets this only after parsing the exact local results and
    # hashing unchanged inputs. It is not inferred from a scheduler exit code.
    if config.get("validated_input_digest") == fingerprint and Path("OUTCAR").exists():
        report.update(
            status="cached_success",
            category="success",
            reason="Previously validated unchanged inputs",
            exit_code=0,
        )
        atomic_json("fp-attempt.json", report)
        return 0
    healthy = choose_devices(allocated, config.get("bad_gpu_uuids", []))
    if allocated and not healthy:
        report.update(
            status="failed",
            category="hardware",
            reason="Every allocated GPU is in the confirmed-fault registry",
            exit_code=75,
        )
        atomic_json("fp-attempt.json", report)
        print(report["reason"], flush=True)
        return 75
    if allocated:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(healthy)
    report["selected_gpu_uuids"] = healthy
    atomic_json("fp-attempt.json", report)
    # Archive the preceding attempt's diagnostics rather than overwrite evidence.
    if Path("fp-runtime.log").exists():
        archive = Path("fp-attempts")
        archive.mkdir(exist_ok=True)
        os.replace("fp-runtime.log", archive / f"{time.time_ns()}.log")
    with Path("fp-runtime.log").open("w") as log:
        process = popen(
            config["command"],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            executable="/bin/bash",
            start_new_session=False,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        if hasattr(process.stdout, "close"):
            process.stdout.close()
    text = Path("fp-runtime.log").read_text(errors="replace")
    if code:
        category, reason = classify(
            text + "\n" + inventory_error + "\n" + mapping_error
        )
        status = "failed"
    else:
        category, reason, status = (
            "success",
            "Application exited successfully; controller must validate scientific results",
            "exited",
        )
    report.update(
        status=status,
        category=category,
        reason=reason,
        exit_code=code,
        finished=time.time(),
        evidence=text[-16000:],
    )
    # Only a single allocated/selected physical UUID can be attributed from a
    # generic hardware error. Multi-GPU errors do not identify the guilty card.
    if category == "hardware" and len(healthy) == 1:
        report["faulty_gpu_uuids"] = healthy
    atomic_json("fp-attempt.json", report)
    return code


if __name__ == "__main__":
    sys.exit(run_attempt())
