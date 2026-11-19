"""dpdispatcher batch type ``Pilot``: run dpgen jobs on the run-wide node pool.

Use it for training and exploration in machine.json::

    "train": [{
        "command": "dp",
        "machine": {"batch_type": "Pilot", "context_type": "LazyLocalContext",
                    "local_root": "./"},
        "resources": {"number_node": 1, "cpu_per_node": 8, "gpu_per_node": 1,
                      "queue_name": "pool", "group_size": 1,
                      "source_list": ["/path/to/deepmd/env.sh"]}
    }]

together with a top-level ``"pool"`` section that describes the allocations
(see :mod:`dpgen.dispatcher.node_pool`). Each dispatcher job becomes one entry
in the pool. It runs inside an allocation, in its usual directory, with
``CUDA_VISIBLE_DEVICES`` limited to ``gpu_per_node`` GPUs, so four training
models or four exploration jobs share one 4-GPU node side by side. Nothing
is staged or copied: ``LazyLocalContext`` is required.
"""

import re
import time
from pathlib import Path

from dpdispatcher.machine import Machine
from dpdispatcher.utils.job_status import JobStatus

from dpgen.dispatcher import node_pool

# Seconds between pool passes. dpdispatcher asks for every job's status on
# each check; one pass serves all of them.
STEP_EVERY = 10

STAGES = {"00.train": "train", "01.model_devi": "model_devi", "02.fp": "fp"}


class Pilot(Machine):
    """Hand each job to an allocation of the dpgen node pool."""

    slurm_factory = None  # tests replace it; default: the real Slurm CLI

    def gen_script_header(self, job):
        return "#!/bin/bash -l\n"

    # ---- helpers ---------------------------------------------------------

    def _workdir(self):
        return Path(self.context.temp_local_root)

    def _pool(self):
        from dpgen.dispatcher.fp_slurm import Slurm

        root = node_pool.pool_root(self._workdir())
        slurm = (Pilot.slurm_factory or Slurm)()
        return node_pool.NodePool(
            root, node_pool.load_config(root), slurm, workdir=self._workdir()
        )

    def _stage(self):
        here = Path(self.context.remote_root)
        kind = STAGES.get(here.name) or re.sub(r"^\d+\.", "", here.name) or "other"
        match = re.fullmatch(r"iter\.(\d+)", here.parent.name)
        label = f"i{int(match[1]):03d}" if match else here.parent.name
        return kind, label

    @staticmethod
    def _describe(job):
        paths = [str(task.task_work_path) for task in job.job_task_list]
        if not paths:
            return job.job_hash[:8]
        more = f" +{len(paths) - 1}" if len(paths) > 1 else ""
        return paths[0].rstrip("/") + more

    # ---- dpdispatcher interface -----------------------------------------

    def do_submit(self, job):
        if job.resources.strategy.get("if_cuda_multi_devices"):
            raise RuntimeError(
                "batch_type Pilot gives each job its own GPUs; remove "
                "resources.strategy.if_cuda_multi_devices"
            )
        if self.context.__class__.__name__ != "LazyLocalContext":
            raise RuntimeError(
                "batch_type Pilot requires context_type LazyLocalContext"
            )
        self.context.write_file(
            fname=job.script_file_name, write_str=self.gen_script(job)
        )
        self.context.write_file(
            fname=f"{job.script_file_name}.run", write_str=self.gen_script_command(job)
        )
        pool = self._pool()
        kind, label = self._stage()
        prefix = f"{label}.{kind}.{job.job_hash[:8]}."
        number = 1 + sum(1 for key in pool.items if key.startswith(prefix))
        key = f"{prefix}{number}"
        ngpus = max(1, int(job.resources.gpu_per_node or 0))
        description = f"{label} {kind} {self._describe(job)}"
        pool.enqueue(
            key,
            kind=kind,
            cwd=self.context.remote_root,
            argv=["bash", job.script_file_name],
            ngpus=ngpus,
            log=f"{key}.pilot.log",
            label=description,
        )
        pool.event(f"{description}: queued as {key} ({ngpus} GPU)")
        pool.step_items(pool.slurm.queue())
        return key

    def check_status(self, job):
        key = job.job_id
        if not key:
            return JobStatus.unsubmitted
        pool = self._pool()
        if time.time() - pool.state.get("last_step", 0) >= STEP_EVERY:
            pool.step_items(pool.slurm.queue())
        item = pool.items.get(str(key))
        if item is None:
            return JobStatus.terminated
        status = item["status"]
        if status in ("waiting", "assigned"):
            return JobStatus.waiting
        if status == "running":
            return JobStatus.running
        if status == "finished" and item.get("exit_code") == 0:
            if self.check_finish_tag(job):
                return JobStatus.finished
        return JobStatus.terminated

    def check_finish_tag(self, job):
        return self.context.check_file_exists(job.job_hash + "_job_tag_finished")

    def kill(self, job):
        if not job.job_id:
            return
        pool = self._pool()
        pool.drop(str(job.job_id), "killed by dpdispatcher")
        pool.save()
