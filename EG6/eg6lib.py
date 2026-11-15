from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.paths import get_paths

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG6_CODE_ROOT = REPO_ROOT / "EG6"
LOCAL_RUNTIME_ROOT = REPO_ROOT / "local_runtime" / "detection_baselines"
GIDS_EVAL_ROOT = _PATHS.work_root
OFFICIAL_RQ5_ROOT = _PATHS.eg6_official_rq5_root
DATA_ROOT = _PATHS.sc_data_root("eg6")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
SELECTION_REPORT_ROOT = REPORT_ROOT / "selection"
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
RUN_LABEL = "eg6"
CONDA_ROOT = _PATHS.conda_root
EG2_CACHE_ROOT = _PATHS.sc_data_root("eg2") / "cache" / "experiments"
OFFICIAL_CHECKPOINT_ROOT = DATA_ROOT / "official-checkpoints"

def _eg2_cache(relpath: str) -> Path:
    return EG2_CACHE_ROOT / relpath


def _optional_rate(value: Any) -> float | str:
    return "" if value in ("", None) else float(value)

DETECTORS = ("argus", "euler", "vgrnn")
DATASETS = ("lanl", "optc", "0501", "cic")
DATASET_ALIASES = {
    "wizard": "0501",
    "wizard_spider": "0501",
    "oilrig": "0501",
    "sandworm": "0501",
}
REPRO_LEVELS = {
    "lanl": "paper_repro",
    "optc": "paper_repro",
    "0501": "code_port",
    "cic": "extension",
}

DETECTOR_ENV_PREFIXES = {
    "argus": CONDA_ROOT / "envs/sc1-argus",
    "euler": CONDA_ROOT / "envs/sc1-euler",
    "vgrnn": CONDA_ROOT / "envs/sc1-torch-template",
}

DETECTOR_REQUIRED_IMPORTS = {
    "argus": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "yaml"),
    "euler": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "yaml"),
    "vgrnn": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "yaml"),
}

ARGUS_CACHE_ROOTS = {
    "lanl": _eg2_cache("featurization/147caca457763fa2e425c721c9809eff59ed53efe805bf6a7c41f4bb3cbcd625/output/data_root/argus/lanl"),
    "optc": _eg2_cache("featurization/28b5ba272f16203e911fa53e07ee9ff22a005189df1404872a87706dec301db2/output/data_root/argus/optc"),
    "0501": _eg2_cache("featurization/84b6062cf718cfadcdcc90789ada67f620bd382829705b9e8dd7502f04ba25e5/output/data_root/argus/0501/0501"),
    "cic": _eg2_cache("featurization/45f485c90d3694fdb99a3ef5b5ee2059bea2312c291b42d3b960fd9605f3e08f/output/data_root/argus/cic"),
}

EULER_CACHE_ROOTS = {
    "lanl": _eg2_cache("featurization/800f248fd16c3092c64247ad9c2f61e4651387e5893d7a78be977ba44f117e16/output/data_root/euler/lanl"),
    "optc": _eg2_cache("featurization/f806c46f712895cf037015582a855a3fd2f4298f726a65e6d77147201e2cb00a/output/data_root/euler/optc"),
    "0501": _eg2_cache("featurization/9133a37f02536fd955325cfa84a81ae613ea25a7547056a8dbf214539abc8b0e/output/data_root/euler/0501/0501"),
    "cic": _eg2_cache("featurization/85400ff4e5589c882f7acbaa3eac7d6c1fccf893822bb0abfa074fcb3a8ab0cd/output/data_root/euler/cic"),
}

DEFAULT_BUDGETS = (0, 2, 5, 10, 20, 50)

@dataclass(frozen=True)
class EG6Job:
    detector: str
    dataset: str
    window_seconds: int
    repro_level: str

    @property
    def experiment_id(self) -> str:
        return f"eg6_{self.detector}_{self.dataset}"

    @property
    def config_path(self) -> Path:
        return GENERATED_CONFIG_ROOT / f"{self.experiment_id}.yaml"

    @property
    def workspace_root(self) -> Path:
        return DATA_ROOT / "experiments" / self.experiment_id / RUN_LABEL

    @property
    def runtime_root(self) -> Path:
        return self.workspace_root / "runtime"

    @property
    def metadata_root(self) -> Path:
        return self.workspace_root / "metadata"

    @property
    def logs_root(self) -> Path:
        return self.metadata_root / "logs"

    @property
    def execution_config_path(self) -> Path:
        return self.metadata_root / "execution_config.yaml"

    @property
    def budget_results_path(self) -> Path:
        return self.metadata_root / "budget_results.csv"

    @property
    def target_results_path(self) -> Path:
        return self.metadata_root / "target_results.csv"

    @property
    def attack_progress_path(self) -> Path:
        return self.metadata_root / "attack_progress.json"

    @property
    def attack_summary_path(self) -> Path:
        return self.metadata_root / "attack_summary.json"

    @property
    def run_summary_path(self) -> Path:
        return self.metadata_root / "run_summary.json"

JOB_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "argus": {
        "lanl": {
            "window_seconds": 3600,
            "template": EG6_CODE_ROOT / "templates/argus_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC" / "loaders",
            "data_root": ARGUS_CACHE_ROOTS["lanl"],
            "data_env_var": "ARGUS_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "ARGUS" / "model_save_LANL_ARGUS.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "architecture": "lanl",
                "threshold": 0.7746,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 10000,
                "device": "cpu",
                "loader": {
                    "module": "load_lanl",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 150885,
                        "end": 1270000,
                        "delta": 3600,
                        "is_test": True,
                        "use_flows": True,
                    },
                },
            },
        },
        "optc": {
            "window_seconds": 360,
            "template": EG6_CODE_ROOT / "templates/argus_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "argus" / "LANL_OpTC" / "loaders",
            "data_root": ARGUS_CACHE_ROOTS["optc"],
            "data_env_var": "ARGUS_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "ARGUS" / "model_save_OPTC_ARGUS.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "architecture": "optc",
                "threshold": 0.9919,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_optc",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 573290,
                        "end": 745983,
                        "delta": 360,
                        "is_test": True,
                    },
                },
            },
        },
        "0501": {
            "window_seconds": 150,
            "template": EG6_CODE_ROOT / "templates/argus_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "argus" / "0501",
            "models_source_root": LOCAL_RUNTIME_ROOT / "argus" / "0501" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "argus" / "0501" / "loaders",
            "data_root": ARGUS_CACHE_ROOTS["0501"],
            "data_env_var": "ARGUS_DATA_ROOT",
            "checkpoint": {
                "path": _eg2_cache("detection/5fd1dd8d8e71fb146435763018b7e97e3173c63809f305138e7b8fbb0a5eb926/output/models/0501/model_save_O_0501.pkl"),
                "source": "eg2_cache",
            },
            "attack": {
                "architecture": "optc",
                "threshold": 0.56673336,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 5000,
                "device": "cpu",
                "loader": {
                    "module": "load_eg1_0501",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 72378559,
                        "end": 86674422,
                        "delta": 150000,
                        "is_test": True,
                    },
                },
            },
        },
        "cic": {
            "window_seconds": 900,
            "template": EG6_CODE_ROOT / "templates/argus_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "argus" / "CIC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "argus" / "CIC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "argus" / "CIC" / "loaders",
            "data_root": ARGUS_CACHE_ROOTS["cic"],
            "data_env_var": "ARGUS_DATA_ROOT",
            "checkpoint": {
                "path": _eg2_cache("detection/7de58bf31b4bf0c7ea43b1463a469826a6102adbb53363af6d56436c0bbdaf24/output/models/CIC/model_save_O_cic.pkl"),
                "source": "eg2_cache",
            },
            "attack": {
                "architecture": "optc",
                "threshold": 0.7389372,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_cic",
                    "function": "load_optc_dist",
                    "kwargs": {
                        "start": 29136,
                        "end": 374762,
                        "delta": 900,
                        "is_test": True,
                    },
                },
            },
        },
    },
    "euler": {
        "lanl": {
            "window_seconds": 10800,
            "template": EG6_CODE_ROOT / "templates/euler_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC" / "loaders",
            "data_root": EULER_CACHE_ROOTS["lanl"],
            "data_env_var": "EULER_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "EULER" / "model_save_lanl.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "threshold": 0.5478,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_lanl",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 150885,
                        "end": 5011199,
                        "delta": 10800,
                        "is_test": True,
                    },
                },
            },
        },
        "optc": {
            "window_seconds": 360,
            "template": EG6_CODE_ROOT / "templates/euler_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "euler" / "LANL_OpTC" / "loaders",
            "data_root": EULER_CACHE_ROOTS["optc"],
            "data_env_var": "EULER_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "EULER" / "model_save_optc.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "threshold": 0.5733,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_optc",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 573290,
                        "end": 745983,
                        "delta": 360,
                        "is_test": True,
                    },
                },
            },
        },
        "0501": {
            "window_seconds": 150,
            "template": EG6_CODE_ROOT / "templates/euler_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "euler" / "0501",
            "models_source_root": LOCAL_RUNTIME_ROOT / "euler" / "0501" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "euler" / "0501" / "loaders",
            "data_root": EULER_CACHE_ROOTS["0501"],
            "data_env_var": "EULER_DATA_ROOT",
            "checkpoint": {
                "path": _eg2_cache("detection/a0e7484aef32780a4621b0810231af72f70939c0c97ede889f4416db6562f891/output/results/0501/0501/learning/model_save.pkl"),
                "source": "eg2_cache",
            },
            "attack": {
                "threshold": 0.5470250844955444,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_eg1_0501",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 72378559,
                        "end": 86674422,
                        "delta": 150000,
                        "is_test": True,
                    },
                },
            },
        },
        "cic": {
            "window_seconds": 3000,
            "template": EG6_CODE_ROOT / "templates/euler_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "euler" / "CIC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "euler" / "CIC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "euler" / "CIC" / "loaders",
            "data_root": EULER_CACHE_ROOTS["cic"],
            "data_env_var": "EULER_DATA_ROOT",
            "checkpoint": {
                "path": _eg2_cache("detection/b3a20777a3f68a99b002cc1e8761a8883fd6e8c72e6ad15052d96549b4fcc5f6/output/results/cic/learning/model_save.pkl"),
                "source": "eg2_cache",
            },
            "attack": {
                "threshold": 0.5083739161491394,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "device": "cpu",
                "loader": {
                    "module": "load_cic",
                    "function": "load_partial_lanl",
                    "kwargs": {
                        "start": 29136,
                        "end": 374762,
                        "delta": 3000,
                        "is_test": True,
                    },
                },
            },
        },
    },
    "vgrnn": {
        "lanl": {
            "window_seconds": 5400,
            "template": EG6_CODE_ROOT / "templates/vgrnn_attack.py",
            "runtime_source_root": OFFICIAL_RQ5_ROOT / "VGRNN" / "lanl",
            "models_source_root": OFFICIAL_RQ5_ROOT / "VGRNN" / "lanl" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "LANL_OpTC" / "loaders",
            "data_root": EULER_CACHE_ROOTS["lanl"],
            "data_env_var": "VGRNN_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "VGRNN" / "VGRNN_lanl.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "threshold": 0.541921854019165,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "loader": {
                    "module": "load_lanl",
                    "function": "load_lanl_dist",
                    "kwargs": {
                        "workers": 8,
                        "start": 150885,
                        "end": 1270000,
                        "delta": 5400,
                        "is_test": True,
                    },
                },
            },
        },
        "optc": {
            "window_seconds": 1800,
            "template": EG6_CODE_ROOT / "templates/vgrnn_attack.py",
            "runtime_source_root": OFFICIAL_RQ5_ROOT / "VGRNN" / "optc",
            "models_source_root": OFFICIAL_RQ5_ROOT / "VGRNN" / "optc" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "LANL_OpTC" / "loaders",
            "data_root": EULER_CACHE_ROOTS["optc"],
            "data_env_var": "VGRNN_DATA_ROOT",
            "checkpoint": {
                "path": OFFICIAL_RQ5_ROOT / "VGRNN" / "VGRNN_optc.pkl",
                "source": "official_checkpoint",
            },
            "attack": {
                "threshold": 0.710624635219574,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "loader": {
                    "module": "load_optc",
                    "function": "load_lanl_dist",
                    "kwargs": {
                        "workers": 8,
                        "start": 573290,
                        "end": 745983,
                        "delta": 1800,
                        "is_test": True,
                    },
                },
            },
        },
        "0501": {
            "window_seconds": 150,
            "template": EG6_CODE_ROOT / "templates/vgrnn_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "0501",
            "models_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "0501" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "0501" / "loaders",
            "extra_runtime_files": [OFFICIAL_RQ5_ROOT / "VGRNN" / "0501" / "model_1.py"],
            "data_root": EULER_CACHE_ROOTS["0501"],
            "data_env_var": "VGRNN_DATA_ROOT",
            "checkpoint": {
                "path": None,
                "source": "bootstrap_local",
            },
            "attack": {
                "threshold": None,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "loader": {
                    "module": "load_eg1_0501",
                    "function": "load_lanl_dist",
                    "kwargs": {
                        "workers": 8,
                        "start": 72378559,
                        "end": 86674422,
                        "delta": 150000,
                        "is_test": True,
                    },
                },
            },
            "bootstrap": {
                "source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "0501",
                "dataset_arg": "0501",
                "snapshot_minutes": 2.5,
                "dimension": 32,
                "lr": 0.005,
                "patience": 10,
                "fw": 0.43,
                "workers": 8,
                "epochs": 200,
            },
        },
        "cic": {
            "window_seconds": 1800,
            "template": EG6_CODE_ROOT / "templates/vgrnn_attack.py",
            "runtime_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "CIC",
            "models_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "CIC" / "models",
            "loaders_source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "CIC" / "loaders",
            "extra_runtime_files": [OFFICIAL_RQ5_ROOT / "VGRNN" / "lanl" / "model_1.py"],
            "data_root": EULER_CACHE_ROOTS["cic"],
            "data_env_var": "VGRNN_DATA_ROOT",
            "checkpoint": {
                "path": None,
                "source": "bootstrap_local",
            },
            "attack": {
                "threshold": None,
                "budgets": list(DEFAULT_BUDGETS),
                "n_choice": 100000,
                "loader": {
                    "module": "load_cic",
                    "function": "load_lanl_dist",
                    "kwargs": {
                        "workers": 8,
                        "start": 29136,
                        "end": 374762,
                        "delta": 1800,
                        "is_test": True,
                    },
                },
            },
            "bootstrap": {
                "source_root": LOCAL_RUNTIME_ROOT / "vgrnn" / "CIC",
                "dataset_arg": "CIC",
                "snapshot_minutes": 30,
                "lr": 0.005,
                "patience": 10,
                "workers": 8,
                "epochs": 200,
            },
        },
    },
}

def ensure_eg6_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT, SELECTION_REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_env_prefix(detector: str) -> Path:
    override = os.environ.get(f"EG6_ENV_{detector.upper()}")
    if override:
        return Path(override)
    return DETECTOR_ENV_PREFIXES[detector]

def detector_env_python(detector: str) -> Path:
    return detector_env_prefix(detector) / "bin/python"

def detector_list() -> list[str]:
    return list(DETECTORS)

def dataset_list() -> list[str]:
    return list(DATASETS)

def canonical_dataset(value: str) -> str:
    lowered = value.strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)

def parse_csv_list(value: str | None, *, canonicalize: bool = False) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if canonicalize:
        items = [canonical_dataset(item) for item in items]
        items = list(dict.fromkeys(items))
    return items

def iter_jobs(
    detectors: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
) -> list[EG6Job]:
    selected_detectors = list(detectors or detector_list())
    selected_datasets = list(datasets or dataset_list())
    jobs: list[EG6Job] = []
    for detector in selected_detectors:
        if detector not in JOB_SPECS:
            raise ValueError(f"Unknown detector: {detector}")
        for dataset in selected_datasets:
            if dataset not in JOB_SPECS[detector]:
                continue
            spec = JOB_SPECS[detector][dataset]
            jobs.append(EG6Job(
                detector=detector,
                dataset=dataset,
                window_seconds=int(spec["window_seconds"]),
                repro_level=str(REPRO_LEVELS[dataset]),
            ))
    return jobs

def same_job_set(left: list[EG6Job], right: list[EG6Job]) -> bool:
    return {job.experiment_id for job in left} == {job.experiment_id for job in right}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload

def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload

def _apply_cache_derived_settings(job: EG6Job, spec: dict[str, Any]) -> None:
    if job.dataset != "0501":
        return

    metadata_path = Path(spec["data_root"]) / "eg1_0501_metadata.json"
    if not metadata_path.exists():
        return

    metadata = _read_json(metadata_path)
    first_positive_ms = int(metadata["first_positive_ms"])
    last_ms = int(metadata["last_ms"])
    loader_kwargs = spec["attack"]["loader"]["kwargs"]
    loader_kwargs["start"] = first_positive_ms
    loader_kwargs["end"] = last_ms
    loader_kwargs["delta"] = int(spec["window_seconds"]) * 1000
    spec["cache_metadata"] = {
        "path": metadata_path,
        "first_positive_ms": first_positive_ms,
        "last_ms": last_ms,
    }

def _job_spec(job: EG6Job) -> dict[str, Any]:
    spec = deepcopy(JOB_SPECS[job.detector][job.dataset])
    _apply_cache_derived_settings(job, spec)
    return spec

def generate_config(job: EG6Job) -> Path:
    ensure_eg6_dirs()
    spec = _job_spec(job)
    payload = {
        "name": f"EG6 {job.detector.upper()} {job.dataset}",
        "experiment_id": job.experiment_id,
        "detector": job.detector,
        "dataset": job.dataset,
        "repro_level": job.repro_level,
        "window_seconds": job.window_seconds,
        "checkpoint": {
            "path": str(spec["checkpoint"]["path"]) if spec["checkpoint"]["path"] else "",
            "source": spec["checkpoint"]["source"],
        },
        "attack": deepcopy(spec["attack"]),
        "runtime": {
            "template": str(spec["template"]),
            "runtime_source_root": str(spec["runtime_source_root"]),
            "models_source_root": str(spec["models_source_root"]),
            "loaders_source_root": str(spec["loaders_source_root"]),
            "data_root": str(spec["data_root"]),
            "data_env_var": spec["data_env_var"],
        },
    }
    if "bootstrap" in spec:
        payload["bootstrap"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in spec["bootstrap"].items()
        }
    dump_yaml(job.config_path, payload)
    return job.config_path

def _runtime_data_root(job: EG6Job) -> Path:
    return job.runtime_root / "data_root"

def _bootstrap_root(job: EG6Job) -> Path:
    return job.workspace_root / "bootstrap"

def _ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)

def _stage_data_root(job: EG6Job, spec: dict[str, Any]) -> Path:
    data_root = _runtime_data_root(job)
    data_root.mkdir(parents=True, exist_ok=True)
    source = Path(spec["data_root"])
    if job.dataset == "0501":
        _ensure_symlink(source, data_root / "0501" / "0501")
    else:
        _ensure_symlink(source, data_root / job.dataset)
    return data_root

def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def _stage_runtime_source(job: EG6Job, spec: dict[str, Any]) -> None:
    job.runtime_root.mkdir(parents=True, exist_ok=True)
    staged_paths = [
        job.runtime_root / "models",
        job.runtime_root / "loaders",
        job.runtime_root / "libauc",
        job.runtime_root / "model_1.py",
    ]
    for extra_file in spec.get("extra_runtime_files", []):
        staged_paths.append(job.runtime_root / Path(extra_file).name)
    for path in staged_paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    _copy_tree(Path(spec["models_source_root"]), job.runtime_root / "models")
    _copy_tree(Path(spec["loaders_source_root"]), job.runtime_root / "loaders")
    libauc_source = Path(spec["runtime_source_root"]) / "libauc"
    if libauc_source.exists():
        _copy_tree(libauc_source, job.runtime_root / "libauc")
    model_1_source = Path(spec["runtime_source_root"]) / "model_1.py"
    if model_1_source.exists():
        shutil.copy2(model_1_source, job.runtime_root / "model_1.py")
    for extra_file in spec.get("extra_runtime_files", []):
        src = Path(extra_file)
        shutil.copy2(src, job.runtime_root / src.name)

def _pythonpath_with(parts: list[Path]) -> str:
    current = os.environ.get("PYTHONPATH", "")
    values = [str(path) for path in parts]
    if current:
        values.append(current)
    return ":".join(values)

def _prepare_bootstrap_env(job: EG6Job, spec: dict[str, Any]) -> tuple[dict[str, str], list[str], Path, Path]:
    bootstrap = spec["bootstrap"]
    bootstrap_root = _bootstrap_root(job)
    results_root = bootstrap_root / "results"
    models_root = bootstrap_root / "models"
    snapshot_minutes = float(os.environ.get("EG6_VGRNN_BOOTSTRAP_SNAPSHOT_MINUTES", str(bootstrap["snapshot_minutes"])))
    workers = int(os.environ.get("EG6_VGRNN_BOOTSTRAP_WORKERS", str(bootstrap["workers"])))
    patience = int(os.environ.get("EG6_VGRNN_BOOTSTRAP_PATIENCE", str(bootstrap["patience"])))
    epochs = int(os.environ.get("EG6_VGRNN_BOOTSTRAP_EPOCHS", str(bootstrap["epochs"])))
    lr = float(os.environ.get("EG6_VGRNN_BOOTSTRAP_LR", str(bootstrap["lr"])))
    fw = bootstrap.get("fw")
    if fw is not None:
        fw = float(os.environ.get("EG6_VGRNN_BOOTSTRAP_FW", str(fw)))
    dimension = bootstrap.get("dimension")
    if dimension is not None:
        dimension = int(os.environ.get("EG6_VGRNN_BOOTSTRAP_DIMENSION", str(dimension)))
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath_with([Path(bootstrap["source_root"]), REPO_ROOT / "src"])
    env[spec["data_env_var"]] = str(_runtime_data_root(job))
    env["VGRNN_RESULTS_ROOT"] = str(results_root)
    env["VGRNN_MODELS_ROOT"] = str(models_root)
    env["VGRNN_RUN_NAME"] = "bootstrap"
    command = [
        str(detector_env_python("vgrnn")),
        "run.py",
        "-n",
        str(bootstrap["dataset_arg"]),
        "-s",
        str(snapshot_minutes),
        "-l",
        str(lr),
        "-p",
        str(patience),
        "--workers",
        str(workers),
        "--tests",
        "1",
        "--epochs",
        str(epochs),
    ]
    if dimension is not None:
        command.extend(["--dimension", str(dimension)])
    if fw is not None:
        command.extend(["--fw", str(fw)])
    return env, command, results_root / "bootstrap" / "metrics.csv", models_root / "bootstrap"

def _official_archive_for(checkpoint_path: Path) -> Path | None:
    detector = checkpoint_path.parent.name.upper()
    archive_names = {
        "ARGUS": "model_argus.zip",
        "EULER": "model_euler.zip",
        "VGRNN": "model_vgrnn.zip",
    }
    archive_name = archive_names.get(detector)
    if not archive_name:
        return None
    archive_path = checkpoint_path.parent / archive_name
    return archive_path if archive_path.exists() else None

def _materialize_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    archive_path = _official_archive_for(checkpoint_path)
    if archive_path is None:
        return checkpoint_path

    detector = checkpoint_path.parent.name.upper()
    target = OFFICIAL_CHECKPOINT_ROOT / detector / checkpoint_path.name
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(checkpoint_path.name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        except KeyError as exc:
            raise FileNotFoundError(f"{checkpoint_path.name} not found in {archive_path}") from exc
    return target

def _maybe_bootstrap_checkpoint(job: EG6Job, spec: dict[str, Any]) -> tuple[str, float | None, dict[str, Any]]:
    checkpoint_path = spec["checkpoint"]["path"]
    threshold = spec["attack"].get("threshold")
    if checkpoint_path:
        return str(_materialize_checkpoint(Path(checkpoint_path))), threshold, {"used": False}

    env, command, metrics_path, model_dir = _prepare_bootstrap_env(job, spec)
    log_root = job.logs_root
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "bootstrap.stdout.log"
    stderr_path = log_root / "bootstrap.stderr.log"

    if not metrics_path.exists() or not any(model_dir.glob("*.pkl")):
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(spec["bootstrap"]["source_root"]),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Bootstrap failed for {job.experiment_id}; see {stdout_path} and {stderr_path}")

    model_paths = sorted(model_dir.glob("*.pkl"))
    if not model_paths:
        raise RuntimeError(f"Bootstrap completed without model checkpoint for {job.experiment_id}")
    if not metrics_path.exists():
        raise RuntimeError(f"Bootstrap completed without metrics.csv for {job.experiment_id}")
    rows = read_csv(metrics_path)
    if not rows:
        raise RuntimeError(f"Bootstrap metrics.csv empty for {job.experiment_id}")
    threshold = float(rows[0]["threshold"])
    return str(model_paths[0]), threshold, {
        "used": True,
        "metrics_path": str(metrics_path),
        "model_path": str(model_paths[0]),
    }

def _write_execution_config(job: EG6Job, config: dict[str, Any], *, checkpoint_path: str, threshold: float | None) -> Path:
    execution = deepcopy(config)
    execution["checkpoint"]["path"] = checkpoint_path
    if threshold is not None:
        execution["attack"]["threshold"] = float(threshold)
    dump_yaml(job.execution_config_path, execution)
    return job.execution_config_path

def _build_attack_env(job: EG6Job, config: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    runtime_root = job.runtime_root
    env["PYTHONPATH"] = _pythonpath_with([runtime_root, REPO_ROOT / "src"])
    env[str(config["runtime"]["data_env_var"])] = str(_runtime_data_root(job))
    env["GIDS_JOB_DATASET"] = job.dataset
    return env

def _load_attack_summary(job: EG6Job) -> dict[str, Any]:
    if not job.attack_summary_path.exists():
        return {}
    with job.attack_summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _clean_job_outputs(job: EG6Job) -> None:
    for path in (
        job.budget_results_path,
        job.target_results_path,
        job.attack_progress_path,
        job.attack_summary_path,
        job.run_summary_path,
    ):
        if path.exists():
            path.unlink()

def _attack_validation_error(job: EG6Job) -> str | None:
    summary = _load_attack_summary(job)
    if not summary:
        return f"missing attack summary: {job.attack_summary_path}"

    target_count = int(summary.get("target_count", 0))
    processed_count = int(summary.get("processed_target_count", 0))
    skipped_count = int(summary.get("skipped_target_count", 0))
    if processed_count + skipped_count != target_count:
        return (
            "incomplete target accounting: "
            f"target_count={target_count}, processed={processed_count}, skipped={skipped_count}"
        )
    if job.dataset == "0501" and target_count == 0:
        return "0501/campus run produced zero attack targets"
    return None

def run_job(job: EG6Job, *, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force and job_status(job) == "success":
        return "skipped"
    if force:
        _clean_job_outputs(job)

    config = load_yaml(config_path)
    spec = _job_spec(job)
    job.metadata_root.mkdir(parents=True, exist_ok=True)
    job.logs_root.mkdir(parents=True, exist_ok=True)
    _stage_data_root(job, spec)
    _stage_runtime_source(job, spec)

    try:
        checkpoint_path, threshold, bootstrap_info = _maybe_bootstrap_checkpoint(job, spec)
        execution_config_path = _write_execution_config(
            job,
            config,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
        )
        env = _build_attack_env(job, config)
        command = [
            str(detector_env_python(job.detector)),
            str(Path(config["runtime"]["template"])),
            "--config",
            str(execution_config_path),
            "--output-dir",
            str(job.metadata_root),
        ]
        stdout_path = job.logs_root / "attack.stdout.log"
        stderr_path = job.logs_root / "attack.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(job.runtime_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        validation_error = None
        if completed.returncode == 0 and job.budget_results_path.exists():
            validation_error = _attack_validation_error(job)
        status = "success" if completed.returncode == 0 and job.budget_results_path.exists() and validation_error is None else "failed"
        summary = {
            "experiment_id": job.experiment_id,
            "detector": job.detector,
            "dataset": job.dataset,
            "repro_level": job.repro_level,
            "status": status,
            "command": command,
            "exit_code": completed.returncode,
            "config_path": str(config_path),
            "execution_config_path": str(execution_config_path),
            "budget_results_path": str(job.budget_results_path),
            "target_results_path": str(job.target_results_path),
            "attack_progress_path": str(job.attack_progress_path),
            "attack_summary_path": str(job.attack_summary_path),
            "stdout_log_path": str(stdout_path),
            "stderr_log_path": str(stderr_path),
            "bootstrap": bootstrap_info,
            "resolved_checkpoint_path": checkpoint_path,
            "resolved_threshold": threshold,
            "attack_summary": _load_attack_summary(job),
        }
        if validation_error is not None:
            summary["validation_error"] = validation_error
    except Exception as exc:
        status = "failed"
        summary = {
            "experiment_id": job.experiment_id,
            "detector": job.detector,
            "dataset": job.dataset,
            "repro_level": job.repro_level,
            "status": "failed",
            "config_path": str(config_path),
            "error": repr(exc),
        }

    job.run_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return "success" if status == "success" else "failed"

def job_status(job: EG6Job) -> str:
    if job.run_summary_path.exists():
        try:
            summary = json.loads(job.run_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "failed"
        if summary.get("status") == "success" and job.budget_results_path.exists():
            return "success"
        if summary.get("status") == "failed":
            return "failed"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

def build_status_rows(jobs: list[EG6Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        summary = {}
        if job.run_summary_path.exists():
            try:
                summary = json.loads(job.run_summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        rows.append({
            "detector": job.detector,
            "dataset": job.dataset,
            "repro_level": job.repro_level,
            "window_seconds": job.window_seconds,
            "experiment_id": job.experiment_id,
            "status": job_status(job),
            "config_path": str(job.config_path),
            "budget_results_path": str(job.budget_results_path),
            "target_results_path": str(job.target_results_path),
            "attack_progress_path": str(job.attack_progress_path),
            "run_summary_path": str(job.run_summary_path),
            "resolved_checkpoint_path": summary.get("resolved_checkpoint_path", ""),
            "resolved_threshold": summary.get("resolved_threshold", ""),
        })
    return rows

def write_matrix_report(jobs: list[EG6Job], path: Path | None = None) -> Path:
    rows = [{
        "detector": job.detector,
        "dataset": job.dataset,
        "repro_level": job.repro_level,
        "window_seconds": job.window_seconds,
        "experiment_id": job.experiment_id,
        "config_path": str(job.config_path),
    } for job in jobs]
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(path, rows, ["detector", "dataset", "repro_level", "window_seconds", "experiment_id", "config_path"])
    return path

def write_status_report(jobs: list[EG6Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(
        path,
        rows,
        [
            "detector",
            "dataset",
            "repro_level",
            "window_seconds",
            "experiment_id",
            "status",
            "config_path",
            "budget_results_path",
            "target_results_path",
            "attack_progress_path",
            "run_summary_path",
            "resolved_checkpoint_path",
            "resolved_threshold",
        ],
    )
    return path

def write_progress_report(
    jobs: list[EG6Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    scope_name: str = "global",
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    status_counts = {status: sum(1 for row in rows if row["status"] == status) for status in ("pending", "partial", "failed", "success")}
    completed = status_counts["success"] + status_counts["failed"]
    completion_ratio = (completed / total) if total else 0.0
    success_ratio = (status_counts["success"] / total) if total else 0.0

    detector_progress = []
    for detector in detector_list():
        detector_rows = [row for row in rows if row["detector"] == detector]
        if not detector_rows:
            continue
        detector_progress.append({
            "detector": detector,
            "total": len(detector_rows),
            "success": sum(1 for row in detector_rows if row["status"] == "success"),
            "failed": sum(1 for row in detector_rows if row["status"] == "failed"),
            "partial": sum(1 for row in detector_rows if row["status"] == "partial"),
            "pending": sum(1 for row in detector_rows if row["status"] == "pending"),
        })

    payload = {
        "scope": scope_name,
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": status_counts["success"],
        "failed_jobs": status_counts["failed"],
        "partial_jobs": status_counts["partial"],
        "pending_jobs": status_counts["pending"],
        "completion_ratio": completion_ratio,
        "success_ratio": success_ratio,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
        "per_detector": detector_progress,
    }

    json_path = json_path or PROGRESS_JSON_PATH
    md_path = md_path or PROGRESS_MD_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# EG6 Progress ({scope_name})",
        "",
        f"- total_jobs: {total}",
        f"- completed_jobs: {completed}",
        f"- remaining_jobs: {total - completed}",
        f"- success_jobs: {status_counts['success']}",
        f"- failed_jobs: {status_counts['failed']}",
        f"- partial_jobs: {status_counts['partial']}",
        f"- pending_jobs: {status_counts['pending']}",
        f"- completion_ratio: {completion_ratio:.1%}",
        f"- success_ratio: {success_ratio:.1%}",
    ]
    if attempted_jobs is not None:
        lines.append(f"- attempted_jobs_in_current_run: {attempted_jobs}")
    if failed_jobs is not None:
        lines.append(f"- failures_in_current_run: {failed_jobs}")
    lines.extend(["", "## Per Detector"])
    for row in detector_progress:
        lines.append(
            f"- {row['detector']}: {row['success'] + row['failed']}/{row['total']} complete "
            f"(success={row['success']}, failed={row['failed']}, partial={row['partial']}, pending={row['pending']})"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

def summarize_jobs(jobs: list[EG6Job], output_root: Path | None = None) -> tuple[Path, Path]:
    output_root = output_root or REPORT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job_status(job) != "success":
            continue
        for row in read_csv(job.budget_results_path):
            raw_rows.append({
                "experiment_id": job.experiment_id,
                "detector": job.detector,
                "dataset": job.dataset,
                "repro_level": job.repro_level,
                "window_seconds": job.window_seconds,
                "budget": int(float(row["budget"])),
                "threshold": float(row["threshold"]),
                "target_count": int(float(row["target_count"])),
                "processed_target_count": int(float(row["processed_target_count"])),
                "skipped_target_count": int(float(row["skipped_target_count"])),
                "flagged_target_count": int(float(row.get("flagged_target_count") or 0)),
                "init_stealthy_rate": float(row["init_stealthy_rate"]),
                "target_stealthy_rate": _optional_rate(row["target_stealthy_rate"]),
                "cover_stealthy_rate": _optional_rate(row["cover_stealthy_rate"]),
                "attack_success_rate": _optional_rate(row["attack_success_rate"]),
                "checkpoint_source": row["checkpoint_source"],
                "checkpoint_path": row["checkpoint_path"],
                "budget_results_path": str(job.budget_results_path),
            })

    raw_path = output_root / "raw_results.csv"
    write_csv(
        raw_path,
        raw_rows,
        [
            "experiment_id",
            "detector",
            "dataset",
            "repro_level",
            "window_seconds",
            "budget",
            "threshold",
            "target_count",
            "processed_target_count",
            "skipped_target_count",
            "flagged_target_count",
            "init_stealthy_rate",
            "target_stealthy_rate",
            "cover_stealthy_rate",
            "attack_success_rate",
            "checkpoint_source",
            "checkpoint_path",
            "budget_results_path",
        ],
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["detector"], row["dataset"]), []).append(row)

    minimal_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for (detector, dataset), rows in sorted(grouped.items()):
        rows_by_budget = {int(row["budget"]): row for row in rows}
        init_rate = rows_by_budget.get(0, {}).get("init_stealthy_rate", 0.0)
        flagged = rows[0]["flagged_target_count"]
        full_budget = ""
        for budget in sorted(budget for budget in rows_by_budget if budget > 0):
            asr = rows_by_budget[budget]["attack_success_rate"]
            if flagged and asr != "" and float(asr) >= 0.999:
                full_budget = budget
                break
        minimal_rows.append({
            "detector": detector,
            "dataset": dataset,
            "repro_level": rows[0]["repro_level"],
            "window_seconds": rows[0]["window_seconds"],
            "target_count": rows[0]["target_count"],
            "processed_target_count": rows[0]["processed_target_count"],
            "skipped_target_count": rows[0]["skipped_target_count"],
            "flagged_target_count": flagged,
            "base_init_stealthy_rate": init_rate,
            "full_evasion_budget": full_budget,
            "max_attack_success_rate": max(
                (float(row["attack_success_rate"]) for row in rows if row["attack_success_rate"] != ""),
                default="",
            ),
            "checkpoint_source": rows[0]["checkpoint_source"],
        })
        evidence_rows.append({
            "detector": detector,
            "dataset": dataset,
            "repro_level": rows[0]["repro_level"],
            "window_seconds": rows[0]["window_seconds"],
            "target_count": rows[0]["target_count"],
            "processed_target_count": rows[0]["processed_target_count"],
            "flagged_target_count": flagged,
            "base_init_stealthy_rate": init_rate,
            "attack_success_rate_k2": rows_by_budget.get(2, {}).get("attack_success_rate", ""),
            "attack_success_rate_k5": rows_by_budget.get(5, {}).get("attack_success_rate", ""),
            "attack_success_rate_k10": rows_by_budget.get(10, {}).get("attack_success_rate", ""),
            "attack_success_rate_k20": rows_by_budget.get(20, {}).get("attack_success_rate", ""),
            "attack_success_rate_k50": rows_by_budget.get(50, {}).get("attack_success_rate", ""),
            "full_evasion_budget": full_budget,
            "verdict": (
                "no_flagged_target" if not flagged
                else "vulnerable" if full_budget != ""
                else "no_full_evasion_observed"
            ),
        })

    minimal_path = output_root / "minimal_budget_by_group.csv"
    write_csv(
        minimal_path,
        minimal_rows,
        [
            "detector",
            "dataset",
            "repro_level",
            "window_seconds",
            "target_count",
            "processed_target_count",
            "skipped_target_count",
            "flagged_target_count",
            "base_init_stealthy_rate",
            "full_evasion_budget",
            "max_attack_success_rate",
            "checkpoint_source",
        ],
    )

    evidence_path = output_root / "evidence_table.csv"
    write_csv(
        evidence_path,
        evidence_rows,
        [
            "detector",
            "dataset",
            "repro_level",
            "window_seconds",
            "target_count",
            "processed_target_count",
            "flagged_target_count",
            "base_init_stealthy_rate",
            "attack_success_rate_k2",
            "attack_success_rate_k5",
            "attack_success_rate_k10",
            "attack_success_rate_k20",
            "attack_success_rate_k50",
            "full_evasion_budget",
            "verdict",
        ],
    )

    _write_conclusion_md(minimal_rows, evidence_rows, output_root / "conclusion.md")
    return raw_path, minimal_path

def _write_conclusion_md(
    minimal_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    unflagged = [row for row in minimal_rows if not row["flagged_target_count"]]
    vulnerable = [row for row in minimal_rows if row["full_evasion_budget"] != ""]
    not_yet = [
        row for row in minimal_rows
        if row["full_evasion_budget"] == "" and row["flagged_target_count"]
    ]
    lines = [
        "# EG6 Conclusion",
        "",
        f"- completed_groups: {len(minimal_rows)}",
        f"- groups_with_no_flagged_target: {len(unflagged)}",
        f"- vulnerable_groups: {len(vulnerable)}",
        f"- groups_without_full_evasion: {len(not_yet)}",
        "",
        "## Minimal Budget",
    ]
    if vulnerable:
        for row in vulnerable:
            lines.append(
                f"- {row['detector']}/{row['dataset']} ({row['repro_level']}): "
                f"full evasion first observed at K={row['full_evasion_budget']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Not Yet Fully Evaded"])
    if not_yet:
        for row in not_yet:
            lines.append(
                f"- {row['detector']}/{row['dataset']} ({row['repro_level']}): "
                f"max_attack_success_rate={float(row['max_attack_success_rate']):.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Evidence Snapshot"])
    for row in evidence_rows:
        lines.append(
            f"- {row['detector']}/{row['dataset']}: "
            f"ASR@2={row['attack_success_rate_k2']} "
            f"ASR@5={row['attack_success_rate_k5']} "
            f"ASR@10={row['attack_success_rate_k10']} "
            f"ASR@20={row['attack_success_rate_k20']} "
            f"ASR@50={row['attack_success_rate_k50']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def detector_env_status(detectors: list[str] | None = None) -> tuple[Path, list[dict[str, str]]]:
    selected = detectors or detector_list()
    rows = []
    for detector in selected:
        python_bin = detector_env_python(detector)
        if not python_bin.exists():
            rows.append({
                "detector": detector,
                "imports_ok": "false",
                "missing_imports": "python",
                "note": f"missing interpreter {python_bin}",
            })
            continue
        command = [
            str(python_bin),
            "-c",
            (
                "import json, pkgutil; "
                f"mods={list(DETECTOR_REQUIRED_IMPORTS[detector])!r}; "
                "missing=[mod for mod in mods if pkgutil.find_loader(mod) is None]; "
                "print(json.dumps({'missing': missing}))"
            ),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            rows.append({
                "detector": detector,
                "imports_ok": "false",
                "missing_imports": "unknown",
                "note": completed.stderr.strip() or "import probe failed",
            })
            continue
        payload = json.loads(completed.stdout.strip() or "{}")
        missing = payload.get("missing", [])
        rows.append({
            "detector": detector,
            "imports_ok": "true" if not missing else "false",
            "missing_imports": ";".join(missing),
            "note": "",
        })

    path = ORCHESTRATOR_ROOT / "env_status.csv"
    write_csv(path, rows, ["detector", "imports_ok", "missing_imports", "note"])
    return path, rows
