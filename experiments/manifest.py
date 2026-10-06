"""
Run manifests: what a run used, and the configuration every calibrated value
was calibrated UNDER.

WHY THIS EXISTS
    FINDINGS 23 was retracted because its arrival rate was calibrated at
    batch=16 and the table then ran at batch=8 -- about 2.2x the intended
    load. Nothing in the output said so. A calibrated number is only valid at
    the configuration it was calibrated under, so that configuration is stored
    with the number, and `assert_consistent` refuses to let a measurement run
    under any other.

    m = Manifest.start("scheduler_table", params={...})
    check_consistent("arrival_rate", under, run_config, also_check=("util",))
    rate = calibrate_rate(...)                       # only after the check
    m.record(Calibration("arrival_rate", rate, under, target="...", achieved=u,
                         also_check=("util",)))
    m.assert_consistent(run_config)                  # before the measurement loop
    out["manifest"] = m.to_dict()

CalibrationMismatch deliberately does NOT subclass RuntimeError: both drivers
catch RuntimeError around engine runs (deadlock, overload) and carry on, and
this must never be swallowed by those handlers.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A calibration made under a different value of any of these does not
# describe the run. `util` is added per calibration (arrival_rate targets it).
SENSITIVE_KEYS = ("trace_set", "scale", "kv_mb", "max_batch", "cache_mode")


class CalibrationMismatch(Exception):
    """A calibrated value is being used at a configuration it was not
    calibrated under. The run must not produce results."""


@dataclass
class Calibration:
    name: str                  # "arrival_rate", "slack_factor", ...
    value: float
    under: dict                # the config this was calibrated at
    target: str                # what it was calibrated to hit, in words
    achieved: Optional[float]  # what it actually hit
    # Keys beyond SENSITIVE_KEYS that must also match the run, e.g. "util" for
    # arrival_rate. Other keys in `under` are recorded for the reader only.
    also_check: tuple = ()

    def checked_keys(self) -> tuple:
        return SENSITIVE_KEYS + tuple(k for k in self.also_check
                                      if k not in SENSITIVE_KEYS)


def check_consistent(name: str, under: dict, run_config: dict,
                     also_check: Iterable[str] = (), value=None) -> None:
    """
    Raise CalibrationMismatch if `under` and `run_config` disagree on any
    checked key, or if either one omits it (an unrecorded key cannot be
    verified, and silently skipping it is how §23 happened).

    Callable before the calibration runs, since `under` is known from the
    arguments it will be called with -- so a mismatch costs nothing.
    """
    keys = SENSITIVE_KEYS + tuple(k for k in also_check if k not in SENSITIVE_KEYS)
    shown = "(not yet calibrated)" if value is None else repr(value)
    for k in keys:
        if k not in under or k not in run_config:
            where = "calibration config" if k not in under else "run config"
            raise CalibrationMismatch(
                f"{name}: key {k!r} is missing from the {where}, so the "
                f"calibration cannot be checked against the run.\n"
                f"  calibrated under: {under}\n"
                f"  run config:       {run_config}\n"
                f"Record {k!r} in both.")
        if under[k] != run_config[k]:
            raise CalibrationMismatch(
                f"{name}={shown} was calibrated at {k}={under[k]!r} but the "
                f"run uses {k}={run_config[k]!r}.\n"
                f"  calibrated under: {under}\n"
                f"  run config:       {run_config}\n"
                f"Recalibrate {name} at {k}={run_config[k]!r} (change the "
                f"CALIBRATION, not the run, unless {k}={run_config[k]!r} is "
                f"itself the mistake). No results were produced.")


def bound_args(fn, *args, **kwargs) -> dict:
    """Every argument `fn` will see for this call, defaults included. Used to
    record what a calibration actually ran under rather than what the caller
    believes it passed -- the §23 rate came from an unpassed default."""
    b = inspect.signature(fn).bind(*args, **kwargs)
    b.apply_defaults()
    return dict(b.arguments)


def default_of(fn, name: str):
    return inspect.signature(fn).parameters[name].default


def _git(*args) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class Manifest:
    experiment: str
    params: dict
    git_hash: Optional[str]
    git_dirty: Optional[bool]
    # Tracked files only: results/ is gitignored and written by the run itself.
    git_dirty_paths: list
    started_at: str
    calibrations: list = field(default_factory=list)
    verified_runs: list = field(default_factory=list)

    @classmethod
    def start(cls, experiment: str, params: dict) -> "Manifest":
        head = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
        paths = ([ln[3:] for ln in status.splitlines() if ln.strip()]
                 if status is not None else [])
        return cls(experiment=experiment, params=dict(params),
                   git_hash=head.strip() if head else None,
                   git_dirty=bool(paths) if status is not None else None,
                   git_dirty_paths=paths,
                   started_at=_dt.datetime.now().astimezone().isoformat(
                       timespec="seconds"))

    def record(self, cal: Calibration) -> Calibration:
        missing = [k for k in cal.checked_keys() if k not in cal.under]
        if missing:
            raise CalibrationMismatch(
                f"{cal.name}: calibration recorded without {missing} in its "
                f"`under` config, so it could never be checked. under={cal.under}")
        self.calibrations.append(cal)
        return cal

    def assert_consistent(self, run_config: dict,
                          calibrations: Optional[Iterable[Calibration]] = None
                          ) -> None:
        """Check `run_config` against every recorded calibration (or only the
        given ones, for drivers with one calibration per cell). Call it
        immediately before the measurement loop."""
        cals = list(self.calibrations if calibrations is None else calibrations)
        if not cals:
            raise CalibrationMismatch(
                "assert_consistent called with no calibrations recorded; "
                "nothing was verified.")
        for c in cals:
            check_consistent(c.name, c.under, run_config, c.also_check, c.value)
        self.verified_runs.append({"config": dict(run_config),
                                   "against": [c.name for c in cals]})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        d = dict(d)
        d["calibrations"] = [Calibration(**{**c, "also_check":
                                            tuple(c.get("also_check", ()))})
                             for c in d.get("calibrations", [])]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Manifest":
        return cls.from_dict(json.loads(s))
