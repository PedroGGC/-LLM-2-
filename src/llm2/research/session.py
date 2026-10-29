"""Whole-session budget shared with bootstrap. Subprocess watchdog leaves export reserve."""

import math
import threading
import time

from .campaign import Campaign


class SessionBudgetExceeded(RuntimeError):
    pass


class SessionBudget:
    def __init__(self, seconds=7200., reserve=300., *, elapsed=0., clock=time.monotonic):
        if (not all(math.isfinite(v) for v in (seconds, reserve, elapsed))
                or not 0 <= elapsed < seconds or not 0 < reserve < seconds):
            raise ValueError("invalid session seconds/reserve/elapsed")
        self.seconds, self.reserve, self.elapsed_before_start = seconds, reserve, elapsed
        self.clock, self.started = clock, clock()

    @property
    def elapsed(self):
        return self.elapsed_before_start + self.clock() - self.started

    @property
    def work_remaining(self):
        return max(0., self.seconds - self.reserve - self.elapsed)

    def require(self, estimated_seconds=0.):
        if not math.isfinite(estimated_seconds) or estimated_seconds < 0:
            raise ValueError("invalid duration estimate")
        if self.work_remaining <= estimated_seconds:
            raise SessionBudgetExceeded("session budget: stopping work with export reserve preserved")

    def snapshot(self):
        return {"limit_seconds": self.seconds, "export_reserve_seconds": self.reserve,
                "elapsed_seconds": self.elapsed, "work_remaining_seconds": self.work_remaining,
                "scope": "bootstrap/setup + tests + pilot + training + evaluation; export uses reserve"}


class SessionCampaign(Campaign):
    def __init__(self, *args, budget, **kwargs):
        self.budget = budget
        self.budget.require()
        super().__init__(*args, **kwargs)

    def command(self, arguments, log_name, *, gpu=None, job=None):
        self.budget.require()
        expired = threading.Event()

        finished = threading.Event()

        def monitor():
            if finished.wait(self.budget.work_remaining):
                return
            expired.set()
            # Retry until registration completes if the deadline races with Popen.
            while not finished.is_set():
                self.abort()
                finished.wait(.1)

        watchdog = threading.Thread(target=monitor, daemon=True)
        watchdog.start()
        try:
            result = super().command(arguments, log_name, gpu=gpu, job=job)
            if expired.is_set():
                raise SessionBudgetExceeded("subprocess reached session work deadline")
            return result
        except RuntimeError as exc:
            if expired.is_set():
                raise SessionBudgetExceeded("subprocess stopped at session work deadline; prior complete checkpoints preserved") from exc
            raise
        finally:
            finished.set()
            watchdog.join(timeout=1.)
            self.manifest["session_budget"] = self.budget.snapshot()
            self.save()
