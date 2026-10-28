"""
Online estimate of how uneven task sizes are, for `kleinrock-switch`
(DECISIONS 39).

The signal is the CV of R, requests per task. R does not depend on the
scheduling policy; its online estimate does, through which tasks have
finished, so tasks still in the system enter as right-censored:

- a completed task is an event at its R;
- a task with a request queued or running has R >= c + 1 (c = requests
  done), so it survives every j <= c;
- a task in a tool phase may already have made its last request, so it is
  only known to have R >= c, and survives every j <= c - 1.

Kaplan-Meier over integer R gives S(j) = P(R > j); the moments are restricted
to m, the largest R or c seen: E[R] = sum_{j<m} S(j) and
E[R^2] = sum_{j<m} (2j + 1) S(j). Restriction truncates the tail, so the CV is
biased low until long tasks have been seen.
"""

import math

import numpy as np


def km_cv(events, censored):
    """CV of R from completed R values (`events`) and in-flight tasks as
    (c, has_request) pairs. None until anything is known."""
    cs = [c for c, _ in censored]
    m = max(max(events, default=0), max(cs, default=0))
    if m < 1 or not events:
        return None
    ev = np.bincount(np.asarray(events, dtype=int), minlength=m + 2)
    lim = [c if q else c - 1 for c, q in censored]
    lim = [x for x in lim if x >= 1]
    cn = np.bincount(np.asarray(lim, dtype=int), minlength=m + 2) if lim else np.zeros(m + 2)
    # at risk at j: events with R >= j plus censored with limit >= j
    at_risk = ev[::-1].cumsum()[::-1] + cn[::-1].cumsum()[::-1]
    j = np.arange(1, m)
    n, d = at_risk[1:m], ev[1:m]
    h = np.divide(d, n, out=np.zeros(len(j)), where=n > 0)
    S = np.concatenate(([1.0], np.cumprod(1.0 - h)))      # S(0) .. S(m-1)
    k = np.arange(m)
    mean = float(S.sum())
    var = float(((2 * k + 1) * S).sum()) - mean ** 2
    return math.sqrt(max(var, 0.0)) / mean


class KMRequestsEstimator:
    """Tracks every task from engine notifications and answers `cv()`."""

    def __init__(self):
        self.arrived = 0
        self.done = {}            # task -> requests finished, while in system
        self.outstanding = {}     # task -> requests queued or running
        self.completed = []       # R of finished tasks

    def arrive(self, tid):
        self.arrived += 1
        self.done[tid] = 0
        self.outstanding[tid] = 0

    def enqueue(self, tid):
        self.outstanding[tid] += 1

    def request_done(self, tid):
        self.done[tid] += 1
        self.outstanding[tid] -= 1

    def task_done(self, tid):
        self.completed.append(self.done.pop(tid))
        self.outstanding.pop(tid)

    def cv(self):
        return km_cv(self.completed, [(c, self.outstanding[t] > 0)
                                      for t, c in self.done.items()])
