"""Discrete-event simulator: concurrent agent sessions against a fixed KV budget.

The single-pause view treats each decision as independent. It is not. A cache
retained for one session is memory another session cannot have, and the cost of
that is queueing delay, which only appears once requests actually compete.

What the simulator adds over the single-pause view:

  - A real budget. The KV pool holds a fixed number of tokens. A session that
    cannot fit waits.
  - Forced eviction. When the pool is full and someone needs space, a victim is
    chosen by LRU among retained-but-idle caches. A policy's tau is therefore an
    upper bound on residency, not a guarantee.
  - Re-prefill cost on a miss. A returning session whose cache is gone occupies
    the GPU for context/prefill_rate seconds before it can decode, delaying
    everyone behind it.

Replay model. Sessions are replayed from their recorded timestamps: the pause
durations and context lengths are real, taken from TraceLab. What is simulated
is the serving system's response to them. `arrival_scale` compresses the
inter-session arrival gaps to raise concurrency without inventing new
workload -- at scale 1.0 the sessions arrive as recorded, and the server is
mostly idle, which is why the sweep goes well above 1.

Assumption, stated rather than buried: session affinity holds. A returning
request reaches the worker that holds its cache. SAGA's ablation shows removing
affinity costs 96% of task completion time, far more than retention policy ever
could, so without this assumption none of these differences would be visible.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .policies import Decision, Policy

# Event kinds, ordered so that frees happen before allocations at equal time.
_PAUSE_END = 0  # tool returned; session wants its cache back
_TTL_EXPIRY = 1  # policy's retention deadline reached
_DECODE_END = 2  # LLM step finished; next pause begins


@dataclass(order=True)
class _Event:
    time: float
    kind: int
    seq: int
    session: str = field(compare=False)
    round_index: int = field(compare=False, default=-1)


@dataclass
class _Session:
    """One agent session's replay state."""

    session_id: str
    pauses: pd.DataFrame
    cursor: int = 0
    resident: bool = False
    last_used: float = 0.0
    context_tokens: float = 0.0
    ttl_deadline: float = float("inf")
    # Accounting
    recomputed_tokens: float = 0.0
    queue_delay_s: float = 0.0
    evictions: int = 0
    first_event_s: float = float("inf")
    last_event_s: float = 0.0


class Simulator:
    """Replays sessions against a fixed KV pool under one retention policy."""

    def __init__(
        self,
        frame: pd.DataFrame,
        policy: Policy,
        cost: CostModel,
        kv_budget_tokens: int | None = None,
        decode_s_per_round: float = 0.5,
        arrival_scale: float = 1.0,
    ):
        self.cost = cost
        self.policy = policy
        self.budget = int(kv_budget_tokens or cost.kv_budget_tokens)
        self.decode_s = decode_s_per_round
        self.arrival_scale = arrival_scale

        self.sessions: dict[str, _Session] = {}
        self.used_tokens = 0.0
        self.gpu_free_at = 0.0  # prefill is serialised: one re-prefill at a time
        self._seq = 0
        self._heap: list[_Event] = []
        # Admissions whose context alone exceeds the whole pool. Those cannot be
        # made to fit by evicting anyone, so the budget is necessarily violated.
        # A configuration with a non-trivial count here is not measuring
        # retention policy -- it is measuring a pool too small for the workload.
        self._admits = 0
        self._oversized_admits = 0
        self._build(frame)

        # Integral of used_tokens over time, for utilisation.
        self._last_t = 0.0
        self._token_seconds = 0.0

    def _build(self, frame: pd.DataFrame) -> None:
        """Lay sessions out on a common clock, compressing arrival gaps."""
        frame = frame.sort_values(["session_id", "round_index"])
        starts = frame.groupby("session_id")["pause_start_at"].min()
        origin = starts.min()
        offsets = (starts - origin).dt.total_seconds() / max(self.arrival_scale, 1e-9)

        for sid, group in frame.groupby("session_id", sort=False):
            g = group.reset_index(drop=True)
            self.sessions[sid] = _Session(session_id=sid, pauses=g)
            # Session begins by decoding its first round.
            self._push(float(offsets[sid]), _DECODE_END, sid, 0)

    def _push(self, time: float, kind: int, session: str, round_index: int = -1) -> None:
        self._seq += 1
        heapq.heappush(self._heap, _Event(time, kind, self._seq, session, round_index))

    # -- memory -----------------------------------------------------------

    def _advance_clock(self, t: float) -> None:
        if t > self._last_t:
            self._token_seconds += self.used_tokens * (t - self._last_t)
            self._last_t = t

    def _release(self, s: _Session) -> None:
        if s.resident:
            self.used_tokens -= s.context_tokens
            s.resident = False
            s.ttl_deadline = float("inf")

    def _admit(self, s: _Session, tokens: float, now: float) -> None:
        """Make room for `tokens` by evicting idle caches, LRU first, then admit.

        A session currently paused is evictable; one that is actively decoding
        is not. If even evicting everything idle is not enough, the request
        still proceeds -- the budget is a soft bound here, and the resulting
        over-subscription shows up as utilisation above 1.0 rather than as a
        deadlock. Configurations where that happens are reported, not hidden.
        """
        self._admits += 1
        if tokens > self.budget:
            self._oversized_admits += 1
        while self.used_tokens + tokens > self.budget:
            victim = self._lru_victim(exclude=s.session_id)
            if victim is None:
                break
            self._release(victim)
            victim.evictions += 1
        if not s.resident:
            s.resident = True
            s.context_tokens = tokens
            self.used_tokens += tokens
        else:
            self.used_tokens += tokens - s.context_tokens
            s.context_tokens = tokens
        s.last_used = now

    def _lru_victim(self, exclude: str) -> _Session | None:
        best, best_t = None, float("inf")
        for sid, s in self.sessions.items():
            if sid == exclude or not s.resident:
                continue
            if s.last_used < best_t:
                best, best_t = s, s.last_used
        return best

    # -- the loop ---------------------------------------------------------

    def run(self) -> dict:
        while self._heap:
            ev = heapq.heappop(self._heap)
            self._advance_clock(ev.time)
            s = self.sessions[ev.session]
            s.first_event_s = min(s.first_event_s, ev.time)
            s.last_event_s = max(s.last_event_s, ev.time)

            if ev.kind == _TTL_EXPIRY:
                # Deadline reached while still paused: let the cache go.
                if s.resident and ev.time >= s.ttl_deadline - 1e-12:
                    self._release(s)
                    s.evictions += 1

            elif ev.kind == _DECODE_END:
                self._begin_pause(s, ev.time)

            elif ev.kind == _PAUSE_END:
                self._resume(s, ev.time)

        self._advance_clock(self._last_t)
        return self._report()

    def _begin_pause(self, s: _Session, now: float) -> None:
        """A round finished decoding; the tool call starts. Ask the policy."""
        if s.cursor >= len(s.pauses):
            return
        row = s.pauses.iloc[s.cursor]
        ctx = float(row["context_tokens"])
        self._admit(s, ctx, now)

        tau = float(
            self.policy(
                Decision(
                    predicted_duration_s=float(row["predicted_duration_s"]),
                    context_tokens=ctx,
                    tool_key=str(row["tool_key"]),
                    true_duration_s=float(row["pause_s"]),
                )
            )
        )
        if tau <= 0:
            self._release(s)
            s.evictions += 1
        elif np.isfinite(tau):
            s.ttl_deadline = now + tau
            self._push(s.ttl_deadline, _TTL_EXPIRY, s.session_id)

        self._push(now + float(row["pause_s"]), _PAUSE_END, s.session_id, int(row["round_index"]))

    def _resume(self, s: _Session, now: float) -> None:
        """The tool returned. Either the cache is there, or we pay to rebuild it."""
        row = s.pauses.iloc[s.cursor]
        ctx = float(row["context_tokens"])
        start = now

        if not s.resident:
            # Miss: re-prefill. The GPU is serialised, so this also queues
            # behind any re-prefill already running -- that queueing delay is
            # the cost the single-pause model cannot see.
            s.recomputed_tokens += ctx
            prefill_s = ctx / self.cost.prefill_tokens_per_s
            begin = max(now, self.gpu_free_at)
            s.queue_delay_s += (begin - now) + prefill_s
            self.gpu_free_at = begin + prefill_s
            now = begin + prefill_s
            self._advance_clock(now)
            self._admit(s, ctx, now)
        else:
            s.ttl_deadline = float("inf")
            s.last_used = now

        s.last_event_s = max(s.last_event_s, now)
        s.cursor += 1
        if s.cursor < len(s.pauses):
            self._push(now + self.decode_s, _DECODE_END, s.session_id)
        else:
            self._release(s)
        _ = start

    def _report(self) -> dict:
        jcts, delays = [], []
        recomputed = evictions = 0.0
        for s in self.sessions.values():
            if np.isfinite(s.first_event_s):
                jcts.append(s.last_event_s - s.first_event_s)
            delays.append(s.queue_delay_s)
            recomputed += s.recomputed_tokens
            evictions += s.evictions
        jcts = np.array(jcts, dtype=float)
        delays = np.array(delays, dtype=float)
        makespan = max(self._last_t, 1e-9)
        return {
            "mean_jct_s": float(jcts.mean()) if jcts.size else 0.0,
            "p95_jct_s": float(np.percentile(jcts, 95)) if jcts.size else 0.0,
            "p99_jct_s": float(np.percentile(jcts, 99)) if jcts.size else 0.0,
            "total_queue_delay_s": float(delays.sum()),
            "mean_queue_delay_s": float(delays.mean()) if delays.size else 0.0,
            "recomputed_tokens": float(recomputed),
            "evictions": float(evictions),
            "kv_utilisation": float(self._token_seconds / (self.budget * makespan)),
            "oversized_admit_fraction": float(self._oversized_admits / max(self._admits, 1)),
            "makespan_s": float(makespan),
            "throughput_sessions_per_min": float(len(self.sessions) / (makespan / 60.0)),
        }


def replicate_sessions(frame: pd.DataFrame, replicas: int, stagger_s: float = 7.0) -> pd.DataFrame:
    """Duplicate the session population to raise concurrency.

    The held-out trace contains 384 sessions, which cannot saturate a realistic
    KV pool no matter how hard the arrival gaps are compressed: utilisation
    plateaus near 0.49 at a 600k-token budget because concurrency is bounded by
    how many sessions overlap, not by how fast they arrive.

    Replicating the population is the standard way to raise offered load
    without inventing workload. Each copy keeps its real pause durations and
    context lengths; only its identity and start offset change, so the
    duration distribution the policies face is unchanged.

    This is load generation, not more data. Results from replicas > 1 describe
    a hypothetical server carrying N times this trace's concurrency -- they are
    not additional evidence about the trace itself.
    """
    if replicas <= 1:
        return frame
    parts = []
    for i in range(replicas):
        part = frame.copy()
        part["session_id"] = part["session_id"].astype(str) + f"__r{i}"
        if i:
            shift = pd.Timedelta(seconds=stagger_s * i)
            part["pause_start_at"] = part["pause_start_at"] + shift
            part["pause_end_at"] = part["pause_end_at"] + shift
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_one(
    frame: pd.DataFrame,
    policy: Policy,
    cost: CostModel,
    kv_budget_tokens: int | None = None,
    arrival_scale: float = 1.0,
    decode_s_per_round: float = 0.5,
) -> dict:
    """Convenience wrapper: one policy, one configuration, one result dict."""
    return Simulator(
        frame,
        policy,
        cost,
        kv_budget_tokens=kv_budget_tokens,
        arrival_scale=arrival_scale,
        decode_s_per_round=decode_s_per_round,
    ).run()
