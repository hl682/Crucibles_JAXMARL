# Sluice Commons and Downstream Passage

Author status: AUTHORED_NOT_EXECUTED. Candidate `game-v3-00130`, source attempt 0, train-only. Static structural screening is recorded separately in `static_receipt.json`; it is not numerical R0, gameplay acceptance, solvability, training or transfer evidence.

A/O/U/H = 3/158/10/160. Source SHA-256: `fe586a7aceb81e759b32214811a9bbb7f6797c110430d946c4c179bbf2dc360c`.

## Task and causal interaction

Deliver the three finite cargo boats to basin 3 while irrigating each indexed crop twice or recording its loss. Each role benefits especially from its same-index cargo and crop; sharing and bridge access affect all roles.

A gate simultaneously controls a walking bridge, one conserved delayed water packet, and boat passage. Irrigation and emergency drainage remove water needed for later downstream cargo. Close votes veto opens. Shared crop and spill allocations rotate by t modulo three.

The reservoir, three basins, gates, crops and boats are one physical economy. Closing a bridge can let a partner reach the lower bank but blocks that partner's launched boat and the following water packet. An upstream owner can consume or drain water before a downstream owner has sufficient depth. Saving water upstream can therefore be locally attractive while imposing delay elsewhere. Conversely, keeping a low gate open can build dangerous crop depth until a partner irrigates or spills.

Plausible policies include pulse-and-cross schedules, downstream-first water sharing, and crop-first drawdown before launching cargo. These are alternative strategy hypotheses, not established feasible or successful schedules. Cargo owner and crop owner coincide by index, but any steward may operate any gate or crop. No proprietary benchmark mechanics or layouts were copied.

The declared accounting identity is sum(water) + sum(transit) + sum(crop_dose) + drained == 15. Arrivals precede departures, departures depend on old donor stock, irrigation/spilling follows flow, and boats then use the resulting stored depths. This is a source-level invariant claim for later R0 checking; it has not been executed. Flood damage never silently discards water or cargo. The two crop-dose capacity and three health points are game parameters, not acceptance thresholds.

Changing the gate layout, storage bounds, crop-owner assignment or cargo starting nodes would change which bridge and reservoir decisions need coordination. Such variants require separately identified source/protocol changes; this source does not claim adaptation or transfer.

## Actions, timing and rewards

- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Vote open adjacent gates.
- 6: Veto and close adjacent gates.
- 7: Irrigate one adjacent viable crop.
- 8: Launch adjacent cargo boats.
- 9: Spill one adjacent basin unit.

Every indexed intent is legal for a live capable role, even when its spatial/resource precondition fails; failed intents become deterministic no-ops. Out-of-range actions become wait. Terminal roles, and temporarily/incapacitated roles where specified, retain wait only. Masks do not leak private feasibility. Targets outside the map, duplicate requested destinations, or another role current cell are rejected. No swaps or following into a just-vacated occupied cell occur in that transition.

Reward expressions are literal in `advance` and are returned per role as float32. The manifest bounds are conservative finite interface bounds, not performance or acceptance gates. No return, score, success rate, optimality, statistical result or learning result was measured.

## Native observations and rendering

Fully observable. Each role receives its identity and the complete public dynamic state and tile layout. Fixed active/visibility arrays are public constants; boat activity is exactly the complement of the observed delivered flag. Static entity positions are included or directly determined by observed node identity.

Every flat observation column is covered by exactly one `observation_fields` slice. `state_visibility_mapping` accounts for every native leaf. The scene binds the exact source and manifest hashes, actual tile map, actual entity positions, activity, native role masks and implemented attributes. Per-step flags, cumulative counters and levels are declared separately. A privileged full-state debug view must be labeled separately by the later renderer; it is not a policy view.

## Termination and numerical boundary

All three boats delivered and every crop either has two doses or zero health. Otherwise the horizon is 160 live transitions. A dead crop remains an explicit negative outcome.

Natural termination takes priority if it coincides with the horizon. `truncated = (next_t >= H) & ~terminated`; `done` equals their disjunction. A terminal raw call retains every state leaf, observation, clock and cause flag and returns zero rewards and all dones true. There is no internal reset. The independent wrapper is responsible for autoreset. State integers use int32, flags bool, and observations/rewards float32. No small signed-byte design distribution was replaced; the reset uint32 key stays external.

## Key schedule and witness status

Deterministic reset and transition. The supplied reset and raw-step keys are intentionally unused; there is no stochastic distribution or hidden PRNG state.

Replay must use `reset(jax.random.PRNGKey(seed))`. Zero-based absolute raw step k receives `jax.random.fold_in(jax.random.PRNGKey(seed), k + 1)`. Paired suffixes continue at prefix length L, with identical reset and prefix keys in both branches. Neither scripts nor pairs have been executed, searched or optimized. The JSON lists are literal intended actions, not recorded trajectories. Any actual illegality, failed objective or mismatch must remain in the later receipt.

## Source line evidence

- `refresh`: `candidate.py:12-22`; literal declaration `def refresh(self, s):`.
- `reset`: `candidate.py:24-55`; literal declaration `def reset(self, key):`.
- `observe`: `candidate.py:57-65`; literal declaration `def observe(self, s):`.
- `legal_actions`: `candidate.py:67-69`; literal declaration `def legal_actions(self, s):`.

`candidate.py:89-92`

```python
        # GATE_VOTES: all nearby votes apply together; any close vetoes opening.
        near_gate = jnp.sum(jnp.abs(n["agent_pos"][:, None, :] - s["gate_pos"][None, :, :]), axis=2) <= 1
        opening = jnp.any(near_gate & (a[:, None] == 5), axis=0)
        closing = jnp.any(near_gate & (a[:, None] == 6), axis=0)
```

`candidate.py:94-97`

```python
        # CONSERVED_FLOW: old packets arrive; departures use pre-arrival donor stock.
        arrived = s["water"].at[1:].add(s["transit"])
        flow = (n["gate_open"] & (s["water"][:-1] > 0)
                & (arrived[1:] < 6)).astype(jnp.int32)
```

`candidate.py:104-107`

```python
        # ALLOCATION: rotating role priority resolves scarce water and shared crops.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            distances = jnp.sum(jnp.abs(s["crop_pos"] - n["agent_pos"][i]), axis=1)
```

`candidate.py:125-128`

```python
        # CARGO_PASSAGE: a launched boat advances at most one wet, open lock per turn.
        near_boat = jnp.sum(jnp.abs(n["agent_pos"][:, None, :] - s["boat_pos"][None, :, :]), axis=2) <= 1
        launched = jnp.any(near_boat & (a[:, None] == 8), axis=0)
        running = s["boat_running"] | (launched & ~s["boat_delivered"])
```

`candidate.py:140-143`

```python
        # FLOOD_COST: unfinished crops lose persistent health at basin depth five or six.
        flood = (water[1:] >= 5) & (doses < 2) & (s["crop_health"] > 0)
        n["crop_health"] = s["crop_health"] - flood.astype(jnp.int32)
        n["last_flood"] = flood
```

`candidate.py:158-161`

```python
        # ABSORBING_TERMINAL: preserve all leaves, causes and clock on terminal input.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        dones = jnp.full((3,), n["done"], dtype=jnp.bool_)
```

## Supplied public reference attribution

- `references/JaxMARL_README.md` (6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15): Parallel agent-keyed boundary and __all__ conventions read as API context. The authored flat kernel does not implement the separate public adapter. No game code copied.
- `references/Pgx_README.md` (153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df): Explicit state, legal-mask and externally supplied random-key API context. No board rules, layouts or source code copied.
- `references/Craftax_README.md` (7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff): Spatial gameplay, native symbolic observation/rendering and explicit reset-boundary context. No Craftax mechanics or code copied.
- `references/PureJaxRL_README.md` (d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0): Algorithm-library identity checked; no game rule source and no replacement for the frozen learner.
- `references/Craftax_Baselines_ppo_rnn.py` (5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011): Read its imports, recurrent-policy/reset handling and PureJaxRL attribution as integration context only. No baseline code copied or executed.

All substantive game rules and kernels were authored for this batch. No upstream source blocks were copied. Only the supplied local official reference snapshots were read; no network lookup, external benchmark implementation, earlier candidate, training outcome or final-test evidence was used. The public JaxMARL adapter, executable replay, animation and learning evaluation remain separate integration work; this source package does not claim those deliveries.
