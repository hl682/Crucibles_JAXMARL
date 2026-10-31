# Conveyor Workshop and Synchronized Assembly

Author status: AUTHORED_NOT_EXECUTED. Candidate `game-v3-00133`, source attempt 0, train-only. Static structural screening is recorded separately in `static_receipt.json`; it is not numerical R0, gameplay acceptance, solvability, training or transfer evidence.

A/O/U/H = 4/303/13/144. Source SHA-256: `16b890167ace04226b713bed25922fbf9cb4a3f0e4582535da5074ef52b6c209`.

## Task and causal interaction

Use six finite parts to satisfy three chassis-plus-coil orders, reworking defects and matching requested coil colors, then physically carry valid products to the dock before their role-linked deadlines when possible.

Routing, braking and occupied conveyor/machine buffers change partners available parts and delays. Inspection or recipe publication changes partner information. Shared power spent on one rework may prevent a different repair. Assembly permanently consumes two finite parts, and poor hidden-quality decisions impose shared losses.

The conveyor is an actual one-cell-buffer graph, not an animated counter. A routing choice can fill a single-part machine and block following material; a partner can free that buffer by carrying the processed item away. The belt uses pre-move occupancy, so a packed chain does not magically shift through occupied cells in one tick. A brake can protect a handoff or inspection opportunity while consuming delivery time.

Only the quality specialist initially observes all defect bits and only the recipe planner initially observes requested coil colors. Those bits affect actual assembly acceptance. Other roles can reveal the information by spatial inspection or publication, which takes an action and travel. Public colors, custody and machine operations remain visible to all. Reworking publishes the now-known zero defect because the operation itself establishes it; there is no private-data display shortcut.

Plausible strategies include inspecting before routing, reworking uncertain parts proactively, staging one order at a time near the bench, dividing the two machines between repair and painting, or braking to clear congestion before restarting the line. These involve different power, time, information and buffer tradeoffs. They are unexecuted hypotheses, not proven schedules. Every role can perform every operation; specialization is informational and positional rather than a hidden action privilege.

Part identity is permanent. Modes are 0 loose/belt, 1 carried, 2 in a machine, 3 constituents of a ready product, 4 constituents of a shipped product, and 5 rejected scrap. Assembly changes the two constituent identities into inactive modes and records their order index; the product entity is their aggregate representation rather than newly created raw material. There are exactly three chassis and three coils, no respawns, and power + spent_power == 12. A rejected order stays rejected. Machine slots and worker custody are single-capacity.

Existing timers finish before new interactions, then control/custody/processing/inspection resolve, assembly consumes parts, product custody and dispatch resolve, and the belt advances last. New jobs therefore do not immediately finish. Static map and node positions are public, while all private attributes are confined to matching masked records. Changing recipes, defect draws, buffer graph or deadlines could alter the useful coordination schedule; no adaptation or learning result is claimed.

## Actions, timing and rewards

- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Cycle nearby junction: straight, upper, lower.
- 6: Pick up or put down one part.
- 7: Start nearby machine rework: three turns and two power.
- 8: Start nearby machine color flip: two turns and one power.
- 9: Publish nearby board recipes or inspect a nearby part.
- 10: Assemble first pending order at bench.
- 11: Pick up one ready product; carry to dock.
- 12: Toggle conveyor brake at nearby control.

Every indexed intent is legal for a live capable role, even when its spatial/resource precondition fails; failed intents become deterministic no-ops. Out-of-range actions become wait. Terminal roles, and temporarily/incapacitated roles where specified, retain wait only. Masks do not leak private feasibility. Targets outside the map, duplicate requested destinations, or another role current cell are rejected. No swaps or following into a just-vacated occupied cell occur in that transition.

Reward expressions are literal in `advance` and are returned per role as float32. The manifest bounds are conservative finite interface bounds, not performance or acceptance gates. No return, score, success rate, optimality, statistical result or learning result was measured.

## Native observations and rendering

Partial native information with public geometry, positions, buffers, custody, processing, order deadlines, visible colors, status and events. The six actual defect bits are visible to role 1 or to everyone after part inspection/rework; requested coil colors are visible to role 2 or after a board publication. Explicit per-record mask rows accompany -1 for unknown bits. Separate quality/recipe entity groups use exactly those masks; the public part/order groups never expose the private attributes.

Every flat observation column is covered by exactly one `observation_fields` slice. `state_visibility_mapping` accounts for every native leaf. The scene binds the exact source and manifest hashes, actual tile map, actual entity positions, activity, native role masks and implemented attributes. Per-step flags, cumulative counters and levels are declared separately. A privileged full-state debug view must be labeled separately by the later renderer; it is not a policy view.

## Termination and numerical boundary

All three product statuses are resolved as shipped (2) or rejected (3). Rejected orders are permanent negative outcomes. Ready but undelivered products do not count as completion. Otherwise truncate at 144 live transitions.

Natural termination takes priority if it coincides with the horizon. `truncated = (next_t >= H) & ~terminated`; `done` equals their disjunction. A terminal raw call retains every state leaf, observation, clock and cause flag and returns zero rewards and all dones true. There is no internal reset. The independent wrapper is responsible for autoreset. State integers use int32, flags bool, and observations/rewards float32. No small signed-byte design distribution was replaced; the reset uint32 key stays external.

## Key schedule and witness status

reset first calls keys = jax.random.split(key, 2), then draws part_defect with randint(keys[0], (6,), 0, 2, dtype=int32) and order_color with randint(keys[1], (3,), 0, 2, dtype=int32). raw_step is deterministic and does not use its supplied key.

Replay must use `reset(jax.random.PRNGKey(seed))`. Zero-based absolute raw step k receives `jax.random.fold_in(jax.random.PRNGKey(seed), k + 1)`. Paired suffixes continue at prefix length L, with identical reset and prefix keys in both branches. Neither scripts nor pairs have been executed, searched or optimized. The JSON lists are literal intended actions, not recorded trajectories. Any actual illegality, failed objective or mismatch must remain in the later receipt.

## Source line evidence

- `refresh`: `candidate.py:12-34`; literal declaration `def refresh(self, s):`.
- `reset`: `candidate.py:36-86`; literal declaration `def reset(self, key):`.
- `observe`: `candidate.py:88-100`; literal declaration `def observe(self, s):`.
- `legal_actions`: `candidate.py:102-103`; literal declaration `def legal_actions(self, s):`.

`candidate.py:120-123`

```python
        # MACHINE_CLOCKS: existing jobs finish before new orders; no instant repair or paint.
        finished = (s["station_timer"] == 1) & (s["station_part"] >= 0)
        n["station_timer"] = jnp.maximum(s["station_timer"] - 1, 0)
        for station in range(2):
```

`candidate.py:133-136`

```python
        # HANDOFF_ORDER: cyclic worker priority serializes contested picks, drops and machines.
        for offset in range(4):
            i = (s["t"] + offset) % 4
            at_control = jnp.sum(jnp.abs(n["agent_pos"][i] - s["control_pos"][0])) <= 1
```

`candidate.py:177-180`

```python
        # ASSEMBLY: first pending order consumes one available chassis and one coil exactly once.
        near_bench = jnp.sum(jnp.abs(n["agent_pos"] - s["bench_pos"][0]), axis=1) <= 1
        workers = near_bench & (a == 10)
        priority = (jnp.arange(4) - s["t"]) % 4
```

`candidate.py:198-201`

```python
        # PRODUCT_CARRYING: one carried part or product per worker; delivery uses the real dock.
        for offset in range(4):
            i = (s["t"] + offset) % 4
            holding = jnp.any(n["part_carrier"] == i) | jnp.any(n["product_carrier"] == i)
```

`candidate.py:214-217`

```python
        # CONVEYOR_QUEUE: one-cell buffers, pre-move occupancy and cyclic part tie-breaking.
        n = self.refresh(n)
        on_node = jnp.all(n["part_pos"][:, None, :] == s["belt_pos"][None, :, :], axis=2)
        node = jnp.argmax(on_node.astype(jnp.int32), axis=1)
```

`candidate.py:248-251`

```python
        # ABSORBING_TERMINAL: terminal observations and native state remain unchanged.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((4,), dtype=jnp.float32), rewards)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
```

## Supplied public reference attribution

- `references/JaxMARL_README.md` (6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15): Parallel agent-keyed boundary and __all__ conventions read as API context. The authored flat kernel does not implement the separate public adapter. No game code copied.
- `references/Pgx_README.md` (153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df): Explicit state, legal-mask and externally supplied random-key API context. No board rules, layouts or source code copied.
- `references/Craftax_README.md` (7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff): Spatial gameplay, native symbolic observation/rendering and explicit reset-boundary context. No Craftax mechanics or code copied.
- `references/PureJaxRL_README.md` (d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0): Algorithm-library identity checked; no game rule source and no replacement for the frozen learner.
- `references/Craftax_Baselines_ppo_rnn.py` (5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011): Read its imports, recurrent-policy/reset handling and PureJaxRL attribution as integration context only. No baseline code copied or executed.

All substantive game rules and kernels were authored for this batch. No upstream source blocks were copied. Only the supplied local official reference snapshots were read; no network lookup, external benchmark implementation, earlier candidate, training outcome or final-test evidence was used. The public JaxMARL adapter, executable replay, animation and learning evaluation remain separate integration work; this source package does not claim those deliveries.
