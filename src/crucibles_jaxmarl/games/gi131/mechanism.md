# Firebreak Crews and Shared Rescue Routes

Author status: AUTHORED_NOT_EXECUTED. Candidate `game-v3-00131`, source attempt 0, train-only. Static structural screening is recorded separately in `static_receipt.json`; it is not numerical R0, gameplay acceptance, solvability, training or transfer evidence.

A/O/U/H = 3/514/10/120. Source SHA-256: `0b76969b21720d1b7319d9611873f0d4c0a58edd84e3da79ce1635892ec18194`.

## Task and causal interaction

Rescue six individually conserved survivors while protecting crews and managing finite water. Every rescue has shared value and extra value to its actual rescuer and the survivor-index priority role.

Cleared fuel blocks future orthogonal ignition, spraying changes immediate hazards, and directed water gifts change a partner resource budget. Survivor custody is exclusive and can be relayed by dropping then picking. A common finite tank, occupied cells and priorities couple rescue decisions.

Fuel and water interventions alter the same grid used for partner movement and civilian exposure. A crew can clear a future corridor while another carries a survivor through it; water spent on an immediate target is unavailable for a partner's later rescue. Waiting for a front to burn out costs survivor health rather than supplying a cost-free independent task. The central masonry strip creates different north/south approaches and the shared safe corner creates traffic conflicts.

Plausible strategy families include early firebreak construction, mobile spray-and-carry teams, tank-to-front water relays, and prioritizing short rescues while another crew protects distant civilians. Shared rewards do not erase role conflict: each role has an additional reward for two named survivor identities. These are source-grounded alternatives, not measured policy behavior or solvability claims.

The water identity is sum(water) + tank_stock[0] + spent_water == 15. Six survivor slots retain their identities throughout waiting, carried, rescued or lost states. A role holds at most one survivor; disabled carriers release custody. Fuel never replenishes. Fire spread uses four padded, nonwrapping neighbors, consumes current fuel, and ignites only on ticks divisible by three. The mechanism is deterministic, not wind sampling.

Tile 1 is nonflammable, while the four-cell rescue area is precisely x <= 1 and y <= 1. Clearing success flags may repeat on an already-cleared ground cell; they mean the legal clearing operation reached that cell, not that new fuel was removed. Resource transfers and actual hazard exposure remain distinct events.

Changing fuel placement, priority identities, tank location or spread cadence would change rescue timing and division of labor. Any such variant requires a new identified source, and no generalization evidence is claimed here.

## Actions, timing and rewards

- 0: Wait.
- 1: Move north and face north.
- 2: Move east and face east.
- 3: Move south and face south.
- 4: Move west and face west.
- 5: Clear fuel on the facing cell if unburning.
- 6: Spray one water unit onto the facing cell.
- 7: Pick up or drop one nearby survivor.
- 8: Take one unit from the nearby shared tank.
- 9: Give one water unit to the facing adjacent crew member.

Every indexed intent is legal for a live capable role, even when its spatial/resource precondition fails; failed intents become deterministic no-ops. Out-of-range actions become wait. Terminal roles, and temporarily/incapacitated roles where specified, retain wait only. Masks do not leak private feasibility. Targets outside the map, duplicate requested destinations, or another role current cell are rejected. No swaps or following into a just-vacated occupied cell occur in that transition.

Reward expressions are literal in `advance` and are returned per role as float32. The manifest bounds are conservative finite interface bounds, not performance or acceptance gates. No return, score, success rate, optimality, statistical result or learning result was measured.

## Native observations and rendering

Fully observable public fire, fuel, smoke, people, supplies and events. Flat cell arrays are row-major duplicates of the corresponding native grids; fixed cell positions are determined by the public 9-by-7 grid. Visibility masks are all true. Lost survivors become inactive; their loss and final health remain in the observation.

Every flat observation column is covered by exactly one `observation_fields` slice. `state_visibility_mapping` accounts for every native leaf. The scene binds the exact source and manifest hashes, actual tile map, actual entity positions, activity, native role masks and implemented attributes. Per-step flags, cumulative counters and levels are declared separately. A privileged full-state debug view must be labeled separately by the later renderer; it is not a policy view.

## Termination and numerical boundary

All six survivors are resolved as rescued or lost, or all three crew health values are zero. Surviving unresolved civilians remain in the final state when all crews are disabled. Otherwise truncate at 120 live steps.

Natural termination takes priority if it coincides with the horizon. `truncated = (next_t >= H) & ~terminated`; `done` equals their disjunction. A terminal raw call retains every state leaf, observation, clock and cause flag and returns zero rewards and all dones true. There is no internal reset. The independent wrapper is responsible for autoreset. State integers use int32, flags bool, and observations/rewards float32. No small signed-byte design distribution was replaced; the reset uint32 key stays external.

## Key schedule and witness status

Deterministic reset and deterministic orthogonal spread. Both supplied keys are unused. No random wind, agent-order draw or stochastic ignition is hidden in the implementation.

Replay must use `reset(jax.random.PRNGKey(seed))`. Zero-based absolute raw step k receives `jax.random.fold_in(jax.random.PRNGKey(seed), k + 1)`. Paired suffixes continue at prefix length L, with identical reset and prefix keys in both branches. Neither scripts nor pairs have been executed, searched or optimized. The JSON lists are literal intended actions, not recorded trajectories. Any actual illegality, failed objective or mismatch must remain in the later receipt.

## Source line evidence

- `refresh`: `candidate.py:12-26`; literal declaration `def refresh(self, s):`.
- `reset`: `candidate.py:28-65`; literal declaration `def reset(self, key):`.
- `observe`: `candidate.py:67-74`; literal declaration `def observe(self, s):`.
- `legal_actions`: `candidate.py:76-78`; literal declaration `def legal_actions(self, s):`.

`candidate.py:103-106`

```python
        # WATER_AND_BREAKS: simultaneous spray spends each actor's unit, then clears resolve.
        spraying = (a == 6) & (s["water"] > 0) & inside
        sprays = jnp.zeros((7, 9), dtype=jnp.int32).at[q[:, 1], q[:, 0]].add(spraying.astype(jnp.int32))
        fire = jnp.where(sprays > 0, 0, s["fire"])
```

`candidate.py:117-120`

```python
        # ROTATING_INTERACTIONS: unique ownership, finite tank and directed water handoffs.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            holding = jnp.any(carrier == i)
```

`candidate.py:138-141`

```python
        # RESCUE: any carried survivor entering the four-cell safe corner is resolved once.
        carried = (carrier >= 0) & (carrier < 3)
        survivor_pos = jnp.where(carried[:, None], n["agent_pos"][jnp.clip(carrier, 0, 2)], survivor_pos)
        rescued = carried & (survivor_pos[:, 0] <= 1) & (survivor_pos[:, 1] <= 1)
```

`candidate.py:145-148`

```python
        # SPREAD_ORDER: deterministic orthogonal fronts every third turn, without wrapping.
        burning = fire > 0
        fuel = jnp.maximum(fuel - burning.astype(jnp.int32), 0)
        ignition = ((n["t"] % 3 == 0) & (self.neighbors(burning.astype(jnp.int32)) > 0)
```

`candidate.py:154-157`

```python
        # CONSEQUENCES: protection changes partners' exposure; disabled carriers release cargo.
        p = n["agent_pos"]
        injury = ((fire[p[:, 1], p[:, 0]] > 0) | (smoke[p[:, 1], p[:, 0]] >= 3)) & (s["agent_health"] > 0)
        health = s["agent_health"] - injury.astype(jnp.int32)
```

`candidate.py:187-190`

```python
        # ABSORBING_TERMINAL: no auto-reset and no clock advance after completion.
        n = {k: jnp.where(state["done"], state[k], n[k]) for k in state}
        rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
        info = {"done_all": n["done"], "terminated": n["terminated"], "truncated": n["truncated"]}
```

## Supplied public reference attribution

- `references/JaxMARL_README.md` (6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15): Parallel agent-keyed boundary and __all__ conventions read as API context. The authored flat kernel does not implement the separate public adapter. No game code copied.
- `references/Pgx_README.md` (153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df): Explicit state, legal-mask and externally supplied random-key API context. No board rules, layouts or source code copied.
- `references/Craftax_README.md` (7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff): Spatial gameplay, native symbolic observation/rendering and explicit reset-boundary context. No Craftax mechanics or code copied.
- `references/PureJaxRL_README.md` (d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0): Algorithm-library identity checked; no game rule source and no replacement for the frozen learner.
- `references/Craftax_Baselines_ppo_rnn.py` (5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011): Read its imports, recurrent-policy/reset handling and PureJaxRL attribution as integration context only. No baseline code copied or executed.

All substantive game rules and kernels were authored for this batch. No upstream source blocks were copied. Only the supplied local official reference snapshots were read; no network lookup, external benchmark implementation, earlier candidate, training outcome or final-test evidence was used. The public JaxMARL adapter, executable replay, animation and learning evaluation remain separate integration work; this source package does not claim those deliveries.
