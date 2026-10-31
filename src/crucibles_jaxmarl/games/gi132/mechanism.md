# Lightkeepers and Shared Shadow Corridors

Author status: AUTHORED_NOT_EXECUTED. Candidate `game-v3-00132`, source attempt 0, train-only. Static structural screening is recorded separately in `static_receipt.json`; it is not numerical R0, gameplay acceptance, solvability, training or transfer evidence.

A/O/U/H = 3/316/13/128. Source SHA-256: `00c7314e75f48ee9438cb033535b2a7ce28d8be8fe51f9d2f19f9152f6ea4f8e`.

## Task and causal interaction

Retrieve the three finite coded caches under real illumination and carry them to the central socket, while retaining safe shadow crossings and enough shared lamp charge. Cache owners have different delivery deadlines.

Rotating or switching a shared lamp changes other roles illumination, native visibility, safe crossing and cache access. Shutters occlude real ray segments and walking, portable mirrors turn rays, and finite packs/socket stock fund future partner opportunities. Captures can repeat unless partners darken the occupied watched cell.

The two lamps initially illuminate different horizontal routes, while the third cache needs a different orientation. A lamp keeper can help a partner read or open a cache while exposing another partner on a watched column. Turning the lamp away creates a safe crossing but removes that cache's illumination. Shutters provide local occlusion and portable mirrors can redirect a bounded beam around a different approach. Finite lamp charge, carried packs and the shared socket create opportunity cost across these schedules.

Plausible strategies include alternating light and crossing phases, rotating the eastern lamp only after another role has collected its horizontal-route cache, moving a mirror to preserve illumination while changing an exposed crossing, and dedicating a keeper to refills. These strategies are hypotheses. A direct dark path is allowed; that can coexist with a need to illuminate a cache for access. No single route or solution is hard-coded as a success claim.

The stored beam is the actual emission for the just-completed turn, used for pickup, exposure and visibility. It is computed with the available pre-payment charge, then the charge is deducted. Thus the last paid emission may remain displayed alongside a zero post-payment battery; it is not an invented extra animation. The next raw step recomputes emission. Reset displays the initial geometry without consuming a simulation step or charge.

Each lamp records a boolean set of visited cells, so a reflected loop counts that lamp once per cell and terminates after twelve segments. Closed shutters absorb before marking their cell. Held mirrors do not reflect. Energy accounting is sum(lamp_energy) + sum(pack) + socket_stock[0] + spent_energy == 51. Caches retain their identities through waiting, carrying, dropping and delivery.

Alarm flags are intentionally public even when the underlying agent or cache position is hidden. Raw hidden positions and codes do not appear in public scene attributes; their entity group's native visibility is the same mask used by observe. Masks are not a lighting renderer approximation. Different shutter placement, mirror orientation, deadline ownership or private codes could change coordination requirements, but no measured adaptation is claimed.

## Actions, timing and rewards

- 0: Wait.
- 1: Move north.
- 2: Move east.
- 3: Move south.
- 4: Move west.
- 5: Rotate nearby lamp clockwise.
- 6: Toggle nearby lamp switch.
- 7: Toggle nearby shutter.
- 8: Pick up or deploy one portable mirror.
- 9: Flip held or nearby mirror orientation.
- 10: Charge nearby lamp or refill pack at socket.
- 11: Enter code zero on a lit nearby cache.
- 12: Enter code one on a lit nearby cache.

Every indexed intent is legal for a live capable role, even when its spatial/resource precondition fails; failed intents become deterministic no-ops. Out-of-range actions become wait. Terminal roles, and temporarily/incapacitated roles where specified, retain wait only. Masks do not leak private feasibility. Targets outside the map, duplicate requested destinations, or another role current cell are rejected. No swaps or following into a just-vacated occupied cell occur in that transition.

Reward expressions are literal in `advance` and are returned per role as float32. The manifest bounds are conservative finite interface bounds, not performance or acceptance gates. No return, score, success rate, optimality, statistical result or learning result was measured.

## Native observations and rendering

Partial native observation. Geometry, wired devices, complete beam/shadow raster, socket stock, deadlines and alarm flags are public. Agents are visible within Manhattan distance two or when their cell is lit, plus self. Mirrors are visible by the same distance/light rule or to their carrier. Cache positions, code, status and carrier are visible within distance two, when lit, or to their same-index owner through its tracker. Unseen record fields are -1 and each record has its explicit role-specific visibility bit. No other-role mask rows enter that role observation.

Every flat observation column is covered by exactly one `observation_fields` slice. `state_visibility_mapping` accounts for every native leaf. The scene binds the exact source and manifest hashes, actual tile map, actual entity positions, activity, native role masks and implemented attributes. Per-step flags, cumulative counters and levels are declared separately. A privileged full-state debug view must be labeled separately by the later renderer; it is not a policy view.

## Termination and numerical boundary

All three caches delivered, or all roles have three exposures. Otherwise truncate after 128 live transitions. Stunned or disabled roles may only wait, but the episode can continue for partners.

Natural termination takes priority if it coincides with the horizon. `truncated = (next_t >= H) & ~terminated`; `done` equals their disjunction. A terminal raw call retains every state leaf, observation, clock and cause flag and returns zero rewards and all dones true. There is no internal reset. The independent wrapper is responsible for autoreset. State integers use int32, flags bool, and observations/rewards float32. No small signed-byte design distribution was replaced; the reset uint32 key stays external.

## Key schedule and witness status

reset draws cache_code = jax.random.randint(key, (3,), 0, 2, dtype=int32) directly from the supplied reset key. No split or fold-in occurs inside reset. raw_step is deterministic and does not use its supplied key.

Replay must use `reset(jax.random.PRNGKey(seed))`. Zero-based absolute raw step k receives `jax.random.fold_in(jax.random.PRNGKey(seed), k + 1)`. Paired suffixes continue at prefix length L, with identical reset and prefix keys in both branches. Neither scripts nor pairs have been executed, searched or optimized. The JSON lists are literal intended actions, not recorded trajectories. Any actual illegality, failed objective or mismatch must remain in the later receipt.

## Source line evidence

- `rays`: `candidate.py:12-32`; literal declaration `def rays(self, s):`.
- `refresh`: `candidate.py:34-51`; literal declaration `def refresh(self, s):`.
- `reset`: `candidate.py:53-94`; literal declaration `def reset(self, key):`.
- `observe`: `candidate.py:96-110`; literal declaration `def observe(self, s):`.
- `legal_actions`: `candidate.py:112-114`; literal declaration `def legal_actions(self, s):`.

`candidate.py:19-22`

```python
        # RAY_GEOMETRY: twelve bounded segments per lamp; closed shutters absorb rays.
        for segment in range(12):
            position = position + directions[direction]
            inside = jnp.all((position >= 0) & (position <= jnp.array([10, 6])), axis=1)
```

`candidate.py:133-136`

```python
        # DEVICE_INTERACTIONS: cyclic actor priority; toggles compose and resources transfer once.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            lamp_d = jnp.sum(jnp.abs(s["lamp_pos"] - n["agent_pos"][i]), axis=1)
```

`candidate.py:166-169`

```python
        # EMISSION: the stored beam is the actual just-emitted light, before paying its charge.
        n["beam"] = self.rays(n)
        consumed = (n["lamp_on"] & (n["lamp_energy"] > 0)).astype(jnp.int32)
        n["lamp_energy"] = n["lamp_energy"] - consumed
```

`candidate.py:172-175`

```python
        # INFORMATION_AND_PICKUP: code entry requires real illumination and free hands.
        for offset in range(3):
            i = (s["t"] + offset) % 3
            close = jnp.sum(jnp.abs(n["cache_pos"] - n["agent_pos"][i]), axis=1) <= 1
```

`candidate.py:189-192`

```python
        # WATCHED_CROSSINGS: partners can darken the cell to prevent repeated exposure.
        p = n["agent_pos"]
        capture = (s["tile"][p[:, 1], p[:, 0]] == 1) & (n["beam"][p[:, 1] * 11 + p[:, 0]] > 0) & (s["exposure"] < 3)
        n["exposure"] = s["exposure"] + capture.astype(jnp.int32)
```

`candidate.py:210-213`

```python
        # ABSORBING_TERMINAL: exact state retention, zero reward, original cause flags.
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
