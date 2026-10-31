# Counterweight Yard — game-v3-00101

Three workers share a 9 by 9 side-view yard, nine real blocks, two lifts and a ground crank. Gravity points toward increasing screen y. Each role benefits most from its corresponding work station: west, east and middle require one, one and two blocks. The other blocks can remain counterweights or be bolted into bridge support. This finite common material stock makes those uses compete.

## Motion, load and work sequence

Actions 0–4 wait or move. Solid map cells, bolted blocks and lift platforms block manual motion. Ladders support a worker against gravity, and an empty-handed worker can climb them. A worker carrying any block cannot climb upward; the shared lifts provide upward cargo transport. Loose waist-high cargo can share a cell with a worker, while bolted cargo is solid support.

Action 5 lifts loose cargo on the worker's cell. If already carrying, it installs cargo at any unfinished work station on that cell, or drops it elsewhere. No private ownership restricts this common stock. Action 6 bolts a held block into one of (3,4), (4,4), (5,4) when the worker is on row 3 within one horizontal cell and that socket has a solid horizontal neighbor. The result is an actual solid cell below the walking row. Once bolted into the rail it remains supported until reclaimed. Action 10 reclaims the nearest bolted block within Manhattan distance two when empty-handed; this immediately removes that support.

The left and right lift platforms occupy columns 2 and 6 with y in [4,8]. Workers and cargo on the cell immediately above a platform add their weight: one per worker, plus each active unbolted block's mass, including carried blocks. Resting unbolted blocks on the opposite ground ballast pad support the lift. A worker actively bracing on that opposite pad contributes two more support units. Held blocks do not count as resting ballast.

At the shared crank, action 7 raises the left lift one cell and action 8 raises the right, only if counterweight is positive and at least the current load. Actions 11 and 12 lower the corresponding platform safely one cell. Action 9 on a ballast pad supplies live ballast for that tick; at the crank it instead applies the two-tick shared brake, preventing both commanded and automatic motion. Without braking or controlled lowering, an elevated underbalanced platform drops one cell and jolts its riders. A partner who stops bracing or picks up ballast can cause that change under the same operator action.

Platform riders and loose platform cargo move with the platform. Thereafter each loose unbolted block can fall at most one cell, using rotating block priority and actual terrain, platforms, bolted cells or resting cargo below it as support. A falling block entering a worker's current cell, or an automatic platform drop with that worker aboard, sets stun two. Worker gravity then moves unsupported nonladder workers down at most one cell. Physics can co-locate workers during a fall; that is permitted and no unique-position invariant is asserted. Stunned workers have only no-op until their input timer becomes zero; physical gravity and impacts still occur.

## Payoffs and causal alternatives

Each transition costs every role one. Each newly installed unit pays four to its station's beneficiary, one to every role, and one to the actual delivering worker. Each impacted worker loses two. Joint completion adds three to everyone and ends naturally; otherwise the horizon is 160. There is no reward for repeatedly reclaiming and rebuilding a bridge.

A crank operator can help another worker reach an upper station, or brake at a time that delays that worker's delivery. A ballast worker can hold the platform stable, remove a useful block for private construction, or substitute live bracing while materials are reassigned. Reclaiming a bridge support for an earlier private installation can remove a partner's walking surface and deny the middle station. Safe sequential work, synchronized ballast substitution, and early station prioritization are plausible alternatives whose practical merit is not measured.

The paired script holds worker 0's no-op and worker 2's crank operation fixed while worker 1 either leaves or lifts the right-pad counterweight. It proposes a rise versus an automatic drop of worker 0's left platform. This is a source-based hypothesis, not a physics replay.

## Presentation, invariants and prospective variants

All changing physical state is public in 141 columns. Displayed lift positions are solid platform cells, and worker/cargo coordinates represent their actual positions above them. The scene binds load, support, carrier, mass, anchored status, work counts, brake and actual per-step events. Automatic drops and controlled descents have distinct event channels.

Nine block identities are conserved: active blocks plus station_count sum to nine. Every worker carries at most one. Bolted blocks lie in the three bridge sockets, are uncarried and remain active. Installed blocks become inactive. All positions stay inside [0,8] by [0,8], workers and blocks stay at y at most seven, platform y remains [4,8], brake and stun remain [0,2], and station counts never exceed [1,1,2]. These are declared source invariants awaiting numerical checking.

Prospective variants could alter pad placement, block masses, station ownership, bridge socket layout, operator access or support requirements. The authored instance does not sample those variants or demonstrate transfer.

## Literal source evidence

**Other workers and physical cargo determine support**

```python
L29:         fixed_mass = s["block_weight"] * (s["block_active"] & (s["block_carrier"] < 0) & ~s["block_anchored"]).astype(jnp.int32)
L30:         human_pad = jnp.all(s["agent_pos"][:, None, :] == s["ballast_pos"][None, :, :], axis=2)
L31:         ballast = (jnp.sum(on_pad.astype(jnp.int32) * fixed_mass[:, None], axis=0, dtype=jnp.int32)
L32:                    + 2 * jnp.sum(human_pad.astype(jnp.int32) * s["braced"].astype(jnp.int32)[:, None], axis=0, dtype=jnp.int32))
L33:         return load, ballast[::-1]
```

**Upward cargo needs lifts**

```python
L150:             climb = (a[i] != 1) | ((s["carrying"][i] < 0) & (here_ladder | there_ladder))
L151:             occupied = jnp.any(jnp.all(pos == target, axis=1) & (jnp.arange(3) != i))
L152:             solid = self.solid(n, target[None, :])[0]
L153:             ok = able[i] & inside & climb & ~occupied & ~solid
```

**Real support cells are constructed**

```python
L184:             eligible = (dx <= 1) & (q["agent_pos"][i, 1] == 3) & ~self.solid(q, bridge) & adjacent_support
L185:             socket = jnp.argmin(jnp.where(eligible, dx, 99))
L186:             build = able[i] & (a[i] == 6) & has & jnp.any(eligible)
L187:             q["block_pos"] = q["block_pos"].at[good].set(jnp.where(build, bridge[socket], q["block_pos"][good]))
L188:             q["block_carrier"] = q["block_carrier"].at[good].set(jnp.where(build, -1, q["block_carrier"][good]))
L189:             q["block_anchored"] = q["block_anchored"].at[good].set(q["block_anchored"][good] | build)
L190:             q["build_event"] = q["build_event"].at[good].set(q["build_event"][good] | build)
```

**Reclaiming removes support**

```python
L194:             reclaim = able[i] & (a[i] == 10) & ~has & jnp.any(bolts)
L195:             q["block_anchored"] = q["block_anchored"].at[anchor].set(q["block_anchored"][anchor] & ~reclaim)
L196:             q["block_carrier"] = q["block_carrier"].at[anchor].set(jnp.where(reclaim, i, q["block_carrier"][anchor]))
L197:             q["block_pos"] = q["block_pos"].at[anchor].set(jnp.where(reclaim, q["agent_pos"][i], q["block_pos"][anchor]))
L198:             q["reclaim_event"] = q["reclaim_event"].at[anchor].set(q["reclaim_event"][anchor] | reclaim)
```

**Safe command versus underbalanced fall**

```python
L215:         controlled_down = lowering & (n["lift_pos"][:, 1] < 8) & free
L216:         down = (support < load) & (n["lift_pos"][:, 1] < 8) & free & ~controlled_down
L217:         shift = (down | controlled_down).astype(jnp.int32) - up.astype(jnp.int32)
L218:         old_tops = n["lift_pos"] - jnp.array([0, 1], dtype=jnp.int32)
```

**Riders actually move with the lift**

```python
L220:         n["agent_pos"] = n["agent_pos"].at[:, 1].add(jnp.sum(riders.astype(jnp.int32) * shift[None, :], axis=1, dtype=jnp.int32))
L221:         cargo_riders = (jnp.all(n["block_pos"][:, None, :] == old_tops[None, :, :], axis=2)
L222:                          & (n["block_carrier"] < 0)[:, None] & n["block_active"][:, None] & ~n["block_anchored"][:, None])
L223:         n["block_pos"] = n["block_pos"].at[:, 1].add(jnp.sum(cargo_riders.astype(jnp.int32) * shift[None, :], axis=1, dtype=jnp.int32))
L224:         n["lift_pos"] = n["lift_pos"].at[:, 1].add(shift)
L225:         n["lift_up_event"] = up
L226:         n["lift_down_event"] = controlled_down
```

**Discrete gravity and impacts**

```python
L238:             fall = q["block_active"][g] & (q["block_carrier"][g] < 0) & ~q["block_anchored"][g] & (q["block_pos"][g, 1] < 7) & ~supported
L239:             q["block_pos"] = q["block_pos"].at[g].set(jnp.where(fall, below, q["block_pos"][g]))
L240:             q["block_fall_event"] = q["block_fall_event"].at[g].set(fall)
L241:             hit = fall & jnp.all(q["agent_pos"] == below, axis=1)
L242:             q["impact_event"] = q["impact_event"] | hit
```

**Gravity and lost actions persist**

```python
L248:         fall_agent = (n["agent_pos"][:, 1] < 7) & ~on_ladder & ~self.solid(n, below_agents)
L249:         n["agent_pos"] = n["agent_pos"].at[:, 1].add(fall_agent.astype(jnp.int32))
L250:         n["stun"] = jnp.maximum(n["stun"], 2 * n["impact_event"].astype(jnp.int32))
L251:         n["block_pos"] = jnp.where((n["block_carrier"] >= 0)[:, None],
```

## Author boundary and provenance

This package is authored, frozen and **not executed**. Only the assigned batch's eight design/request files, delivery requirements, six contract/interface files and five supplied official reference files were inputs. There was no candidate import, JAX/NumPy import, compiler, contract validator, simulator, training, SSH or network operation. The only local source check is Python AST parsing; JSON parsing and SHA-256 establish file syntax and identity, not runtime validity.

All dynamics and integer design fields use fixed-shape int32/bool arrays; observations and rewards are float32. The supplied reset and step keys are unused because this original instance is deterministic. The key is not duplicated into state. Natural completion has priority at the horizon. A terminal raw call preserves every state leaf and observation, keeps the clock unchanged, returns zero reward and preserves its cause flags. The outer training wrapper remains responsible for resetting.

Unavailable contextual intents and blocked movement are legal attempts with no effect. Out-of-range or masked actions become wait. Normal manual movement uses rotating role priority (clock modulo the role count), refuses occupied target cells and has no implicit swap. This is a public rule. Every script and paired branch in playthrough.json is UNEXECUTED_AUTHOR_HYPOTHESIS; it contains actions and intended differences only, with no invented rollout states, returns or successful-completion claims.

## Supplied reference use

The supplied official JaxMARL README informed the distinction between this fixed-array kernel and an external agent-keyed parallel API adapter with an __all__ done entry. The supplied Pgx README informed explicit legal-action and terminal-state metadata and the separation of caller keys from hidden state. The supplied Craftax README informed the goal of a persistent spatial world with real objects, resource use and visible consequences. No existing game map, benchmark implementation, game rules or upstream code was copied.

PureJaxRL and the supplied Craftax_Baselines PPO-RNN file were read as algorithm/integration references only. They are not game libraries and are not incorporated here. This package does not replace the frozen experimental learner or implement the coordinator's separate community adapter and replay runtime. All five input snapshots and their actual hashes are recorded in generation_record.json.

Reference attribution: JaxMARL (Rutherford et al., NeurIPS Datasets and Benchmarks 2024), Pgx (Koyamada et al., NeurIPS 2023), Craftax (Matthews et al., ICML 2024), and PureJaxRL/Discovered Policy Optimisation (Lu et al., NeurIPS 2022). The Craftax PPO-RNN integration source credits Chris Lu's PureJaxRL implementation. No license from those references is asserted to cover this new author's own rules or source.

## Scientific limits

The intended strategies and interventions are hypotheses from source reasoning. Static syntax and real future causal replay do not establish learnability, optimality, solvability, strategic richness, non-isomorphism, generalization or meta-learning transfer. Negative, failed, zero-effect and indeterminate validation outcomes must remain visible. No reward, success-count, percentage or seed-count acceptance gate is introduced. Rule, role and layout variations mentioned below are prospective design possibilities, not implemented sampled task distributions or measured results.
