# Prism Commons — game-v3-00098

Three mobile workers share a 9 by 9 optical workshop. Their own receivers are north (role 0), east (role 1) and south (role 2). Seven mirrors occupy real board cells; one emitter launches eastward from (0,4), and a pump at (1,7) replenishes the same energy pool used by everyone. The task is three deliveries to every receiver within at most 96 nonterminal steps.

## Rules and temporal consequences

Actions 0–4 wait or move one cell north/east/south/west. Action 5 rotates the nearest mirror within Manhattan distance one through transparent, slash and backslash, spending one common energy and locking it against manipulation for the next two calls. Action 6 spends one energy to clamp the current orientation for five subsequent calls. Lowest mirror index resolves equal distances. A mirror still locked at the input of a call cannot be manipulated on that call, even if its timer declines to zero.

Action 7 launches only while adjacent to the emitter, no other pulse is active, and at least three energy remain. Action 8 pumps up to two energy while adjacent to the battery, capped at twenty. A launch costs the acting role one reward; turns and clamps cost energy, without an extra private reward penalty. Contested operations use rotating role priority. No worker owns a particular mirror.

A live pulse, including a newly launched pulse, moves exactly one cell each call. Direction indices are east/south/west/north. Slash reflection maps d to 3-d; backslash uses [1,0,3,2]. Transparent mirrors preserve direction. Entering any receiver consumes the pulse; only an unfilled slot gains charge. Board exit or exhaustion of the forty-cell propagation budget also ends the pulse, preserving its last in-bounds position. Workers do not block light. The off-axis mirrors permit detours and loops, but the TTL prevents an unbounded loop.

Each new delivery pays four to its receiver's role and one to its launcher. Filling all receivers pays four to everyone and ends naturally. The last state is preserved; no raw auto-reset occurs. A pulse reaching an already full receiver is consumed with no delivery reward.

## Causal interdependence and strategy hypotheses

A worker at the central mirror can send another worker's in-flight pulse north, east or south. Its partner can lock that direction before the pulse arrives, or spend scarce shared energy on a conflicting turn. Pumping consumes a worker's time away from the central routing contest and makes later partners' launches feasible. These are changes to geometry, resources and private returns, not independent parallel navigation tasks.

Plausible approaches include time-sharing the central route, precommitting a useful optical configuration with clamps, or prioritizing one's own receiver while another role supplies the pump and launches. Off-axis mirror settings can create longer routes and timing windows. Their practical value is unmeasured. The paired witness changes only role 2's central-mirror action after a shared pulse prefix; it proposes an effect on roles 0 and 1.

All dynamic board information, attributes and event flags are public in the 73-column observation. The scene binds those actual arrays; beam coordinates are propagated state, not decorative interpolation. Mirror lock is a persistent level rather than a fresh turn event.

## Declared invariants and prospective variations

Manual movement keeps three distinct worker positions within the board. Energy remains in [0,20], mirror orientations in [0,2], locks in [0,5], receiver charges in [0,3], pulse TTL in [0,40], and the pulse position remains in bounds even after exiting. There is at most one active pulse; no reward is issued for a receiver slot twice. These are intended source invariants, not tested receipts.

A separately frozen variant could change recipient assignment, mirror coordinates, pumping access, launch/turn costs or commitment duration. Such changes would alter assistance, delay and denial incentives; this source implements one fixed instance only.

## Literal source evidence

**Shared-resource commitment**

```python
L102:             turn = (a[i] == 5) & near & unlocked & (q["energy"][0] > 0)
L103:             clamp = (a[i] == 6) & near & unlocked & (q["energy"][0] > 0)
L104:             q["mirror_dir"] = q["mirror_dir"].at[target].set(
L105:                 jnp.where(turn, (q["mirror_dir"][target] + 1) % 3, q["mirror_dir"][target]))
L106:             q["mirror_lock"] = q["mirror_lock"].at[target].set(
```

**Shared launch preconditions**

```python
L111:             fire = (a[i] == 7) & at_source & ~q["beam_active"][0] & (q["energy"][0] >= 3)
L112:             q["beam_active"] = q["beam_active"] | fire
L113:             q["beam_pos"] = jnp.where(fire, q["emitter_pos"], q["beam_pos"])
L114:             q["beam_dir"] = jnp.where(fire, jnp.zeros((1,), dtype=jnp.int32), q["beam_dir"])
L115:             q["beam_ttl"] = jnp.where(fire, jnp.array([40], dtype=jnp.int32), q["beam_ttl"])
L116:             q["beam_owner"] = jnp.where(fire, jnp.array([i], dtype=jnp.int32), q["beam_owner"])
L117:             q["energy"] = q["energy"] - 3 * fire.astype(jnp.int32)
```

**Actual one-cell propagation**

```python
L129:         proposed = n["beam_pos"][0] + directions[n["beam_dir"][0]]
L130:         inside = jnp.all((proposed >= 0) & (proposed <= 8))
L131:         live = n["beam_active"][0]
L132:         visit = live & inside
L133:         n["beam_pos"] = jnp.where(visit, proposed[None, :], n["beam_pos"])
```

**Bounded receiver effect**

```python
L143:         n["delivery_event"] = receiver_hit & (n["receiver_charge"] < 3)
L144:         n["receiver_charge"] = jnp.minimum(n["receiver_charge"] + n["delivery_event"].astype(jnp.int32), 3)
L145:         n["beam_ttl"] = jnp.where(live, jnp.maximum(n["beam_ttl"] - 1, 0), 0)
L146:         n["beam_active"] = jnp.array([visit & ~jnp.any(receiver_hit) & (n["beam_ttl"][0] > 0)], dtype=jnp.bool_)
```

**Private and shared payoffs**

```python
L152:         rewards = 4 * n["delivery_event"].astype(jnp.int32)
L153:         rewards = rewards + ((jnp.arange(3) == n["beam_owner"][0]) & jnp.any(n["delivery_event"])).astype(jnp.int32)
L154:         rewards = rewards - (jnp.arange(3) == firing_actor).astype(jnp.int32) + 4 * success.astype(jnp.int32)
```

**Absorbing raw state**

```python
L159:         next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
L160:         rewards = jnp.where(state["done"], jnp.zeros((3,), dtype=jnp.float32), rewards)
L161:         dones = jnp.full((3,), next_state["done"], dtype=jnp.bool_)
L162:         info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
L163:                 "truncated": next_state["truncated"]}
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
