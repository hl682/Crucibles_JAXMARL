# Lantern Courtyard — game-v3-00099

Two controlled wardens (roles 0 and 1) and two controlled couriers (roles 2 and 3) occupy a 9 by 7 courtyard. A three-cell central wall separates upper and lower approaches. Couriers must collect the two pennants and each exit with one at (8,3). Wardens seek to freeze every courier still inside. Both teams' actions affect the other side's future action set and opportunities.

## Transition order and role actions

Actions 0–4 wait or move. Input-frozen or escaped roles have only wait. Movement is followed by warden shelter seals, courier cover expenditure, courier interaction, then warden tagging. Finally carried pennants follow their actual carriers, termination is determined, and visibility is recomputed.

For a warden, action 5 tags the nearest eligible courier at Manhattan distance at most one; equal distances choose the lower courier index. A courier is eligible only when not escaped, not already frozen, without rescue grace and without cover. Tagging freezes them for six ticks, drops any carried pennant at their current cell and clears their inventory. Timers decline once per call; a courier frozen at input cannot act on the call that brings its timer to zero. Retagging after release is possible.

For a courier, action 5 first rescues an adjacent frozen teammate; otherwise it exits while at the gate with a pennant, or picks a loose pennant on its own cell when empty-handed. Rescue clears the teammate's freeze timer and grants grace two, protecting that transition and the following transition. The rescued player does not gain an extra same-tick action. Escape makes the pennant inactive and leaves the escaped worker visible as an inert role; it no longer blocks movement and cannot rescue.

Action 6 seals a shelter for a warden within distance one, or spends one cover charge for a courier standing exactly on an unsealed shelter. Wardens' seals take precedence that tick. Cover lasts only for the current tag phase; the level is visible in the resulting state. Each shelter starts with four charges. Every sixth tick it regains one, capped at four, only if it was unsealed at input. Cover therefore creates shared resource and timing dependence.

The natural endings are both couriers escaped, or every courier that has not escaped is frozen. The latter can occur before the six-tick timers expire; if one courier already exited, tagging the other ends the remaining chase. Otherwise the horizon is 96. This is a team contest with possible mixed earlier escapes, not an assertion that either side has a winning strategy.

## Payoffs, information and partner dependence

Each successful tag pays its warden two and penalizes its target two. A rescue pays one to the rescuer and one to the rescued courier. Each escape pays its courier five, the other courier two, and costs each warden two. A joint escape adds three to both couriers and subtracts three from both wardens. A capture ending does the reverse. Those integer event payoffs are the returned rewards; there is no measured outcome in this document.

Teammate positions and status are always radio-visible. Enemy workers are visible only within Manhattan distance three; walls do not occlude this proximity sensor. Loose pennants use that radius; a teammate-carried pennant is visible to its team. Delivered pennants are inactive. Hidden coordinates and attributes are masked, with explicit visibility bits. The public map, shelters and their resource levels, gate, and identity-only event announcements remain available to all roles. Public resource changes can themselves support inference; no additional secret positional feed is provided to the role renderer.

A courier can divert a warden so a partner crosses, spend a shared charge, return to rescue, or escape early and remove the team's remaining rescue option. Wardens can split approaches, coordinate a freeze window, or spend a turn sealing cover while the other tags. These alternatives depend on partners' timing and opponents' actions. The proposed paired scripts separately vary a warden's tag and a courier's rescue; their predicted mask, cargo and protection differences await real execution.

## Declared invariants and prospective variations

The two wardens never acquire pennants or become frozen. A courier carries at most one pennant; an active carried pennant has exactly one courier owner. Frozen timers remain in [0,6], grace in [0,2], charges in [0,4] and seals in [0,3]. Escaped courier positions may overlap later entrants at the gate because they are nonblocking; unique active positions are deliberately not asserted for that group. All physical positions remain in map bounds.

Prospective variants could change wall gaps, shelter placement and regeneration, sensing radius, team composition or the rule for leaving a partner behind. None is sampled or evaluated here.

## Literal source evidence

**Actual proximity and teammate visibility**

```python
L16:         n["agent_vis"] = (teams[:, None] == teams[None, :]) | (distance <= 3)
L17:         flag_distance = jnp.sum(jnp.abs(s["agent_pos"][:, None, :] - s["flag_pos"][None, :, :]), axis=2, dtype=jnp.int32)
L18:         carrier_team = teams[jnp.clip(s["flag_owner"], 0, 3)]
L19:         friendly_carrier = (s["flag_owner"][None, :] >= 0) & (teams[:, None] == carrier_team[None, :])
L20:         n["flag_vis"] = ((flag_distance <= 3) | friendly_carrier) & s["flag_active"][None, :]
```

**Inactive-role action mask**

```python
L89:         able = (s["frozen"] == 0) & ~s["escaped"] & ~s["done"]
L90:         return able[:, None] | (jnp.arange(7)[None, :] == 0)
```

**Rescue changes partner state**

```python
L153:             rescue = act & adjacent & (q["frozen"][mate] > 0) & ~q["escaped"][mate]
L154:             q["frozen"] = q["frozen"].at[mate].set(jnp.where(rescue, 0, q["frozen"][mate]))
L155:             q["grace"] = q["grace"].at[mate].set(jnp.where(rescue, 2, q["grace"][mate]))
L156:             q["rescue_event"] = q["rescue_event"].at[mate].set(q["rescue_event"][mate] | rescue)
L157:             credit = credit.at[i].add(rescue.astype(jnp.int32))
```

**Tag removes opportunities and drops cargo**

```python
L182:             hit = (a[i] == 5) & jnp.any(candidates)
L183:             q["frozen"] = q["frozen"].at[target].set(jnp.where(hit, 6, q["frozen"][target]))
L184:             q["tag_event"] = q["tag_event"].at[target].set(q["tag_event"][target] | hit)
L185:             credit = credit.at[i].add(hit.astype(jnp.int32))
L186:             dropped = (q["flag_owner"] == target) & hit
L187:             q["flag_pos"] = jnp.where(dropped[:, None], q["agent_pos"][target], q["flag_pos"])
L188:             q["flag_owner"] = jnp.where(dropped, -1, q["flag_owner"])
L189:             q["carrying"] = q["carrying"].at[target].set(jnp.where(hit, -1, q["carrying"][target]))
L190:             return q, credit
```

**Natural contest ending**

```python
L195:         all_escaped = jnp.all(n["escaped"][2:])
L196:         caught = jnp.all(n["escaped"][2:] | (n["frozen"][2:] > 0)) & ~all_escaped
L197:         n["t"] = s["t"] + 1
L198:         n["terminated"] = all_escaped | caught
L199:         n["truncated"] = (n["t"] >= self.max_steps) & ~n["terminated"]
L200:         n["done"] = n["terminated"] | n["truncated"]
```

**Absorbing raw state**

```python
L212:         next_state = jax.tree.map(lambda old, new: jnp.where(state["done"], old, new), state, proposed)
L213:         rewards = jnp.where(state["done"], jnp.zeros((4,), dtype=jnp.float32), rewards)
L214:         dones = jnp.full((4,), next_state["done"], dtype=jnp.bool_)
L215:         info = {"done_all": next_state["done"], "terminated": next_state["terminated"],
L216:                 "truncated": next_state["truncated"]}
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
