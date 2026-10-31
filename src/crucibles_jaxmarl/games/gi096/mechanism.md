# Flank-and-Hold Meadow

A/O/U/H = 3/133/8/64. Three shepherds guide four actual herd animals through a held gate into two safe pens. Herd reactions sum the effects of multiple nearby shepherds, making one role's best pressure or calming action depend on the others.

## Rules and causally coupled decisions

The 9 by 7 meadow has a divider at x=5, a gate at [5,3], safe pens at [7,2] and [7,4], and ditches at [4,1] and [4,5]. Two shepherds begin on the west side and a third begins beside the east side of the gate. Roles have the same commands. Shepherd movement cannot enter hedges, ditches, a closed gate, another shepherd's pre-step cell, an active animal's cell or a competing destination.

A shepherd's ordinary proximity produces unit repulsion within Manhattan distance 2. A whistle spends one of at most four focus units and produces weight-two repulsion within distance 3. An exhausted whistle command emits only ordinary pressure. Calm restores one focus, emits no repulsion and reduces panic of animals within distance 2. An adjacent gate-hold command is also quiet; the helper is not forced to repel the animals away from the opening while holding it.

Per-animal pressure weights are summed. A total weight at least 3 adds one panic. Nearby calmers subtract panic, and no pressure subtracts one more. Panic is clipped to [0,4]. Animal motion uses the vector sum of signed displacements from every pressuring shepherd. In the absence of pressure, low-panic animals receive a cohesion vector toward nearby active peers. On the safe side at x at least 6, low-panic animals follow a nearest marked pen using the actual pen coordinate.

Each animal takes one physical microstep, or two when panic is at least 3. The larger force component determines direction, with horizontal tie-breaking; an obstructed chosen direction does not receive an invented pathfinder correction. Animal moves resolve in rotating animal order each tick and cannot overlap other active animals or current shepherd cells. High panic permits entry into a ditch, which permanently loses the animal; low-panic animals will not enter a ditch. Reaching a pen saves it permanently. This exposes the tradeoff between rapid pressure and controlled placement while retaining possible trapping or failed coordination as honest outcomes.

Holding the gate from distance at most 1 sets a timer to 3. Otherwise the timer falls by one. The gate remains open if a shepherd or active animal occupied its tile at the start of the step, preventing closure onto that occupant. Gate state is computed before movement and is retained as the actual passability used that transition. It can consequently stay open one clearance tick after an occupant leaves.

Commands are 0 wait with ordinary pressure; 1/2/3/4 movement; 5 whistle; 6 calm; 7 adjacent gate hold. Each saved animal gives all shepherds +1; each lost animal gives -1. Natural completion is all four resolved, regardless of success or failure; the horizon is 64.

Two plausible strategies are calm flankers that aim a compact herd through a gate helper, or stronger short whistle bursts followed by recovery and repositioning. A helper may hold the opening while partners create force, or a flanker may temporarily take over gate duty. Simultaneous whistles can create two-cell panic motion and negate a calmer's desired timing. Standing at the wrong location can block the path or reverse the summed force. These are explicit rule consequences and strategy hypotheses, not evidence of controllability or learned teamwork.

## Information, visible state and declared bounds

Every animal's position, active status, panic, saved/lost status and last-step outcomes are state. Shepherd focus and effective signals are state. Gate position, timer and passability are the exact transition inputs. Every active entity is public to every role; the fixed terrain is public; the complete flat public observation plus own-role one-hot has 133 columns. There is no hidden herd controller or private shepherd view.

Static capacities are three shepherds, four animals, one gate, two pens and two ditches. Active shepherd and animal positions are separately unique, and their transitions prevent cross-group overlap. Focus and panic stay in [0,4], gate timer in [0,3], cumulative saved/lost in [0,4], and active plus saved plus lost remains four animals. Saved and lost status cannot be reversed. Shared rewards are bounded by [-4,4] from the finite herd, not by a desired empirical effect.

## Literal source evidence

- Original world and capacities: `candidate.py:12`, literal `def reset(self, key):`.
- Exact public observation packing: `candidate.py:42`, literal `def observe(self, s):`.
- Shepherd-herd physical blocking: `candidate.py:61`, literal `herd = jnp.any(jnp.all(q[:,None,:] == s["animal_pos"][None,:,:],axis=2) & s["animal_active"][None,:],axis=1)`.
- Real gate hold: `candidate.py:70`, literal `hold = jnp.any((a == 7) & at_gate)`.
- Quiet gate helper: `candidate.py:78`, literal `quiet = calm | ((a == 7) & at_gate)`.
- Summed pressure and bounded panic: `candidate.py:87`, literal `panic_delta = (pressure >= 3).astype(jnp.int32) - soothing - (pressure == 0).astype(jnp.int32)`.
- Animal response to every shepherd: `candidate.py:99`, literal `repulse = jnp.sum(jnp.sign(d)*w[:,None],axis=0,dtype=jnp.int32)`.
- Cohesion without external pressure: `candidate.py:102`, literal `force = repulse + jnp.where((jnp.sum(w) == 0) & (panic[i] < 3),cohesion,jnp.zeros(2,dtype=jnp.int32))`.
- Safe-side pen direction: `candidate.py:110`, literal `force = jnp.where((pos[i,0] >= 6) & (panic[i] < 3),pen_route,force)`.
- Actual moving and loss checks: `candidate.py:122`, literal `lost_i = active[i] & (s["map"][pos[i,1],pos[i,0]] == 3) & (panic[i] >= 3)`.
- Natural end from active herd: `candidate.py:137`, literal `s["terminated"] = ~jnp.any(active)`.
- Absorbing terminal branch: `candidate.py:149`, literal `def absorb(self, key, state, actions):`.

## Witnesses and possible future variation

The short script alternates whistles, calm and gate holds to inspect real focus, pressure, gate state and movement. One pair changes only the gate helper's command; another replaces a flanker whistle with calm. Neither claims all animals are saved, and no safety ordering is presumed.

Future independently assigned animal counts, gate positions, force rules, panic thresholds or pen locations could change useful formations and timing. Current capacity and rules are fixed; no variant, learned transfer or safety certificate has been measured.

## Provenance, interface and status

Source SHA-256: dc0778d0198f820eb1619db6f0f6b666b6bca60afb7655ac7f1ba5ed4e83634f.

This is original author-written game code and layout for this assigned train-stream proposal, source attempt 00, generation window 0. No previous candidate, execution result, replay, ranking, training output or final-test material was read. No upstream game code, benchmark layout, texture, heldout implementation or scoring code was copied. Original authorship does not prove structural non-isomorphism.

The supplied official JaxMARL README informed the distinction between the native flat kernel and an agent-keyed public integration API. The supplied Pgx README informed explicit state, legal-mask and terminal API concepts. The supplied Craftax README informed JAX game-state and reset/render separation concepts. The supplied PureJaxRL README was read as an algorithm/integration reference, not a game library; no PPO implementation was copied and the frozen scientific learner is unchanged. The supplied Craftax_Baselines/ppo_rnn.py was not read or used. Exact copied-reference hashes and official source URLs are in generation_record.json.

Every declared nonterminal command is legal. A physically blocked movement or unavailable object operation is a deterministic failed attempt, not an out-of-range action. Out-of-range integers map to command 0. No role is individually removed before joint termination. Each role retains command 0; after joint termination only 0 is legal. The absorbing raw call ignores the provided actions, returns the same complete state and observation, leaves t unchanged, returns zero float32 rewards and all-true dones, and preserves actual terminal causes. Otherwise t advances exactly once. Natural termination has priority over the horizon, so terminated and truncated cannot both be true. No reset occurs inside raw_step.

All observation columns are covered exactly once in manifest.json, in the literal concatenate order used by observe. Raw integer/bool state is cast to float32 without normalization. Visibility is fully public: active entity visibility is broadcast to all roles; map.public is true. Fixed or derived active/visibility masks do not add hidden policy information. Resolved objects retain their last public position and status in flat state, while the role renderer suppresses their inactive glyphs. A separate full-state debug view must remain labelled as such. The source uses only int32 and bool state, representing small categorical design values losslessly as int32. The reset key remains external provenance and is intentionally unused by these deterministic initial rules.

The native kernel is not a public adapter implementation. Agent-keyed dictionaries, __all__, space objects and replay/UI integration belong to the coordinator's separate presentation layer. The author has not exercised that layer. The six author files satisfy a source-delivery scope; they do not by themselves establish community product acceptance.

No candidate import, reset, raw_step, JAX call, validator, training run, replay, browser playback or numerical parity test was executed by this author. Pure text and JSON metadata review and byte hashing are the only checks performed. playthrough.json contains literal finite action vectors, seeds and explicitly unexecuted paired intervention hypotheses. A later sealed replay must retain mismatches, zero effects, failures and early terminals as actual evidence. No script is a successful trajectory receipt, solvability proof, learned policy, optimal strategy, meta-learning result or generalization result. Model tokens and API charges are unknown rather than invented.

Reference URLs read as supplied snapshots:
- https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md
- https://github.com/sotetsuk/pgx/blob/main/README.md
- https://github.com/luchris429/purejaxrl/blob/main/README.md
- https://github.com/MichaelTMatthews/Craftax/blob/main/README.md
