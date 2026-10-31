# Counterweight Crew

Status: AUTHORED_NOT_EXECUTED. Initial blind attempt_00; train split. Declared A/O/U/H = (3, 119, 7, 80).

A three-worker salvage game on a 9 by 7 integer grid. Bulkheads at x=4 leave passages at y=1,3,5. The central cargo needs net force two and two light cargoes need one. Position governs walking, cargo clearance, tether motion and depot delivery.

Actions are rest, north/east/south/west, attach and release. Attach chooses minimum distance among active cargo within Manhattan distance one, then smallest cargo ID. A tether persists through rest. A free worker walks; a tethered worker contributes one directional force. Directions cost one stamina and one reward even if a collective move fails. Rest recovers two stamina up to ten. All tethers translate with moving cargo, including a worker who rested or opposed the resulting force.

Workers may co-locate. Walking into active cargo or a bulkhead is forbidden. Cargo cannot enter another active cargo or a worker not tethered to it. Its tethered workers must also fit on floor and avoid other active cargo. Forces are simultaneous, with cargo occupancy resolved in ID order 0,1,2. The axis of greater force magnitude determines direction, with x on ties. Magnitude must reach cargo weight. Thus commitment, opposition, depletion, resting and blocking alter peer opportunities over time.

Depots are west, northeast and southeast. Public worker-by-depot values are [[6,1,2],[1,6,2],[2,1,6]]. Delivery pays each worker its destination value times cargo weight, removes cargo and releases its tethers. Helping heavy transport, alternating effort and rest, disagreeing about destination, releasing to retrieve light cargo and blocking routes are prospective alternatives. No controller coordinates any worker.

All information is public in 119 columns. Scene attributes expose actual stamina, attachment, force and displacement. Invariants stated from source: integer positions in bounds and on floor; active cargo positions distinct; attachment is -1 or an active cargo index; attached distance at most one; stamina [0,10]; force [-3,3]; cargo weight [2,1,1]; each cargo delivered at most once. Reward is bounded [-1,24] because total cargo weight is four and depot value at most six. These are source-level deductions, not numerical results.


Terminal interface: a nonterminal raw transition advances scalar int32 t exactly once. Natural completion has priority at a coincident horizon. Truncated equals (t >= max_steps) and not terminated; done_all is their disjunction and equals episode_done. All agent dones share that value. A completed input preserves every state leaf, observation, clock and terminal cause, returning zero float32 reward. Legal masks retain action 0; inactive roles retain only action 0. Invalid IDs or masked actions deterministically become action 0. Raw-step never auto-resets.

Representation: fixed-shape int32/bool state, float32 observations and rewards. Small signed counters and IDs use int32. The supplied uint32 reset key is trusted trajectory provenance, not duplicated native state. Random traits, where present, use only that key. Subsequent transitions are deterministic. These are declared interface choices, not actual execution receipts.

Provenance and limits: AUTHORED_NOT_EXECUTED, initial blind attempt_00. The five supplied upstream references were read only as bytes for integrity checks, not semantically read or copied. No upstream implementation or license text is incorporated. Rules, native kernel and literal scripts are original to this authoring turn. Trusted static AST, manifest and scene data validators are the only checks performed. Candidate code was never imported or executed. JAX shape behavior, eager/jit/vmap parity, numerical bounds, causal effects, actual rendering, solvability, optimality, strategic richness, non-isomorphism, learning and meta-RL transfer remain unverified. No reward or success acceptance threshold is introduced.

Independent gpt-6.1-sol/high integration owns the JaxMARL adapter, sealed execution and real replay/animation. No CSD3, remote browser, model call, training, heldout evidence, prior source/result or unrestricted search was used. The final input freeze records all 22 actual-read inputs with paths, byte counts, hashes and use.

Playthroughs: every seed, prefix and suffix is an original literal prospective hypothesis. Source reasoning suggests legality, but no execution verified legality or outcomes. There was no seed search or numerical feedback. Each paired branch differs in exactly one scalar action; later actions are identical. Delay and intended differences explain a testable source dependency, not a measured effect or successful play.

## Literal source evidence

- candidate.py line 14: tiles = tiles.at[jnp.array([0, 2, 4, 6]), 4].set(1)
- candidate.py line 75: attachment = jnp.where(a == 5, nearest, jnp.where(a == 6, -1, s['attachment']))
- candidate.py line 79: force = jnp.sum(attached[:, :, None].astype(jnp.int32) * dirs[a][:, None, :], axis=0, dtype=jnp.int32)
- candidate.py line 87: enough = jnp.maximum(jnp.abs(fx), jnp.abs(fy)) >= s['cargo_weight'][k]
- candidate.py line 94: worker_block = jnp.any(jnp.all(pos == target[None, :], axis=1) & ~tether)
- candidate.py line 102: payoff = jnp.sum(s['values'][:, destination] * (delivered.astype(jnp.int32) * s['cargo_weight'])[None, :], axis=1)
- candidate.py line 117: n['terminated'] = ~jnp.any(active)
- candidate.py line 124: state['episode_done'],

## Method locations

- reset: candidate.py lines 12-40.
- floor_ok: candidate.py lines 42-45.
- legal_actions: candidate.py lines 47-56.
- observe: candidate.py lines 58-66.
- advance: candidate.py lines 68-120.
- raw_step: candidate.py lines 122-131.
