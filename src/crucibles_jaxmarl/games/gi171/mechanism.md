# Shared Firebreak

Status: AUTHORED_NOT_EXECUTED. Initial blind attempt_00; train split. Declared A/O/U/H = (3, 291, 15, 72).

Three controlled crews manage native cellular fire on a 9 by 7 grid. This bounded process is not a physical fire model. Fires start at (4,3) and (1,1). Cells hold three fuel except the reservoir at (4,6), which has none. Assets at (6,3),(6,1),(2,5) start with four health. The shared tank starts at eighteen, gains two every six transitions and is capped at twenty-four. Crews carry at most four water and start with two water and six labor.

Order is action validation, movement, spray/clear, resource settlement, fuel/spread, asset damage and terminal evaluation. Sprays in four directions or on the own cell consume one water and one labor; redundant sprays each pay. Clearing a non-burning own cell costs two labor and removes fuel permanently. Other non-rest actions cost one labor. Rest recovers two up to eight. Crews may co-locate and traverse fire: damage applies to assets rather than crew bodies.

Refills transfer up to two units near the reservoir; gifts transfer exactly one to an adjacent crew. Both settle in public (t+offset) mod 3 order. Recipient movement can invalidate a pre-legally chosen gift; labor is still spent. Stock and capacity are rechecked at settlement, preserving water without overdraft. Periodic tank replenishment is the only new water.

Every third transition, burning cells consume one fuel, then surviving fire ignites a single downwind neighbor with fuel. Newly ignited cells do not recursively spread that tick. Public wind rotates east/south/west/north every six ticks. A spread uses input-state wind; returned wind names the next phase. Every transition, a burning live asset loses one health. Clearing (5,3) interrupts a delayed route from (4,3) to the asset at (6,3), affecting others after the clearing labor has been spent.

Reset independently samples one private priority index per crew from the supplied key. Priority weight is four, other weights one. Each damaged asset imposes the corresponding penalty on every crew; each non-rest action subtracts one reward. Extinction pays twice each crew's surviving-asset weight sum once. Bounds are [-7,12]. Natural end is extinction or all assets lost; fallback horizon is 72.

Own priority is private. Other crew positions/water/labor and all cells are visible only within Manhattan radius three. Wind, tank, assets and health are public. The native tile map has an explicit per-role visibility state rather than public-map permission. Private per-cell ignition/suppression/transfer logs are declared state for independent checking but omitted from unrestricted scene event channels. Visible native cells show actual fuel, fire and clearing.

Prospective choices include preventive common firebreaks, private-priority defense, aid by water transfer, and consuming a refill before a peer arrives. Fire is a native process; no helper policy controls a crew. Source invariants include water [0,4], labor [0,8], tank [0,24], fuel [0,3], health [0,4], in-bounds positions, no burning zero-fuel cells, and no regrowth on cleared cells. No numerical or strategic outcome has been established.


Terminal interface: a nonterminal raw transition advances scalar int32 t exactly once. Natural completion has priority at a coincident horizon. Truncated equals (t >= max_steps) and not terminated; done_all is their disjunction and equals episode_done. All agent dones share that value. A completed input preserves every state leaf, observation, clock and terminal cause, returning zero float32 reward. Legal masks retain action 0; inactive roles retain only action 0. Invalid IDs or masked actions deterministically become action 0. Raw-step never auto-resets.

Representation: fixed-shape int32/bool state, float32 observations and rewards. Small signed counters and IDs use int32. The supplied uint32 reset key is trusted trajectory provenance, not duplicated native state. Random traits, where present, use only that key. Subsequent transitions are deterministic. These are declared interface choices, not actual execution receipts.

Provenance and limits: AUTHORED_NOT_EXECUTED, initial blind attempt_00. The five supplied upstream references were read only as bytes for integrity checks, not semantically read or copied. No upstream implementation or license text is incorporated. Rules, native kernel and literal scripts are original to this authoring turn. Trusted static AST, manifest and scene data validators are the only checks performed. Candidate code was never imported or executed. JAX shape behavior, eager/jit/vmap parity, numerical bounds, causal effects, actual rendering, solvability, optimality, strategic richness, non-isomorphism, learning and meta-RL transfer remain unverified. No reward or success acceptance threshold is introduced.

Independent gpt-6.1-sol/high integration owns the JaxMARL adapter, sealed execution and real replay/animation. No CSD3, remote browser, model call, training, heldout evidence, prior source/result or unrestricted search was used. The final input freeze records all 22 actual-read inputs with paths, byte counts, hashes and use.

Playthroughs: every seed, prefix and suffix is an original literal prospective hypothesis. Source reasoning suggests legality, but no execution verified legality or outcomes. There was no seed search or numerical feedback. Each paired branch differs in exactly one scalar action; later actions are identical. Delay and intended differences explain a testable source dependency, not a measured effect or successful play.

## Literal source evidence

- candidate.py line 14: visible = distance <= 3
- candidate.py line 43: 'priority': jax.random.randint(key, (3,), 0, 3, dtype=jnp.int32),
- candidate.py line 80: clear = (s['fuel'][own] > 0) & ~s['fire'][own] & (s['labor'] >= 2)
- candidate.py line 134: water = water.at[i].add(-unit).at[target].add(unit)
- candidate.py line 137: burn_tick = (t % 3) == 0
- candidate.py line 145: ignited = burn_tick & inside & survivors[index] & (fuel > 0) & ~survivors
- candidate.py line 148: damage = fire[zi] & (s['zone_health'] > 0)
- candidate.py line 152: weights = 1 + 3 * jax.nn.one_hot(s['priority'], 3, dtype=jnp.int32)
- candidate.py line 165: n['wind'] = (t // 6) % 4
- candidate.py line 178: n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((3,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)

## Method locations

- refresh: candidate.py lines 12-27.
- reset: candidate.py lines 29-69.
- legal_actions: candidate.py lines 71-86.
- observe: candidate.py lines 88-101.
- advance: candidate.py lines 103-175.
- raw_step: candidate.py lines 177-180.
