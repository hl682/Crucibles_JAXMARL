# Relay Arena

Status: AUTHORED_NOT_EXECUTED. Initial blind attempt_00; train split. Declared A/O/U/H = (4, 66, 11, 80).

Four controlled players form pairs {0,1} and {2,3} on an 11 by 7 grid. Team 0 attacks x=10 and team 1 x=0, with goal lanes y=2,3,4. Initial positions are (3,3),(3,4),(7,3),(7,2); player 0 carries the token. Reset is deterministic and ignores its key. No helper, receiver or opponent policy is embedded.

Free walking costs one energy and carrier walking two. Rest recovers two up to eight. Catch/tackle costs one; pass and screen cost two. Special actions require old cooldown zero, set cooldown two, and become available after two subsequent transitions. Attempts pay even when their mechanics fail. A catch stance is legal for a noncarrier without guaranteed success, so support is a policy decision.

Order is validation; screen declarations and desired movement; blocking; energy/cooldown; tackles; launches; loose flight; catches; goals; optional kickoff; clock and terminal evaluation. Moving into another player's old cell, swaps, mutually contested destinations, and destinations within one cell of an enemy screen are rejected. Players remain distinct. Blocked moves pay energy. A screen protects a teammate carrier within one cell against tackles in this transition. Thus a teammate action alters carrier safety and enemy feasible movement.

Tackles and catches require explicit action 5 within Manhattan distance one. Distance takes priority; ties use (role - t mod 4) mod 4. Tackles precede passes and can prevent launch. The carrier chooses a cardinal direction, not an oracle target. A launched ball advances immediately and can travel four steps total. A catch places the token on the receiver's native position and clears flight. An opposing catch relative to the last passer records interception. A loose stopped token still requires explicit pickup. Screens affect movement and tackles, not loose-ball catch priority.

Goals are evaluated after catches on actual token position. Entering the wrong goal is an own goal because boundary direction determines scoring. Each scoring teammate gets ten, opponents minus ten. First to two terminates naturally. Before the horizon, a nonwinning goal places the original formation and grants kickoff possession to the conceding side's player 2 or 0. Score, clock, energy and cooldown persist. Winning or horizon goals retain actual goal state. Raw-step never invokes reset. Horizon is 80 with natural-priority semantics.

All information relevant to gameplay is public in 66 columns. Native player and ball positions, possession, energy, cooldown, screen, flight and pass/interception/goal flags bind the scene. Prospective alternatives include passing to conserve carrier energy, dribbling when a receiver is marked, screening a vulnerable carrier and repositioning to intercept. None is asserted optimal or learned. Invariants stated from source: distinct in-bounds players, ball in bounds, at most one carrier, ball at carrier location when held, energy [0,8], cooldown [0,2], flight [0,4], and scores [0,2].


Terminal interface: a nonterminal raw transition advances scalar int32 t exactly once. Natural completion has priority at a coincident horizon. Truncated equals (t >= max_steps) and not terminated; done_all is their disjunction and equals episode_done. All agent dones share that value. A completed input preserves every state leaf, observation, clock and terminal cause, returning zero float32 reward. Legal masks retain action 0; inactive roles retain only action 0. Invalid IDs or masked actions deterministically become action 0. Raw-step never auto-resets.

Representation: fixed-shape int32/bool state, float32 observations and rewards. Small signed counters and IDs use int32. The supplied uint32 reset key is trusted trajectory provenance, not duplicated native state. Random traits, where present, use only that key. Subsequent transitions are deterministic. These are declared interface choices, not actual execution receipts.

Provenance and limits: AUTHORED_NOT_EXECUTED, initial blind attempt_00. The five supplied upstream references were read only as bytes for integrity checks, not semantically read or copied. No upstream implementation or license text is incorporated. Rules, native kernel and literal scripts are original to this authoring turn. Trusted static AST, manifest and scene data validators are the only checks performed. Candidate code was never imported or executed. JAX shape behavior, eager/jit/vmap parity, numerical bounds, causal effects, actual rendering, solvability, optimality, strategic richness, non-isomorphism, learning and meta-RL transfer remain unverified. No reward or success acceptance threshold is introduced.

Independent gpt-6.1-sol/high integration owns the JaxMARL adapter, sealed execution and real replay/animation. No CSD3, remote browser, model call, training, heldout evidence, prior source/result or unrestricted search was used. The final input freeze records all 22 actual-read inputs with paths, byte counts, hashes and use.

Playthroughs: every seed, prefix and suffix is an original literal prospective hypothesis. Source reasoning suggests legality, but no execution verified legality or outcomes. There was no seed search or numerical feedback. Each paired branch differs in exactly one scalar action; later actions are identical. Delay and intended differences explain a testable source dependency, not a measured effect or successful play.

## Literal source evidence

- candidate.py line 78: occupied = jnp.any(jnp.all(desired[:, None, :] == s['agent_pos'][None, :, :], axis=-1) & different, axis=1)
- candidate.py line 80: enemy_screen = jnp.any((jnp.sum(jnp.abs(desired[:, None, :] - s['agent_pos'][None, :, :]), axis=-1) <= 1) & screen[None, :] & (s['team'][:, None] != s['team'][None, :]), axis=1)
- candidate.py line 83: cost = walking.astype(jnp.int32) * (1 + holding.astype(jnp.int32)) + (a == 5).astype(jnp.int32) + 2 * (a >= 6).astype(jnp.int32)
- candidate.py line 90: protected = jnp.any(screen & (s['team'] == s['team'][holder]) & (jnp.sum(jnp.abs(p - p[holder]), axis=1) <= 1))
- candidate.py line 91: tackle = (old_carrier >= 0) & (a == 5) & (s['team'] != s['team'][holder]) & (distance <= 1) & ~protected
- candidate.py line 93: priority = (jnp.arange(4) - (s['t'] % 4)) % 4
- candidate.py line 99: launch = (old_carrier >= 0) & ~taken & (a[holder] >= 6) & (a[holder] <= 9)
- candidate.py line 111: receiver = (carrier < 0) & (a == 5) & (catch_distance <= 1)
- candidate.py line 122: goal = jnp.array([(ball[0] == 10) & lane, (ball[0] == 0) & lane], jnp.bool_)
- candidate.py line 125: kickoff = jnp.any(goal) & ~terminal & (s['t'] + 1 < self.max_steps)
- candidate.py line 147: reward = (10 * (own - opponent)).astype(jnp.float32)
- candidate.py line 151: n, reward = jax.lax.cond(state['episode_done'], lambda _: (state, jnp.zeros((4,), jnp.float32)), lambda _: self.advance(state, actions), operand=None)

## Method locations

- reset: candidate.py lines 12-42.
- legal_actions: candidate.py lines 44-55.
- observe: candidate.py lines 57-66.
- advance: candidate.py lines 68-148.
- raw_step: candidate.py lines 150-153.
