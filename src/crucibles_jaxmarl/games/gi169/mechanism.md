# Mutual Cover

Three cargo carriers start at (1,1), (1,3), (1,5), with own destinations at x=7 on the same rows. The public periodic hazard cells are at x=4 on those rows; routes through other rows are possible. Hazard j is active when (t+j) mod four is zero or one. There is no random or private hazard forecast; all players can reason from native public phase. Movement and object interaction are real spatial transitions. Co-location is permitted for social proximity, repairs and escorts.

Each carrier starts with health three and six coins. A two-coin self-protection purchase lasts three ticks including the purchase tick and removes one damage per hazard exposure. A one-coin escort targets the nearest other active carrier within one Manhattan cell and removes one damage in that transition; multiple escorts can prevent all two points. A one-coin repair selects an injured adjacent peer before self, deterministic distance/index ties, and restores one health before current hazard damage. Peers can respond by moving out of range; effective post-movement proximity is rechecked.

An offer identifies the nearest active peer within two cells and persists three ticks. Acceptance reads the input-state offer, rechecks range/resources and locks two spendable coins from both members for six ticks. Each carrier can hold one symmetric pact. Simultaneous conflicts use rotating actor priority. Collateral remains native escrow and cannot purchase protection, repairs or another pact. The accepted pair therefore exchanges future cover against immediate lost spending opportunities.

The first tick in which either member actually takes positive native hazard damage settles the pact. A sole damaged member receives all four escrowed coins; if both take damage, each receives two. Damage-triggered allocation precedes non-damage delivery/expiry refund. Delivery, destruction or expiry also settles the pact; absent differential damage, each gets its two coins back. All settlement clears partner/escrow/timers together. Shielding or escort that prevents damage can delay a payout. Cover is a financial transfer and does not secretly restore health. Real repair and escort remain separate decisions.

Reward is the change in total cash plus escrow, plus ten for own delivery, minus six for new destruction and two per damage point. Thus merely locking collateral is not a fabricated wealth loss, but its liquidity has a genuine action-mask opportunity cost. Transfers redistribute wealth, expenditures consume it, and safer routes trade time against hazard exposure. The source supports plausible refusal, mutual cover, self-insurance, escort or repair choices without prescribing an oracle strategy or claiming measured trust/cooperation.

Declared invariants: health 0..3, cash nonnegative and at most initial system wealth eighteen, escrow zero or two, symmetric one-partner pacts, pact TTL 0..6, offer TTL 0..3, shield 0..3, hazard phase 0..3, fixed bounded positions. System cash plus escrow never increases; purchases consume it. Damage can exceed remaining health but is bounded at two and health is clamped at zero. All observations are raw integer/bool values cast once to float32; every scene attribute is public policy information. Finished cargo becomes inactive with only no-op; its final outcome persists. Invalid requests become no-op. Joint completion can mix delivery and destruction, and is not labelled all-success. At time truncation an unexpired pact remains recorded and absorbing; it is not falsely called settled.


Authorship and status: gpt-6-astra / max, original attempt_00, AUTHORED_NOT_EXECUTED, source_split=train, final_test_used=false. This identity is the assigned author self-report; root retains independent dispatch evidence. Source SHA256: 07ba52771b39560e4c187a9d82f4f79f3ba6a6a8dcfd55a0019008e14d08f721. A/O/U/H = 3/79/10/64.

The author has not imported or executed candidate.py, JAX or NumPy; no reset, raw_step, observe, legal_actions, JIT, vmap, rollout, training, rendering, browser, remote execution or additional model/API call occurred. Only supplied trusted stdlib AST/manifest/scene checks and literal metadata checks were run. AST/schema acceptance is not R0, numerical parity, gameplay validation, learnability or transfer. Eager/JIT/vmap parity remains unmeasured, with no tolerance invented.

All state small signed fields use int32; bool flags remain bool. Reset keys are trusted-interface inputs, not duplicated as mutable PRNG state. Clock increments once for a nonterminal call. Natural termination takes priority over horizon; truncated=(t>=H)&~terminated. An already-terminal call preserves every leaf and observation, returns zero reward and all dones=true, retaining the real causes. The source does not autoreset. Native map layout is explicitly public. Literal per-role observation and rendering masks are included in the manifest and source.

The five supplied upstream reference files were byte-read for SHA256 integrity only, not semantically read or copied. No upstream source snippet was used. The assigned community ecosystem requirements refer to JaxMARL multiagent interfaces, Pgx board-game APIs, Craftax game/world mechanics, and PureJaxRL / Craftax PPO algorithm integration as distinct concerns. This original game kernel does not substitute any of those PPO implementations for the frozen learner. Runtime adapter/replay/animation are later integration responsibilities; no generated animation or empirical acceptance is claimed.

Input provenance is exact in author_input_freeze.json: all twenty enumerated files plus final batch authority and scope clarification. Current 166-169 requests/designs control identity; inherited requirement games 154-157 is retained only as the byte-exact product/interaction requirement source. No prior source, result, payoff, selection, hidden/final-test input or parent history was read.

The playthrough is a finite literal hypothesis. Its original paired intervention changes exactly one actor-action component, uses the same seed/prefix and same-length branches, and states the source-derived delay. Later negative, zero, blocked, illegal or early-terminal evidence must remain unchanged. No author-executed solvability or success claim is made.

Interface source lines: reset=24, observe=84, legal_actions=65, raw_step=88.
Literal source evidence:
- candidate.py:57: def repair_target(self, s, i):
- candidate.py:109: accept = (a[i] == 8) & (n["partner"][i] < 0) & (n["cash"][i] >= 2) & jnp.any(incoming)
- candidate.py:121: offer = (a[i] == 7) & (d <= 2) & (n["partner"][i] < 0) & (n["partner"][q] < 0) & (n["cash"][i] >= 2)
- candidate.py:125: insure = (a[i] == 5) & (n["cash"][i] >= 2)
- candidate.py:133: escort = (a[i] == 9) & (d <= 1) & (n["cash"][i] >= 1)
- candidate.py:139: danger = jnp.any(on & n["hazard_active"][None, :], axis=1) & n["agent_active"]
- candidate.py:141: n["damage"] = jnp.where(danger, jnp.maximum(2 - (n["shield"] > 0).astype(jnp.int32) - escort_count, 0), 0)
- candidate.py:154: settle = linked & (hi | hj | finish | (n["pact_ttl"][i] == 0))
- candidate.py:162: reward = n["cash"] + n["escrow"] - s["cash"] - s["escrow"] + 10 * n["delivery_event"].astype(jnp.int32) - 6 * new_failure.astype(jnp.int32) - 2 * n["damage"]
- candidate.py:163: n["terminated"] = jnp.all(n["delivered"] | n["failed"])
- candidate.py:167: n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
