# Lifeline Grid

Four moving maintainers control a connected, finite-resource power network. Seven node coordinates and seven endpoint pairs are native state. Node 0 is the generator, nodes 1 and 2 are buses, and nodes 3..6 are the four individually valued services. Six default-closed tree links plus an initially open bus tie provide distinct routing choices. Link fixture positions determine where a worker can toggle or repair; the scene shows those control fixtures and actual endpoint indices rather than inventing wire motion.

Every enabled unfinished service requests at most two units per tick until its cumulative 24-unit quota. Conducting links are those closed with positive integrity. Fixed seven-pass distance propagation establishes shortest paths to generator; each service then traces a decreasing-distance parent path, with lowest edge index as deterministic tie break. This is network physics and allocation, not an agent/learner strategy. Served demand is zero if disconnected; actual integer path loads are recorded. No capacity clipping hides excess demand: load is transmitted first, then overload damage is settled.

Trunks and tie have base capacity three, four service links capacity two. Each third consecutive overload tick reduces integrity by one and resets stress; a link at integrity zero is disconnected for subsequent ticks. Hence the flow attribute is last-transition transmitted load, not a promise of present conductivity. Low load resets stress. Repair restores integrity three and stress zero but uses one carried unit and a reward cost of one. Each worker starts with two units, can carry three and may refill from the shared four-unit generator stock.

A worker adjacent to generator may consume one of twelve shared reserve units to add one capacity to all links for that tick. Simultaneous requests consume at most one unit, with rotating actor priority. Any adjacent worker may toggle a service demand. Relieving an overloaded trunk can protect another worker future service at the cost of foregone individually valuable service. Closing the bus tie offers an alternate path if a trunk is isolated, potentially concentrating load on the remaining trunk. Repair, preemptive shedding, boost and rerouting are distinct source-derived alternatives; their payoff tradeoffs depend on others choices.

Each worker private integer service value is sampled 1..3. All physical network and supply state is public, but value is observed only by that worker and absent from scene attributes. Workers may share a cell; all movement stays inside 9 by 7 bounds. Nearest fixture/site ties use array order. Control is spatial: the same action at another position can affect another link or fail after peer changes. Invalid requests become wait.

Declared source invariants: delivered units 0..24, carried supply 0..3, shared reserve 0..12, shared refill stock 0..4, integrity 0..3, stress 0..2, all integer/bool state bounded and fixed shape. Per-step flow sums are integer and at most aggregate demand eight per edge. All arithmetic is integer until float32 boundary conversion. The six-tick blackout is a literal task failure deadline even if an unchosen repair could have helped; it is not a solver impossibility certificate. Success precedes both blackout and horizon on a shared step. Raw terminal calls are absorbing.


Authorship and status: gpt-6-astra / max, original attempt_00, AUTHORED_NOT_EXECUTED, source_split=train, final_test_used=false. This identity is the assigned author self-report; root retains independent dispatch evidence. Source SHA256: 573c40fad10327cf71da9e2994e43ed8972161d4fe4b6a8f5bdc0861c252b0d9. A/O/U/H = 4/141/10/72.

The author has not imported or executed candidate.py, JAX or NumPy; no reset, raw_step, observe, legal_actions, JIT, vmap, rollout, training, rendering, browser, remote execution or additional model/API call occurred. Only supplied trusted stdlib AST/manifest/scene checks and literal metadata checks were run. AST/schema acceptance is not R0, numerical parity, gameplay validation, learnability or transfer. Eager/JIT/vmap parity remains unmeasured, with no tolerance invented.

All state small signed fields use int32; bool flags remain bool. Reset keys are trusted-interface inputs, not duplicated as mutable PRNG state. Clock increments once for a nonterminal call. Natural termination takes priority over horizon; truncated=(t>=H)&~terminated. An already-terminal call preserves every leaf and observation, returns zero reward and all dones=true, retaining the real causes. The source does not autoreset. Native map layout is explicitly public. Literal per-role observation and rendering masks are included in the manifest and source.

The five supplied upstream reference files were byte-read for SHA256 integrity only, not semantically read or copied. No upstream source snippet was used. The assigned community ecosystem requirements refer to JaxMARL multiagent interfaces, Pgx board-game APIs, Craftax game/world mechanics, and PureJaxRL / Craftax PPO algorithm integration as distinct concerns. This original game kernel does not substitute any of those PPO implementations for the frozen learner. Runtime adapter/replay/animation are later integration responsibilities; no generated animation or empirical acceptance is claimed.

Input provenance is exact in author_input_freeze.json: all twenty enumerated files plus final batch authority and scope clarification. Current 166-169 requests/designs control identity; inherited requirement games 154-157 is retained only as the byte-exact product/interaction requirement source. No prior source, result, payoff, selection, hidden/final-test input or parent history was read.

The playthrough is a finite literal hypothesis. Its original paired intervention changes exactly one actor-action component, uses the same seed/prefix and same-length branches, and states the source-derived delay. Later negative, zero, blocked, illegal or early-terminal evidence must remain unchanged. No author-executed solvability or success claim is made.

Interface source lines: reset=12, observe=60, legal_actions=40, raw_step=91.
Literal source evidence:
- candidate.py:65: def power(self, s):
- candidate.py:84: parent = adjacent & live & (distance[other] < distance[node])
- candidate.py:107: toggle = (a[i] == 5) & (d[e] <= 1)
- candidate.py:110: repair = (a[i] == 6) & (d[e] <= 1) & (n["supply"][i] > 0) & ((n["health"][e] < 3) | (n["stress"][e] > 0))
- candidate.py:119: boost = (a[i] == 8) & generator & (n["storage"][0] > 0) & ~n["boost"]
- candidate.py:124: shed = (a[i] == 9) & (sd[site] <= 1)
- candidate.py:129: capacity = jnp.array([3, 3, 2, 2, 2, 2, 3], jnp.int32) + n["boost"].astype(jnp.int32)
- candidate.py:132: n["tripped"] = stress >= 3
- candidate.py:139: failure = (n["blackout"] >= 6) & ~success
- candidate.py:145: n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
