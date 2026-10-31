# Echo Pursuit

Three controlled agents are two hunters and one evader. The map has a vertical movement barrier at x=4 except openings at y=0,4,8; sight and scan use Manhattan radius and do not model wall occlusion. This explicit rule avoids implying a richer line-of-sight sensor than the implementation. All players can occupy the same cell; capture requires an explicit hunter action, never incidental contact.

Scans cost two energy, produce the nearest echo within five Manhattan cells, and broadcast the scanner current position for two ticks. The echo is only coordinates and a three-tick lifetime. It is not a typed evader measurement: a live decoy at an equal or shorter distance is returned instead. Decoy placement uses a finite budget of two and leaves a physical four-tick beacon. An evader may continue moving after placing it. A scan with no contact clears that hunter contact. Local direct sight, separately, can reveal agents and physical decoys within radius two.

Relay costs one energy and copies the sender previous contact coordinates with its remaining lifetime reduced by one. No relay refreshes a contact and there is no same-tick information chain: relays read the input state. Receiver scans have precedence over received relays on that transition. A last-tick contact cannot be prolonged by relaying. The teammate can intercept from a different approach while the scanner accepts the disclosure cost. Evader movement, false echoes or a one-energy two-tick cloak can alter that plan. These are source-reasoned alternatives, with no behavioral or learning measurements.

Only own energy, budget and cloak are directly observed. The scene binds no hidden energy or cloak attribute. Native agent visibility is local sight, self visibility or an active scanner beacon; a cloaked evader requires distance at most one. Native contacts are visible only to their respective receiver, including relay recipients. Locally visible decoys expose the same TTL as their observation fields. All coordinates shown in a role view are actually available to that role. Full-state/debug presentation, if later supplied, must remain separately labelled.

Energy bounds are 0..8, decoy budget 0..2, cloak/pulse 0..2, decoy TTL 0..4 and contact TTL 0..3. Movement cannot enter walls or leave [0,8]^2. Observation hidden-coordinate sentinel is -1; other numeric fields are raw integers cast once to float32. No stochastic state is used in this original kernel; key and script seed are retained by the interface/provenance, with no substitute random distribution. Invalid requests deterministically become wait and therefore recover energy under the literal wait rule.

Capture requires hunter action 7 and post-movement Manhattan distance at most one. Evader action 7 requires its input gate-open mask and location; capture has natural priority if both events coincide. Time limit alone grants neither team a terminal victory reward. All raw terminal calls preserve cause/state/clock and return zero rewards.


Authorship and status: gpt-6-astra / max, original attempt_00, AUTHORED_NOT_EXECUTED, source_split=train, final_test_used=false. This identity is the assigned author self-report; root retains independent dispatch evidence. Source SHA256: 4063107ebba39493e0237a3e79e2b174741ff016dbe2e19c5afac143c66e10f4. A/O/U/H = 3/40/8/48.

The author has not imported or executed candidate.py, JAX or NumPy; no reset, raw_step, observe, legal_actions, JIT, vmap, rollout, training, rendering, browser, remote execution or additional model/API call occurred. Only supplied trusted stdlib AST/manifest/scene checks and literal metadata checks were run. AST/schema acceptance is not R0, numerical parity, gameplay validation, learnability or transfer. Eager/JIT/vmap parity remains unmeasured, with no tolerance invented.

All state small signed fields use int32; bool flags remain bool. Reset keys are trusted-interface inputs, not duplicated as mutable PRNG state. Clock increments once for a nonterminal call. Natural termination takes priority over horizon; truncated=(t>=H)&~terminated. An already-terminal call preserves every leaf and observation, returns zero reward and all dones=true, retaining the real causes. The source does not autoreset. Native map layout is explicitly public. Literal per-role observation and rendering masks are included in the manifest and source.

The five supplied upstream reference files were byte-read for SHA256 integrity only, not semantically read or copied. No upstream source snippet was used. The assigned community ecosystem requirements refer to JaxMARL multiagent interfaces, Pgx board-game APIs, Craftax game/world mechanics, and PureJaxRL / Craftax PPO algorithm integration as distinct concerns. This original game kernel does not substitute any of those PPO implementations for the frozen learner. Runtime adapter/replay/animation are later integration responsibilities; no generated animation or empirical acceptance is claimed.

Input provenance is exact in author_input_freeze.json: all twenty enumerated files plus final batch authority and scope clarification. Current 166-169 requests/designs control identity; inherited requirement games 154-157 is retained only as the byte-exact product/interaction requirement source. No prior source, result, payoff, selection, hidden/final-test input or parent history was read.

The playthrough is a finite literal hypothesis. Its original paired intervention changes exactly one actor-action component, uses the same seed/prefix and same-length branches, and states the source-derived delay. Later negative, zero, blocked, illegal or early-terminal evidence must remain unchanged. No author-executed solvability or success claim is made.

Interface source lines: reset=25, observe=68, legal_actions=49, raw_step=80.
Literal source evidence:
- candidate.py:12: def refresh(self, s):
- candidate.py:49: def legal_actions(self, s):
- candidate.py:94: place = a[2] == 5
- candidate.py:102: relay = (a[i] == 6) & (s["signal_ttl"][i] > 1)
- candidate.py:110: use_decoy = (n["decoy_ttl"][0] > 0) & (false_d <= real_d)
- candidate.py:116: n["pulse"] = n["pulse"].at[i].set(jnp.where(n["scan_event"][i], 2, n["pulse"][i]))
- candidate.py:119: n["capture"] = jnp.any((a[:2] == 7) & contact)
- candidate.py:120: n["escape"] = (a[2] == 7) & (n["t"] >= 8) & jnp.all(n["pos"][2] == n["gate_pos"][0]) & ~n["capture"]
- candidate.py:127: n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
