# Switchyard Claims

The three trains control their own movement, reservation, switching, release, yielding and delivery. There is no dispatcher policy. Tracks form three horizontal routes joined at x=1,4,7, with actual switch blocks at x=4. Agent starts are (3,3), (4,4), (7,5); the shared depot is (7,3). The map is public.

A block can have one owner. Beginning-of-step expiry releases claims; explicit release or yield follows; then claims and toggles are processed in priority (t+k) mod 3. A claim lasts four recorded ticks including its grant state. Entry requires the selected axis and no peer reservation. Legal masks use the input state, including old occupancy, so an old occupied cell cannot be entered even if its occupant moves away. Updated occupancy also blocks entry and resolves simultaneous targets. A peer can change a switch or claim after masking; that can block an otherwise legal move. No swaps or collisions are silently permitted.

Yield is a credible two-decision restriction: subsequent movement masks remain false until its timer has elapsed. Release without commitment preserves the option to move later. Nearest adjacent switch ties use lowest block index. Delivery is explicit at the shared depot; its cooldown=2 makes delivery unavailable for the next two input states. Agents with completed cargo become inactive and may only wait. All three delivered is natural completion, not an inferred optimality or solvability assertion.

Different private urgencies change delay costs. A train can yield for a faster peer, reserve a junction before approaching, change the axis, or take a longer track route. These are source-derived strategic alternatives, without claims that a learner discovers them. Public reservations and binding commitments let another train react over several ticks; this is not three independent routes.

All output observations are raw bounded integers cast once to float32. Urgency is private and absent from scene attributes. Train visibility excludes delivered entities while their final position and status remain public in observations. Events describe the last transition; absorption retains that last terminal record and does not create a new event.

Declared source invariants: active train positions are unique traversable rail cells; block owner is -1..2; TTL is 0..4; axis is 0/1; yield and depot timers are 0..2; all resource/timer arithmetic is int32. The no-op is legal in every state. Invalid/out-of-range or masked requests become no-op deterministically. Claimed and blocked events describe actual effective transitions, not the original unvalidated request.


Authorship and status: gpt-6-astra / max, original attempt_00, AUTHORED_NOT_EXECUTED, source_split=train, final_test_used=false. This identity is the assigned author self-report; root retains independent dispatch evidence. Source SHA256: 0cc9d5510948860eb0d3a62219138aeb88583506f997f50759e7df1f2a7823fb. A/O/U/H = 3/51/10/80.

The author has not imported or executed candidate.py, JAX or NumPy; no reset, raw_step, observe, legal_actions, JIT, vmap, rollout, training, rendering, browser, remote execution or additional model/API call occurred. Only supplied trusted stdlib AST/manifest/scene checks and literal metadata checks were run. AST/schema acceptance is not R0, numerical parity, gameplay validation, learnability or transfer. Eager/JIT/vmap parity remains unmeasured, with no tolerance invented.

All state small signed fields use int32; bool flags remain bool. Reset keys are trusted-interface inputs, not duplicated as mutable PRNG state. Clock increments once for a nonterminal call. Natural termination takes priority over horizon; truncated=(t>=H)&~terminated. An already-terminal call preserves every leaf and observation, returns zero reward and all dones=true, retaining the real causes. The source does not autoreset. Native map layout is explicitly public. Literal per-role observation and rendering masks are included in the manifest and source.

The five supplied upstream reference files were byte-read for SHA256 integrity only, not semantically read or copied. No upstream source snippet was used. The assigned community ecosystem requirements refer to JaxMARL multiagent interfaces, Pgx board-game APIs, Craftax game/world mechanics, and PureJaxRL / Craftax PPO algorithm integration as distinct concerns. This original game kernel does not substitute any of those PPO implementations for the frozen learner. Runtime adapter/replay/animation are later integration responsibilities; no generated animation or empirical acceptance is claimed.

Input provenance is exact in author_input_freeze.json: all twenty enumerated files plus final batch authority and scope clarification. Current 166-169 requests/designs control identity; inherited requirement games 154-157 is retained only as the byte-exact product/interaction requirement source. No prior source, result, payoff, selection, hidden/final-test input or parent history was read.

The playthrough is a finite literal hypothesis. Its original paired intervention changes exactly one actor-action component, uses the same seed/prefix and same-length branches, and states the source-derived delay. Later negative, zero, blocked, illegal or early-terminal evidence must remain unchanged. No author-executed solvability or success claim is made.

Interface source lines: reset=18, observe=71, legal_actions=53, raw_step=76.
Literal source evidence:
- candidate.py:42: def destination_ok(self, s, i, p, direction):
- candidate.py:92: release = (a[i] == 6) | (a[i] == 9)
- candidate.py:102: claim = (a[i] == 5) & available
- candidate.py:106: toggle = (a[i] == 7) & available
- candidate.py:114: ok = moving & self.destination_ok(n, i, p, a[i])
- candidate.py:121: deliver = (a[i] == 8) & (n["cooldown"][0] == 0) & ~n["delivered"][i] & jnp.all(n["pos"][i] == n["depot_pos"][0])
- candidate.py:125: reward = -(~s["delivered"]).astype(jnp.int32) * s["urgency"] - n["claimed"].astype(jnp.int32) + n["delivery_event"].astype(jnp.int32) * (12 + 2 * s["urgency"])
- candidate.py:126: n["terminated"] = jnp.all(n["delivered"])
- candidate.py:131: n = {k: jnp.where(s["done"], s[k], n[k]) for k in s}
