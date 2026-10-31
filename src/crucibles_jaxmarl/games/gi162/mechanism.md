# Canal Commons — game-v3-00162

Status: AUTHORED_NOT_EXECUTED. This package is an original blind author proposal in the prospective spatial train stream. No candidate import, reset, step, JIT, rollout, seed search, training or browser replay was performed by its author. Static source/data checks are described separately and establish none of those results.

## Rules, resource consequences and strategic hypotheses

The three gardens begin with four moisture units each. Forty further units start in a finite reservoir; up to sixteen active packets are represented explicitly. Existing packets move one grid cell per step before a new packet is released. A main-channel packet meeting a condition-zero mouth is destroyed. Otherwise an open shutter turns it into that garden branch. Branch arrival occurs at y=5; a packet that passes all mouths drains after x=12. These are physical route decisions, not reward-only transfers.

Any gardener within one Manhattan cell of a mouth can toggle or maintain it. Even toggle counts cancel; maintenance restores condition to three and takes priority over the every-twelve-step deterioration. Garden ownership restricts drawing and harvesting, while pouring can help any nearest unresolved garden. Each fourth step, a wet unresolved crop consumes one moisture and grows toward five. Twelve consecutive dry steps wilt it permanently. Moisture clipping can spill arrivals; water conservation is not claimed after spilling, plant use or drainage. Harvest gives four privately plus one quarter to every gardener; wilt costs the owner one. A toggle/draw/pour costs one eighth, maintenance one quarter, and every live raw step one sixty-fourth.

Partner-dependent choices include taking early upstream packets versus closing a mature garden's shutter, paying for a mouth that chiefly serves downstream players, and carrying water to a peer while sacrificing a private action. Conditioning maintenance or future sharing on a peer's earlier shutter behaviour is possible. These are proposed strategic affordances, not a proof of multiple equilibria or learnability. The fixed candidate does not implement a random family of canal layouts. Changing source budget, crop demand, ownership order or maintenance intervals would require a separately frozen variant and could change needed responses.

## Native interface, visibility and termination

The kernel class is GeneratedEnv with A=3, O=118, U=10, H=64. reset(key) returns float32[A,O] observation and a fixed-key array dictionary. observe(state) returns the same role-native observations. legal_actions(state) returns bool[A,U]. raw_step(key,state,int32[A]) returns observation, actual next state, float32[A] rewards, bool[A] joint dones, and scalar done_all/terminated/truncated. The source uses only the two allowed JAX imports. Integers use int32 and booleans use bool; the PRNG key is supplied externally and is not duplicated into game state.

An invalid, out-of-range or masked action is deterministically converted to wait. Every agent retains wait. A nonterminal raw call increments t once. Natural termination has priority over the time limit. Already terminal calls preserve all leaves and observations, keep the original terminal cause and clock, and return zero rewards. Reset is never called by raw_step. All three gardens have been harvested or permanently wilted; otherwise the natural state is truncated at 64 raw steps.

Fully public world and state. Each observation begins with own-role one-hot, then the labelled public dynamic vector. Immutable terrain, mouth/garden coordinates and ownership by index are known game rules. Packet visibility equals activity. Co-location on walkable canal/bank cells is permitted.

The map, entity coordinates, active masks, visibility matrices, attribute values and events in manifest.json bind actual leaves. scene.json adds only inert presentation labels/glyphs/colours. Per-step events are separate from persistent levels and cumulative histories. Per-role frames must obey the supplied visibility matrices; debug full-state frames must be labelled separately. Native animated frames and any community adapter still require independent execution/integration. No animation is fabricated in this package.

Agent co-location is intentionally permitted. Declared nonnegative state leaves: tank, integrity, moisture, growth, dry, reservoir, water_route. Their enforcement is an unexecuted source assertion, not an R0 receipt. Observation bounds are 0 through 64; per-step reward bounds are -2 through 6. These are contract envelopes, never acceptance-score gates. Static geometry is public known game data; it is not extra privileged learner information.

The JaxMARL-compatible public adapter is a separate integration responsibility: map agent_0 through agent_2 dictionaries to the fixed array axis, preserve the exact observations and masks, and expose the actual info.done_all as the done dictionary's __all__. The adapter must not substitute a strategy, add observation fields, change role inactivity or reset semantics, or silently change the frozen experimental learner. Pgx sequential board APIs are not this game's simultaneous action protocol.

## Literal action evidence and limits

playthrough.json contains a complete source-reasoned spatial interaction episode and an independent paired hypothesis with exactly one changed agent action. All action vectors are literal integers. The seed is fixed at zero without looking up its outcome. The scripts are unexecuted hypotheses; legality, completion, visual comprehensibility and learnability remain unverified. A later executor must record actual rejected or ineffective actions and failure states, rather than rewriting the script as though success occurred. No optimality, reward learning, statistical superiority, diversity metric, generalization or meta-RL result is claimed.

## Literal source locations

- Finite reset and actual layout: candidate.py:13 — `gate_pos = jnp.array`.
- Observation vector: candidate.py:42 — `public = jnp.concatenate`.
- Spatial action masks: candidate.py:53 — `dg = jnp.sum`.
- Delayed packet route changes: candidate.py:84 — `# Existing packets move`.
- Storage and crop dynamics: candidate.py:106 — `# Storage transfers`.
- Natural terminal and rewards: candidate.py:121 — `term = jnp.all(harvested | dead)`.
- Absorbing raw boundary: candidate.py:132 — `ns = jax.tree.map`.

## Reference use and author provenance

The frozen official JaxMARL README supplied the parallel agent-keyed integration concept and __all__ convention. Craftax README supplied the broad inspiration of a stateful spatial resource game with native reset/step state, not copied recipes or a benchmark implementation. Pgx README supplied the distinction between native JAX game state, legal masks and external randomness; no Pgx board rules were copied. PureJaxRL README and the supplied Craftax Baselines PPO/RNN excerpt are algorithm integration references only, not game libraries and not a replacement learner. No upstream code was copied. Exact snapshot URLs/hashes remain in the frozen request and author_input_freeze.json; no external pages were fetched.

Author dispatch: gpt-6-astra, reasoning effort max, task /root/astra_community_games_162_165. This is the assignment's model identity; the coordinator's actual dispatch receipt is the authoritative execution provenance. No model API was called by this author. Input identity: current ordinal 162, train only, parentless initial window, source attempt zero. The inherited delivery-requirements games list is a preserved older scope label, explicitly clarified by root_dispatch_scope_clarification.json.
