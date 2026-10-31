# Workshop Exchange — game-v3-00164

Status: AUTHORED_NOT_EXECUTED. This package is an original blind author proposal in the prospective spatial train stream. No candidate import, reset, step, JIT, rollout, seed search, training or browser replay was performed by its author. Static source/data checks are described separately and establish none of those results.

## Rules, resource consequences and strategic hypotheses

Each of four workers has a fixed complementary tool and its own workstation. One raw unit starts a job lasting three ticks for types zero/two and four ticks for types one/three. The completed component remains on a real shelf until the owner collects it. Work continues asynchronously if the owner leaves. There are initially eight carried raw units and sixteen gatherable units; private sales and assembled machines consume this finite material. Co-location is allowed, while trades require actual Manhattan-distance-one proximity.

Posting removes one own-type component into escrow and asks for either one next-type component or one raw unit. A live offer lasts eight ticks; cancellation or unsold expiry returns that exact component. Buyers name their closest affordable pre-step offer, breaking distance ties by seller index. A public rotating buyer priority clears the fixed joint intentions; each offer settles at most once. Same-step seller cancellation takes precedence. Newly posted offers cannot be bought until the next step. Eligibility uses pre-step inventory, so an agent cannot create a policy-dependent action cascade by spending a just-received payment. Receiving and selling an existing escrow on the same step is permitted and accounted for.

At the assembly cell, each deposit intention chooses the lowest missing component the worker held pre-step. Rotating priority decides duplicate-slot claims. Filling all four slots starts three further ticks of assembly. On completion everyone receives one, and each contributed slot gives its depositor one more. Selling a component privately at the worker's own station pays one and a half immediately. Starting work costs one eighth, posting/cancelling one sixteenth, and time one sixty-fourth. There is no automatic replanning, ghost inventory or renderer-manufactured handoff.

Specialization can support reciprocal exchange, selling can strand the common recipe, and reservations may temporarily make useful parts unavailable. Waiting for better terms costs time and risks expiry; a worker with many acquired types can earn multiple contribution credits. No result proves market efficiency, optimal prices, non-isomorphism or meta-learning. A separately frozen family could vary workstation travel, production delays, raw scarcity, recipes and expiry rules; this source fixes those choices.

## Native interface, visibility and termination

The kernel class is GeneratedEnv with A=4, O=94, U=14, H=96. reset(key) returns float32[A,O] observation and a fixed-key array dictionary. observe(state) returns the same role-native observations. legal_actions(state) returns bool[A,U]. raw_step(key,state,int32[A]) returns observation, actual next state, float32[A] rewards, bool[A] joint dones, and scalar done_all/terminated/truncated. The source uses only the two allowed JAX imports. Integers use int32 and booleans use bool; the PRNG key is supplied externally and is not duplicated into game state.

An invalid, out-of-range or masked action is deterministically converted to wait. Every agent retains wait. A nonterminal raw call increments t once. Natural termination has priority over the time limit. Already terminal calls preserve all leaves and observations, keep the original terminal cause and clock, and return zero rewards. Reset is never called by raw_step. Completion of two shared machines ends the game; otherwise truncate at 96 steps. Finite raw supply can be consumed by private sales, leaving shared completion impossible within that trajectory.

All inventories, reservations, stock, jobs, contributions and spatial positions are public. Own-role one-hot precedes the public vector. Immutable tool assignment, station/pile locations and topology are known rules. Colocation is permitted for physical exchange at a common workshop tile; only Manhattan-distance-one handoffs settle.

The map, entity coordinates, active masks, visibility matrices, attribute values and events in manifest.json bind actual leaves. scene.json adds only inert presentation labels/glyphs/colours. Per-step events are separate from persistent levels and cumulative histories. Per-role frames must obey the supplied visibility matrices; debug full-state frames must be labelled separately. Native animated frames and any community adapter still require independent execution/integration. No animation is fabricated in this package.

Agent co-location is intentionally permitted. Declared nonnegative state leaves: raw, inventory, job_timer, shelf, stock, offer_ttl, assembly_parts, machine_timer, products. Their enforcement is an unexecuted source assertion, not an R0 receipt. Observation bounds are -1 through 96; per-step reward bounds are -1 through 8. These are contract envelopes, never acceptance-score gates. Static geometry is public known game data; it is not extra privileged learner information.

The JaxMARL-compatible public adapter is a separate integration responsibility: map agent_0 through agent_3 dictionaries to the fixed array axis, preserve the exact observations and masks, and expose the actual info.done_all as the done dictionary's __all__. The adapter must not substitute a strategy, add observation fields, change role inactivity or reset semantics, or silently change the frozen experimental learner. Pgx sequential board APIs are not this game's simultaneous action protocol.

## Literal action evidence and limits

playthrough.json contains a complete source-reasoned spatial interaction episode and an independent paired hypothesis with exactly one changed agent action. All action vectors are literal integers. The seed is fixed at zero without looking up its outcome. The scripts are unexecuted hypotheses; legality, completion, visual comprehensibility and learnability remain unverified. A later executor must record actual rejected or ineffective actions and failure states, rather than rewriting the script as though success occurred. No optimality, reward learning, statistical superiority, diversity metric, generalization or meta-RL result is claimed.

## Literal source locations

- Specialist physical layout: candidate.py:13 — `stations = jnp.array`.
- Physical trade eligibility: candidate.py:50 — `def offers`.
- Asynchronous jobs: candidate.py:82 — `job_done = s["job_timer"] == 1`.
- Escrow settlement: candidate.py:99 — `# Every buyer names`.
- Expiry and return: candidate.py:121 — `expired = live &`.
- Complementary timed assembly: candidate.py:132 — `# Deposits are real parts`.
- Private sale alternative: candidate.py:149 — `sold = ac == 12`.
- Rewards and natural termination: candidate.py:152 — `contribution = jnp.sum`.
- Absorbing raw boundary: candidate.py:165 — `ns = jax.tree.map`.

## Reference use and author provenance

The frozen official JaxMARL README supplied the parallel agent-keyed integration concept and __all__ convention. Craftax README supplied the broad inspiration of a stateful spatial resource game with native reset/step state, not copied recipes or a benchmark implementation. Pgx README supplied the distinction between native JAX game state, legal masks and external randomness; no Pgx board rules were copied. PureJaxRL README and the supplied Craftax Baselines PPO/RNN excerpt are algorithm integration references only, not game libraries and not a replacement learner. No upstream code was copied. Exact snapshot URLs/hashes remain in the frozen request and author_input_freeze.json; no external pages were fetched.

Author dispatch: gpt-6-astra, reasoning effort max, task /root/astra_community_games_162_165. This is the assignment's model identity; the coordinator's actual dispatch receipt is the authoritative execution provenance. No model API was called by this author. Input identity: current ordinal 164, train only, parentless initial window, source attempt zero. The inherited delivery-requirements games list is a preserved older scope label, explicitly clarified by root_dispatch_scope_clarification.json.
