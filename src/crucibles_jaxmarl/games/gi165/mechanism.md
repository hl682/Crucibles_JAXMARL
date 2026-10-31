# Crossing Accord — game-v3-00165

Status: AUTHORED_NOT_EXECUTED. This package is an original blind author proposal in the prospective spatial train stream. No candidate import, reset, step, JIT, rollout, seed search, training or browser replay was performed by its author. Static source/data checks are described separately and establish none of those results.

## Rules, resource consequences and strategic hypotheses

Agents zero/one form team zero and agents two/three form team one. Each has one individually addressed cargo destination across the river. All four remain controlled and active after delivering. The first team with both cargos delivered ends the game; simultaneous completion preserves both team counters and is not assigned a fabricated sole winner. Each bridge has exactly one grid cell and one allowed entry direction. The upper deck slides between y=1/2 and the lower between y=4/5; moving them rewrites actual map passability.

Any agent within one Manhattan cell of either switch plaza can request a slide or reversal, provided that bridge was empty at the beginning of the step and the agent can afford it. A public rotating priority selects at most one operator per bridge. Only the winner pays. Controls resolve before movements; therefore a legal pre-step entry intention can be blocked by another player's same-step bridge change. No actor is asked for a new action after seeing that change. An occupied bridge cannot move, reverse or throw off its occupant.

Movement cannot enter a cell occupied at the start of the step, even if the occupant intends to leave. Swaps, following and pushing are disallowed. Competing entries to an empty cell use the same rotating priority. Yielding must therefore happen physically before another agent can exit a bridge into that bank cell. Plaza claiming spends one energy and changes pressure toward the claimant's team. Reaching pressure plus/minus two changes persistent ownership. The owner gets a local crank discount from two energy to one. Waiting on the home bank restores one energy up to six. Every eighth step owned plazas produce a small opposing team payoff.

A delivery pays the delivering agent three, its teammate one, and each opponent minus one, before time/territory terms. Delaying at a bridge, aiding a teammate through a costly reversal, opening a rival route in exchange for future yielding, or diverting to the other rail are all source-level options. They are not measured policies or established equilibria. Changing rail placement, movement conflict semantics, goal arrangement or energy economics requires new frozen rule identity and may change strategic responses.

## Native interface, visibility and termination

The kernel class is GeneratedEnv with A=4, O=68, U=11, H=96. reset(key) returns float32[A,O] observation and a fixed-key array dictionary. observe(state) returns the same role-native observations. legal_actions(state) returns bool[A,U]. raw_step(key,state,int32[A]) returns observation, actual next state, float32[A] rewards, bool[A] joint dones, and scalar done_all/terminated/truncated. The source uses only the two allowed JAX imports. Integers use int32 and booleans use bool; the PRNG key is supplied externally and is not duplicated into game state.

An invalid, out-of-range or masked action is deterministically converted to wait. Every agent retains wait. A nonterminal raw call increments t once. Natural termination has priority over the time limit. Already terminal calls preserve all leaves and observations, keep the original terminal cause and clock, and return zero rewards. Reset is never called by raw_step. Either team has delivered both individual cargos. If both finish on the same step the terminal state retains both completion counts; otherwise truncate at 96 steps.

Fully public topology and dynamic bridge deck map, positions, energy, territory and delivery state. Own-role one-hot identifies the actor. All four roles remain active after personal delivery so they can help or block until joint termination. No extra information is introduced by the full-state scene.

The map, entity coordinates, active masks, visibility matrices, attribute values and events in manifest.json bind actual leaves. scene.json adds only inert presentation labels/glyphs/colours. Per-step events are separate from persistent levels and cumulative histories. Per-role frames must obey the supplied visibility matrices; debug full-state frames must be labelled separately. Native animated frames and any community adapter still require independent execution/integration. No animation is fabricated in this package.

Agent positions are unique and all live bridge occupants remain on their actual deck. Declared nonnegative state leaves: energy, team_deliveries. Their enforcement is an unexecuted source assertion, not an R0 receipt. Observation bounds are -2 through 96; per-step reward bounds are -3 through 6. These are contract envelopes, never acceptance-score gates. Static geometry is public known game data; it is not extra privileged learner information.

The JaxMARL-compatible public adapter is a separate integration responsibility: map agent_0 through agent_3 dictionaries to the fixed array axis, preserve the exact observations and masks, and expose the actual info.done_all as the done dictionary's __all__. The adapter must not substitute a strategy, add observation fields, change role inactivity or reset semantics, or silently change the frozen experimental learner. Pgx sequential board APIs are not this game's simultaneous action protocol.

## Literal action evidence and limits

playthrough.json contains a complete source-reasoned spatial interaction episode and an independent paired hypothesis with exactly one changed agent action. All action vectors are literal integers. The seed is fixed at zero without looking up its outcome. The scripts are unexecuted hypotheses; legality, completion, visual comprehensibility and learnability remain unverified. A later executor must record actual rejected or ineffective actions and failure states, rather than rewriting the script as though success occurred. No optimality, reward learning, statistical superiority, diversity metric, generalization or meta-RL result is claimed.

## Literal source locations

- Two-team map and positions: candidate.py:13 — `terrain = jnp.zeros`.
- Bridge-dependent geometry: candidate.py:45 — `def crossing`.
- Occupancy and cost masks: candidate.py:50 — `def legal_actions`.
- Simultaneous crank resolution: candidate.py:81 — `for b in range(2)`.
- Physical yielding and move conflict: candidate.py:92 — `# Current occupancy`.
- Territory and resource dynamics: candidate.py:109 — `claim = ac == 9`.
- Team rewards: candidate.py:122 — `team_gain = jnp.sum`.
- Natural termination: candidate.py:129 — `term = jnp.any(team_deliveries >= 2)`.
- Absorbing raw boundary: candidate.py:136 — `ns = jax.tree.map`.

## Reference use and author provenance

The frozen official JaxMARL README supplied the parallel agent-keyed integration concept and __all__ convention. Craftax README supplied the broad inspiration of a stateful spatial resource game with native reset/step state, not copied recipes or a benchmark implementation. Pgx README supplied the distinction between native JAX game state, legal masks and external randomness; no Pgx board rules were copied. PureJaxRL README and the supplied Craftax Baselines PPO/RNN excerpt are algorithm integration references only, not game libraries and not a replacement learner. No upstream code was copied. Exact snapshot URLs/hashes remain in the frozen request and author_input_freeze.json; no external pages were fetched.

Author dispatch: gpt-6-astra, reasoning effort max, task /root/astra_community_games_162_165. This is the assignment's model identity; the coordinator's actual dispatch receipt is the authoritative execution provenance. No model API was called by this author. Input identity: current ordinal 165, train only, parentless initial window, source attempt zero. The inherited delivery-requirements games list is a preserved older scope label, explicitly clarified by root_dispatch_scope_clarification.json.
