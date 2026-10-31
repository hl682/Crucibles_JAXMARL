# Beacon Convoy — game-v3-00163

Status: AUTHORED_NOT_EXECUTED. This package is an original blind author proposal in the prospective spatial train stream. No candidate import, reset, step, JIT, rollout, seed search, training or browser replay was performed by its author. Static source/data checks are described separately and establish none of those results.

## Rules, resource consequences and strategic hypotheses

Reset draws exactly one dangerous crossing with jax.random.randint(key, (), 0, 2). Agent zero receives both danger bits; the two carriers initially receive unknown values. Everyone knows the road geometry and crossing locations. A role retains direct survey knowledge once its vehicle comes within Manhattan distance one of a crossing. Public damage, claims and audits permit further inference; the implementation does not pretend those public clues reveal nothing. Radio contents remain arbitrary controlled-agent actions, never an oracle response.

A moving agent entering a dangerous crossing needs another same-step guard within one cell of the crossing. Without protection the vehicle remains at its old position, loses one hull and two extra fuel in addition to movement fuel. Two hits wreck it. Same-cell convoy formation is explicitly allowed. Each move, radio claim, guard or promise consumes one fuel. A two-fuel gift goes to the adjacent active peer with lowest pre-step fuel, index breaking ties; simultaneous gifts can spill above the cap of thirty-two. Each depot holds eight units and refuels up to four under rotating priority. The two routes and different private goals create choices between regrouping, detouring, free-riding on escort, providing aid and delivering early.

Messages last eight ticks. A current unaudited claim is checked when any vehicle attempts entry onto the claimed crossing; correct and false claim counters are public. A pledge lasts ten ticks and counts as kept only if its maker actually guards a peer's dangerous crossing on the promised lane. Merely guarding a safe route does not count. An unresolved expired pledge counts as broken; this is interaction history, not an enforced belief model. Delivery yields four privately and a quarter to all agents, while hits/wrecks and communication/guard/aid carry their source-defined costs. A delivered or wrecked role stays at its recorded position but is inactive and can only wait.

Risk identity is a native reset variation. Different fuel budgets, private destination arrangements, deadlines or hidden-information assignments could create other response requirements, but are not silently included in this candidate. No measured trust, deception, cooperation, adaptation or transfer is claimed.

## Native interface, visibility and termination

The kernel class is GeneratedEnv with A=3, O=79, U=13, H=64. reset(key) returns float32[A,O] observation and a fixed-key array dictionary. observe(state) returns the same role-native observations. legal_actions(state) returns bool[A,U]. raw_step(key,state,int32[A]) returns observation, actual next state, float32[A] rewards, bool[A] joint dones, and scalar done_all/terminated/truncated. The source uses only the two allowed JAX imports. Integers use int32 and booleans use bool; the PRNG key is supplied externally and is not duplicated into game state.

An invalid, out-of-range or masked action is deterministically converted to wait. Every agent retains wait. A nonterminal raw call increments t once. Natural termination has priority over the time limit. Already terminal calls preserve all leaves and observations, keep the original terminal cause and clock, and return zero rewards. Reset is never called by raw_step. All agents have delivered or wrecked; otherwise truncate at 64 raw steps. Delivered and wrecked roles are inactive with wait as their only legal action.

Public topology, vehicle telemetry, destinations, messages and actual interaction histories. Scout agent 0 knows both danger bits at reset; other roles learn a crossing only within Manhattan distance one and retain that knowledge. Observe returns only the observing role knowledge mask and danger bits, unknown=-1. Hidden danger is never in a movement mask or public map. Public damage and audited claims can support inference beyond direct sightings. Static landmark positions are known.

The map, entity coordinates, active masks, visibility matrices, attribute values and events in manifest.json bind actual leaves. scene.json adds only inert presentation labels/glyphs/colours. Per-step events are separate from persistent levels and cumulative histories. Per-role frames must obey the supplied visibility matrices; debug full-state frames must be labelled separately. Native animated frames and any community adapter still require independent execution/integration. No animation is fabricated in this package.

Agent co-location is intentionally permitted. Declared nonnegative state leaves: fuel, hull, stock, message_ttl, promise_ttl, kept, broken, honest, false_signal. Their enforcement is an unexecuted source assertion, not an R0 receipt. Observation bounds are -4 through 64; per-step reward bounds are -4 through 6. These are contract envelopes, never acceptance-score gates. Static geometry is public known game data; it is not extra privileged learner information.

The JaxMARL-compatible public adapter is a separate integration responsibility: map agent_0 through agent_2 dictionaries to the fixed array axis, preserve the exact observations and masks, and expose the actual info.done_all as the done dictionary's __all__. The adapter must not substitute a strategy, add observation fields, change role inactivity or reset semantics, or silently change the frozen experimental learner. Pgx sequential board APIs are not this game's simultaneous action protocol.

## Literal action evidence and limits

playthrough.json contains a complete source-reasoned spatial interaction episode and an independent paired hypothesis with exactly one changed agent action. All action vectors are literal integers. The seed is fixed at zero without looking up its outcome. The scripts are unexecuted hypotheses; legality, completion, visual comprehensibility and learnability remain unverified. A later executor must record actual rejected or ineffective actions and failure states, rather than rewriting the script as though success occurred. No optimality, reward learning, statistical superiority, diversity metric, generalization or meta-RL result is claimed.

## Literal source locations

- Private risk draw: candidate.py:13 — `unsafe = jax.random.randint`.
- Role-filtered observations: candidate.py:51 — `known = s["hazard_visible"]`.
- Geometry-only move mask: candidate.py:61 — `tile = s["terrain"]`.
- Peer escort transition: candidate.py:88 — `helpers = (`.
- Physical aid and shared fuel: candidate.py:97 — `# Fuel transfers`.
- Messages and audit history: candidate.py:114 — `# Radio states`.
- Actual pledge fulfilment: candidate.py:126 — `fulfilled = jnp.any`.
- Natural termination: candidate.py:133 — `term = jnp.all(delivered | wrecked)`.
- Absorbing raw boundary: candidate.py:144 — `ns = jax.tree.map`.

## Reference use and author provenance

The frozen official JaxMARL README supplied the parallel agent-keyed integration concept and __all__ convention. Craftax README supplied the broad inspiration of a stateful spatial resource game with native reset/step state, not copied recipes or a benchmark implementation. Pgx README supplied the distinction between native JAX game state, legal masks and external randomness; no Pgx board rules were copied. PureJaxRL README and the supplied Craftax Baselines PPO/RNN excerpt are algorithm integration references only, not game libraries and not a replacement learner. No upstream code was copied. Exact snapshot URLs/hashes remain in the frozen request and author_input_freeze.json; no external pages were fetched.

Author dispatch: gpt-6-astra, reasoning effort max, task /root/astra_community_games_162_165. This is the assignment's model identity; the coordinator's actual dispatch receipt is the authoritative execution provenance. No model API was called by this author. Input identity: current ordinal 163, train only, parentless initial window, source attempt zero. The inherited delivery-requirements games list is a preserved older scope label, explicitly clarified by root_dispatch_scope_clarification.json.
