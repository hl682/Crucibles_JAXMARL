# Orbit Salvage

Status: AUTHORED_NOT_EXECUTED. Original source attempt 00 in the train-only prospective stream. The four supported controlled-role counts and dimensions belong to this batch interface, not a universal game quota.

Native A/O/U/H = 3/44/8/96. Source SHA-256: `05b69a62c906aaabddd68e35bd6c5b8877b664f2dc4595ea1637576418bb451b`.

## Implemented rules and transition order

The arena has integer cell centers x=0..12,y=0..10. Tugs start at (5,5),(6,4),(7,5), cargo at (6,5), docks at (2,5),(10,5). Each velocity component is an integer in [-1,1]. Coasting retains velocity. This is a discrete endpoint-collision game, not a continuous mechanics model; diagonal crossing trajectories are not swept for collisions.

At reset, each tug independently samples a binary private preference from the supplied reset key. Its preferred dock pays 4 and the other 1. Coincident preferences are possible. Peers cannot recover preferences from a shared parity bit. The raw-step key is accepted but unused because transitions are deterministic given state and actions.

Valid latch/release occurs before thrust. A newly latched tug stores its cardinal offset; all latched offsets thereafter move rigidly with the cargo. Releasing retains the prior inherited velocity. Tether midpoint and displacement fields are computed from actual endpoints, and active tethers are exactly the attached mask.

Each powered thrust or brake consumes one fuel. Attached thrust vectors, including brake forces opposite cargo component velocity, are summed. Per-axis acceleration is sign(total_force)*floor(abs(total_force)/2), then velocity is clipped to [-1,1]. Thus one attached tug cannot initiate a stopped heavy body alone; disagreement can spend fuel without acceleration. Free-tug brake directly zeros its velocity.

The proposed rigid assembly is rejected if cargo or any attached endpoint leaves bounds or enters an old free-tug cell. Rejection zeros assembly velocity and costs one cargo integrity if attempted velocity was nonzero. Free tugs resolve afterward against the settled assembly and old other free-tug cells; all contenders for the same free destination fail. A rejected moving free tug retains position and receives zero velocity. Position swaps through old free occupancy are blocked.

Giving moves one fuel from a donor to the adjacent old-position peer with least old fuel, ties by lower id. Transfers are simultaneous; movement does not retroactively cancel the handover. Sum(fuel)+burned remains 54 by the literal update equations, an unexecuted invariant claim.

Docking tests the next clock and final position/velocity: exact dock center, cargo speed zero, at least two attached tugs, positive integrity, and next_t modulo 6 less than 3. All tugs receive their private dock value, including unattached helpers or free riders. Reward additionally subtracts 0.02 per powered action, 0.15 per moving free-tug collision and 0.1 for a cargo hit. Declared per-step bounds [-0.3,4] contain the rule formula.

Natural termination is docking or cargo integrity zero. No additional fuel-based solvability rule is invented. A collision reducing integrity to zero cannot also yield docking. Otherwise the time limit is 96.

## Action and observation permissions

Only coast at joint terminal. Fuel actions require positive fuel; toggling requires attachment or cardinal adjacency; giving requires an old adjacent peer. Coast is always valid. All out-of-range or masked actions become coast. No per-agent early removal.

All geometry, fuel, velocity, attachment, integrity and dock windows are public. Each role sees only its own preferred dock. Visibility matrices are all true for public groups; private_visible is the identity and private values are omitted from scene attributes.

The manifest covers every returned flat observation column and every literal reset-state key. All state leaves are declared as fixed-shape bool/int32/float32. Signed small integer fields use int32 losslessly. The uint32 reset key is trusted provenance and is not copied into state; its literal private-value distribution is preserved. No map or entity is made public just to render it. Private values are omitted from entity attributes and available only through the self observation fields. Local dust events are not broadcast as global scene events.

Declared observation bounds are [-1,1]; reward bounds are [-0.3,4.0]. These are interface bounds, not performance gates, and have not been tested with numerical trajectories.

## Partner-dependent strategy hypotheses

Coordinated impulses followed by shared braking save time but create dependence on partners continuing or releasing their tethers. A single useful thrust can be withheld while keeping later actions fixed.

When preferences conflict, a tug can support an unfavorable port for a smaller shared payout, oppose the other direction, retain fuel for later influence, or detach and reposition. Free riding receives the same terminal valuation without paid thrust, while cargo damage is shared.

Fuel gifts can keep a necessary brake or partner thrust available; a donor sacrifices its later maneuvering budget. These are source-level strategy hypotheses, not demonstrated equilibria, learnability or richness.

## Original scripts and paired intervention

Literal seed 1785903 was chosen without calling reset or searching seeds. `playthrough.json` contains fixed integer joint-action scripts and a one-component paired intervention. Their intended events and legality are source reasoning only.

With only agent 1 first suffix action changed, the source force threshold predicts a change in shared cargo velocity and later positions of agents 0 and 2. Both branches keep the later coasting and braking actions identical. This is a causal hypothesis, not an observed effect.

A paired suffix may be compared as actual observations, native state and legal masks only after a separately authorized frozen-source replay. This package contains no trace, observed effect magnitude, acceptance score or proof of success.

## Terminal and rendering boundary

A nonterminal raw call advances t exactly once. Natural completion takes priority at H; truncated = (next_t >= H) and not terminated. done equals terminated or truncated. An already-done raw call preserves all state, observations, clock and original causes and returns zero reward with all role dones true. The raw kernel never resets; outer autoreset belongs to trusted integration.

The declarative scene binds actual native entity positions, active/visibility masks, public attributes and event levels to exact source/manifest hashes. It supplies a usable state mapping for later rendering, not recorded animation. Tether midpoints, dust cells, rights and bridge components derive from actual causal state where present. A privileged full-state view must be labelled separately from each role view.

The array interface is GeneratedEnv reset/observe/legal_actions/raw_step. A JaxMARL agent-keyed adapter with __all__, real trajectory capture, animation playback/pause/step, R0 and any learning work remain independent integration responsibilities; none is silently added to these seven source-package files.

## Literal source evidence

- `candidate.py:20`: `preference = jax.random.randint(key, (3,), 0, 2, dtype=jnp.int32)`
- `candidate.py:73`: `attached = jnp.where(a == 6, ~s['attached'], s['attached'])`
- `candidate.py:79`: `total_force = jnp.sum(jnp.where(attached[:, None], force, 0), axis=0)`
- `candidate.py:80`: `acceleration = jnp.sign(total_force) * (jnp.abs(total_force) // 2)`
- `candidate.py:90`: `group_blocked = group_wall | cargo_free | arm_free`
- `candidate.py:105`: `contention = jnp.any(other & (~attached)[None, :] &`
- `candidate.py:62`: `has_fuel = s['fuel'] > 0`
- `candidate.py:120`: `docking = (jnp.all(s['dock_pos'] == cp, axis=1) & jnp.all(cv == 0) &`
- `candidate.py:127`: `reward = reward - 0.02 * spent.astype(jnp.float32) - 0.15 * collision.astype(jnp.float32)`
- `candidate.py:148`: `return jnp.concatenate([own, jnp.broadcast_to(public, (3, 33)), jnp.eye(3)], axis=1).astype(jnp.float32)`
- `candidate.py:151`: `n, reward = jax.lax.cond(state['done'],`

Method locations from stdlib AST only: refresh 12-17, reset 19-56, legal_actions 58-66, advance 68-135, observe 137-148, raw_step 150-155.

## Allowed reference use and source-only checks

The supplied JaxMARL README was read for its parallel agent-keyed API and __all__ convention. The supplied PureJaxRL README was read to distinguish end-to-end algorithm implementations from a game library. No upstream implementation or game rules were copied. The Pgx README, Craftax README and Craftax PPO-RNN baseline were byte-read and hashed for integrity only; their contents were not semantically used. Craftax game mechanics, Pgx board APIs and PureJaxRL algorithm code are not claimed as implementations of these games. All exact paths, bytes, hashes and actual use are recorded in author_input_freeze.json.

The supplied AST whitelist accepted this final source, and supplied manifest/scene validators accepted data-only metadata. Class constants, literal reset-key inventory, action vector dimensions/ranges, one-component branch changes and SHA links were checked without importing or executing candidate code. These checks do not establish runtime shape/dtype correctness, eager/JIT/vmap parity, action-mask validity along trajectories, conservation, causality, solvability, strategic richness, non-isomorphism, animation, training, adaptation or transfer. Zero, negative, failed and indeterminate future results remain admissible.

During this unsealed initial authoring pass, static privacy review replaced a shared reset preference bit with independently sampled per-role bits. No numerical feedback or alternate source attempt was used.
