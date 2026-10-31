# Bridge Pact

Status: AUTHORED_NOT_EXECUTED. Original source attempt 00 in the train-only prospective stream. The four supported controlled-role counts and dimensions belong to this batch interface, not a universal game quota.

Native A/O/U/H = 4/137/11/120. Source SHA-256: `41fd76cd8f3a07938911f7529ee4b6f4591ba9f5f43b175c780f754426fdf19e`.

## Implemented rules and transition order

Two islands occupy x=0..2 and x=6..8 in a 9x7 world. Potential crossings each contain three actual cells at x=3,4,5 on y=1 or y=5. Other water is impassable. Four builders start at (2,1),(2,2),(6,1),(6,2) with [wood,bolts] inventories [4,0],[0,4],[0,4],[4,0]. Each resource totals eight.

A contribution reaches the current or a cardinal-adjacent bridge cell with that component missing. The lowest bridge index is targeted. For simultaneous requests for the same material and cell, the lowest agent id contributes and losers retain their material. Complementary wood and bolts can arrive together. Components persist permanently; first completion sets integrity twelve.

Movement permissions use old geometry. Solid land and completed positive-health bridge cells admit entry only if the target was empty. Same empty target contention goes to lowest agent id; swaps and following into an old occupied cell are blocked. A single bridge cell has capacity one. Connectivity arises from individual adjacent steps across all three cells, with no portal or teleport.

Every actual entry causes one integrity wear. Every fourth next clock causes one decay on every completed cell, including newly completed ones on that transition. Integrity bottoms at zero. Existing occupants remain on a worn-out cell without displacement or damage; they may wait, repair, or exit to an otherwise legal neighbor. Only new entry is prohibited. A stranded actor is retained until ordinary termination.

Repair selects the reachable completed cell with least old integrity, ties by bridge index. It consumes one wood and one bolts, restores six up to twelve, and can restore an occupied zero-integrity cell. Concurrent repairs all consume their materials even if the cap wastes restoration. Materials installed in construction are not returned when integrity is zero.

Giving transfers one material to the old-adjacent peer with least old stock of that material, ties by id. Same-step gifts do not satisfy old-state action permissions. For each resource, carried stock plus installed components plus cumulative repair_used remains eight by the source equations.

Ordering is old-mask movement, same-component contention and simultaneous complementary contributions, repair, old-adjacency handoff, entry wear and periodic decay, then delivery. A new or repaired cell cannot legalize a move that was masked out in old state.

Each builder has independently sampled private values 2..5 for north/south opposing terminals. Own delivery pays that value; every role receives 0.2 per current delivery. An accepted contribution costs 0.02, repair costs 0.04 and movement costs 0.005. Delivered builders remain able to help. All deliveries terminate; H=120 truncates any incomplete, stranded or undecided case.

## Action and observation permissions

Movement targets must be old-empty ground or a complete positive-health bridge cell; common empty targets go to lowest agent id. Construction reaches own/cardinal-adjacent cells and needs the missing material. Giving requires positive stock and an old adjacent peer. Repair needs both resources and a reachable completed cell below maximum integrity. Delivery requires an undelivered parcel at an opposing terminal. Old live bridges govern movement; newly built cells become available next transition. A zero-health occupied cell keeps its occupant, allows waiting/repair/valid exit and rejects new entry. Joint terminal allows only wait; invalid actions become wait.

All geometry, material stock, construction components, integrity, occupancy, repair consumption and delivery flags are public. Each role alone sees its two private terminal values. The scene does not bind private value attributes; private_visible is the identity mask.

The manifest covers every returned flat observation column and every literal reset-state key. All state leaves are declared as fixed-shape bool/int32/float32. Signed small integer fields use int32 losslessly. The uint32 reset key is trusted provenance and is not copied into state; its literal private-value distribution is preserved. No map or entity is made public just to render it. Private values are omitted from entity attributes and available only through the self observation fields. Local dust events are not broadcast as global scene events.

Declared observation bounds are [0,1]; reward bounds are [-0.1,5.8]. These are interface bounds, not performance gates, and have not been tested with numerical trajectories.

## Partner-dependent strategy hypotheses

Builders can consolidate complementary stocks on one route, work from both ends and meet at the middle, or divide stock between the two alternative crossings. An early one-material commitment can induce or fail to induce a peer matching contribution.

Transporting material and yielding occupied cells can let a better-positioned peer complete a route. Spending scarce complements on repair protects everyone using an existing crossing but competes directly with opening another route.

A builder can free ride on others construction, block a narrow cell, prefer a privately valuable destination, or cooperate for the shared delivery term. These are partner-dependent possibilities in the rules, not verified strategic quality or learning.

## Original scripts and paired intervention

Literal seed 1816323 was chosen without calling reset or searching seeds. `playthrough.json` contains fixed integer joint-action scripts and a one-component paired intervention. Their intended events and legality are source reasoning only.

Only builder 0 initial wood commitment differs. Builder 1 later makes the same bolts commitment from the same bank cell. The source predicts complete versus partial first-cell construction and a different east-entry permission for builder 1; no movement into an illegal branch is scripted. This is an unexecuted delayed opportunity hypothesis.

A paired suffix may be compared as actual observations, native state and legal masks only after a separately authorized frozen-source replay. This package contains no trace, observed effect magnitude, acceptance score or proof of success.

## Terminal and rendering boundary

A nonterminal raw call advances t exactly once. Natural completion takes priority at H; truncated = (next_t >= H) and not terminated. done equals terminated or truncated. An already-done raw call preserves all state, observations, clock and original causes and returns zero reward with all role dones true. The raw kernel never resets; outer autoreset belongs to trusted integration.

The declarative scene binds actual native entity positions, active/visibility masks, public attributes and event levels to exact source/manifest hashes. It supplies a usable state mapping for later rendering, not recorded animation. Tether midpoints, dust cells, rights and bridge components derive from actual causal state where present. A privileged full-state view must be labelled separately from each role view.

The array interface is GeneratedEnv reset/observe/legal_actions/raw_step. A JaxMARL agent-keyed adapter with __all__, real trajectory capture, animation playback/pause/step, R0 and any learning work remain independent integration responsibilities; none is silently added to these seven source-package files.

## Literal source evidence

- `candidate.py:32`: `'bridge_pos': jnp.array([[3, 1], [4, 1], [5, 1], [3, 5], [4, 5], [5, 5]], jnp.int32),`
- `candidate.py:62`: `move = jnp.all(target == clipped, axis=2) & ~occupied & ((tile == 0) | bridge_ok)`
- `candidate.py:98`: `wood_lost = jnp.any((wood_target[:, None] == wood_target[None, :]) & wants_wood[None, :] & lower, axis=1)`
- `candidate.py:102`: `contributed = jnp.stack([put_wood, put_bolt], axis=1).astype(jnp.int32)`
- `candidate.py:106`: `completed_now = complete & ~((s['wood'] == 1) & (s['bolts'] == 1))`
- `candidate.py:112`: `repair_count = jnp.sum(jax.nn.one_hot(repair_target, 6, dtype=jnp.int32) * repaired[:, None], axis=0)`
- `candidate.py:116`: `materials = s['materials'] - contributed - repaired[:, None].astype(jnp.int32)`
- `candidate.py:124`: `wear = jnp.sum(entry, axis=0).astype(jnp.int32)`
- `candidate.py:126`: `decay = ((jnp.mod(t, 4) == 0) & complete).astype(jnp.int32)`
- `candidate.py:127`: `health = jnp.maximum(health - wear - decay, 0)`
- `candidate.py:133`: `reward = value + 0.2 * jnp.sum(delivered_now) - 0.02 * jnp.sum(contributed, axis=1)`
- `candidate.py:159`: `n, reward = jax.lax.cond(state['done'],`

Method locations from stdlib AST only: refresh 12-14, reset 16-50, legal_actions 52-76, advance 78-144, observe 146-156, raw_step 158-163.

## Allowed reference use and source-only checks

The supplied JaxMARL README was read for its parallel agent-keyed API and __all__ convention. The supplied PureJaxRL README was read to distinguish end-to-end algorithm implementations from a game library. No upstream implementation or game rules were copied. The Pgx README, Craftax README and Craftax PPO-RNN baseline were byte-read and hashed for integrity only; their contents were not semantically used. Craftax game mechanics, Pgx board APIs and PureJaxRL algorithm code are not claimed as implementations of these games. All exact paths, bytes, hashes and actual use are recorded in author_input_freeze.json.

The supplied AST whitelist accepted this final source, and supplied manifest/scene validators accepted data-only metadata. Class constants, literal reset-key inventory, action vector dimensions/ranges, one-component branch changes and SHA links were checked without importing or executing candidate code. These checks do not establish runtime shape/dtype correctness, eager/JIT/vmap parity, action-mask validity along trajectories, conservation, causality, solvability, strategic richness, non-isomorphism, animation, training, adaptation or transfer. Zero, negative, failed and indeterminate future results remain admissible.
