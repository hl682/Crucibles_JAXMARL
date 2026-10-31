# Beacon Auction

Status: AUTHORED_NOT_EXECUTED. Original source attempt 00 in the train-only prospective stream. The four supported controlled-role counts and dimensions belong to this batch interface, not a universal game quota.

Native A/O/U/H = 4/114/10/80. Source SHA-256: `f86308ea53c8962a99ee7ee2bba0038550c9f3b13858411f8edbd96b2bcab05a`.

## Implemented rules and transition order

The 9x7 partition has auction gates at (4,1),(4,5) and an always-available charge-burning bypass at (4,3). Fixed partners are (0,1) from the west and (2,3) from the east. Everyone starts with six liquid charge; total conserved charge is 24 including escrow and consumed charge. Movement costs no charge on ordinary floor.

Each agent has independently sampled private delivery values 2..5 for north/south terminals on the opposing side. All charge, geometry, bids, rights and clocks are public. Agents can help after delivering, but cannot deliver a second time.

A bid of one or two charge requires distance at most one from an inactive site and sufficient old liquid charge. Bids accumulate in an agent-by-site public escrow array. Clearing occurs after the transition when next_t is divisible by four. At each eligible site, highest accumulated bid wins; equal bids use ascending (agent_id - (next_t//4 modulo 4)) modulo 4. A positive bid always beats every zero bid.

Losers receive the full escrow refund. The winner payment moves to spent_charge and all that site escrow clears. The winner becomes owner with ttl six. Bidding is disabled while old ttl is positive. Rights count down after movement; the last old tick still grants access, and expiry removes ownership. Pending escrow is not forcibly refunded at time-limit termination.

Movement checks old rights and old occupancy. Entering a live gate is free for the owner pair; a rival must transfer one liquid charge to the actual owner. Unowned gates block new entry; existing occupants can wait or exit. Entry into the neutral bypass burns two charge. Competing empty destinations select the lowest id; old occupied destinations forbid swaps.

An old-adjacent fixed partner can receive two charge, with simultaneous conservative gifts. A courier standing on a beacon can relay to the other beacon when both are live and owned by its pair, the exit is old-empty and it has one charge. Relay burns one charge. This is an explicit movement edge between two native physical sites.

Ordering is old-mask validation, movement and old-right tolls, partner gifts, escrow commitments, old-right countdown, clearing and refunds, then delivery bookkeeping. Affordability is always tested on old resources, so same-step incoming transfers never authorize a previously unaffordable action.

Reward is own delivered private value plus one quarter of the partner same-step delivery value, minus 0.02 times the amount committed in this step and travel charge burned. The bid effort penalty is not refunded even when financial escrow is refunded; tolls change charge without direct reward transfer. Four deliveries terminate, otherwise H=80. Charge+escrow+spent_charge=24 is a source-derived conservation claim awaiting execution.

## Action and observation permissions

Movement rejects walls, bounds and old occupied cells. Entering an active gate is free for the owner pair; rivals need one charge, transferred to the owner. The unowned gate rejects new entry. Bypass entry requires two charge. Bid requires at most one-cell distance, inactive rights and sufficient old liquid charge. Sharing requires adjacent fixed partner; relay requires standing on a beacon, both rights held by the own pair, one charge and an old-empty destination. Delivered actors may still help; only deliver is disabled. Joint terminal permits only wait; all invalid actions become wait.

Positions, map, charge, bid escrows, rights, lifetimes, terminals, spent-charge total and delivery receipts are public. Only a role sees its two private terminal values. The scene omits value attributes; private_visible is the exact identity permission mask.

The manifest covers every returned flat observation column and every literal reset-state key. All state leaves are declared as fixed-shape bool/int32/float32. Signed small integer fields use int32 losslessly. The uint32 reset key is trusted provenance and is not copied into state; its literal private-value distribution is preserved. No map or entity is made public just to render it. Private values are omitted from entity attributes and available only through the self observation fields. Local dust events are not broadcast as global scene events.

Declared observation bounds are [0,1]; reward bounds are [-0.1,6.25]. These are interface bounds, not performance gates, and have not been tested with numerical trajectories.

## Partner-dependent strategy hypotheses

A pair can split purchases across sites to obtain a complementary relay edge, concentrate funds on one gate, donate charge to a nearby partner, or rely on a rival gate and pay its toll.

Delayed commitment permits overbidding, waiting for public bids, or declining an auction for the neutral bypass. Ties are predictable from the public clock, not random or hidden.

Private terminal preferences can justify a longer route, a bid at one site, or helping a partner complete first. Finite charge and expiry make early spending constrain future access; no equilibrium or learned strategy is asserted.

## Original scripts and paired intervention

Literal seed 1806199 was chosen without calling reset or searching seeds. `playthrough.json` contains fixed integer joint-action scripts and a one-component paired intervention. Their intended events and legality are source reasoning only.

Agent 2 alone changes wait to a two-charge late bid. At the first clearing the literal rotating tie rule predicts a different north-site owner. Agent 0 then enters the same gate, so its refund, toll event and later liquid charge should differ. No clearing or effect has been executed.

A paired suffix may be compared as actual observations, native state and legal masks only after a separately authorized frozen-source replay. This package contains no trace, observed effect magnitude, acceptance score or proof of success.

## Terminal and rendering boundary

A nonterminal raw call advances t exactly once. Natural completion takes priority at H; truncated = (next_t >= H) and not terminated. done equals terminated or truncated. An already-done raw call preserves all state, observations, clock and original causes and returns zero reward with all role dones true. The raw kernel never resets; outer autoreset belongs to trusted integration.

The declarative scene binds actual native entity positions, active/visibility masks, public attributes and event levels to exact source/manifest hashes. It supplies a usable state mapping for later rendering, not recorded animation. Tether midpoints, dust cells, rights and bridge components derive from actual causal state where present. A privileged full-state view must be labelled separately from each role view.

The array interface is GeneratedEnv reset/observe/legal_actions/raw_step. A JaxMARL agent-keyed adapter with __all__, real trajectory capture, animation playback/pause/step, R0 and any learning work remain independent integration responsibilities; none is silently added to these seven source-package files.

## Literal source evidence

- `candidate.py:66`: `can_bid = jnp.any(near & (s['ttl'][None, :] == 0), axis=1)`
- `candidate.py:61`: `permission = jnp.any(matches & owned & (free | (s['charge'][:, None, None] >= 1)), axis=2)`
- `candidate.py:101`: `tolls = entered & (s['ttl'][None, :] > 0) & (s['owner'][None, :] // 2 != team[:, None])`
- `candidate.py:109`: `shared = 2 * (a == 7).astype(jnp.int32)`
- `candidate.py:114`: `escrow = s['escrow'] + jax.nn.one_hot(site, 2, dtype=jnp.int32) * bid[:, None]`
- `candidate.py:121`: `cleared = (jnp.mod(t, 4) == 0) & (ttl == 0) & (jnp.sum(escrow, axis=0) > 0)`
- `candidate.py:122`: `priority = jnp.mod(ids - jnp.mod(t // 4, 4), 4)`
- `candidate.py:126`: `refund = jnp.sum(jnp.where(cleared[None, :] & ~winners, escrow, 0), axis=1)`
- `candidate.py:130`: `owner = jnp.where(cleared, winner, owner)`
- `candidate.py:136`: `reward = value + 0.25 * value[mate]`
- `candidate.py:149`: `public = jnp.concatenate([jnp.array([s['t'] / 80.0]),`
- `candidate.py:161`: `n, reward = jax.lax.cond(state['done'],`

Method locations from stdlib AST only: refresh 12-13, reset 15-49, legal_actions 51-81, advance 83-146, observe 148-158, raw_step 160-165.

## Allowed reference use and source-only checks

The supplied JaxMARL README was read for its parallel agent-keyed API and __all__ convention. The supplied PureJaxRL README was read to distinguish end-to-end algorithm implementations from a game library. No upstream implementation or game rules were copied. The Pgx README, Craftax README and Craftax PPO-RNN baseline were byte-read and hashed for integrity only; their contents were not semantically used. Craftax game mechanics, Pgx board APIs and PureJaxRL algorithm code are not claimed as implementations of these games. All exact paths, bytes, hashes and actual use are recorded in author_input_freeze.json.

The supplied AST whitelist accepted this final source, and supplied manifest/scene validators accepted data-only metadata. Class constants, literal reset-key inventory, action vector dimensions/ranges, one-component branch changes and SHA links were checked without importing or executing candidate code. These checks do not establish runtime shape/dtype correctness, eager/JIT/vmap parity, action-mask validity along trajectories, conservation, causality, solvability, strategic richness, non-isomorphism, animation, training, adaptation or transfer. Zero, negative, failed and indeterminate future results remain admissible.
