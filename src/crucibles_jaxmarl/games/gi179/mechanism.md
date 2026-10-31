# Filter Couriers

Status: AUTHORED_NOT_EXECUTED. Original source attempt 00 in the train-only prospective stream. The four supported controlled-role counts and dimensions belong to this batch interface, not a universal game quota.

Native A/O/U/H = 3/251/11/96. Source SHA-256: `d4be54af328065660b722ca32aaa60d2e62e4378d803bece02858f14e18c3acf`.

## Implemented rules and transition order

The public 9x7 floor has a partition at x=4 except crossings y=1,3,5. Couriers start at (3,3),(5,3),(2,1) with one issued load each. Loads 0 and 2 target station (7,5); load 1 targets station (1,1). Every agent starts with stamina 20; portable filters are [2,0,2], station stocks [2,2]. Floor dust starts at (3,3)=2,(4,3)=3,(5,3)=2.

Each original issuer independently draws a private priority 1..3. That issuer receives its own load quality-weighted delivery reward even after a handoff. No medical interpretation is attached to dust, burden or quality.

Movement permissions use old state: floor, bounds, old-empty destination and stamina at least 1+(old target dust>=2). Competing legal moves to an empty target go to the lowest id; losers stay without travel cost. Swaps and movement into a peer old cell are blocked. Waiting, including substituted invalid actions, recovers one stamina capped at 20.

Before cleanup/contact, movement deposits at the departure cell up to one of four finite load dust units and one carried body unit. Explicit spill deposits up to two remaining load units and all body burden. Deposits saturate each cell at three; excess increments vented. Depositing consumes exactly that load stock/body amount. There is no autonomous diffusion or hidden random spread.

A sweep spends one portable filter and subtracts three dust from its own cell and cardinal neighbors, with simultaneous overlapping sweeps summed and clamped at zero. Afterward every final-position courier contacts floor dust, including waiting couriers. An available filter consumes one dust and one filter. Otherwise contact lowers carried load quality by one; one dust unit transfers to body if its capacity of three permits. At capacity, contacted floor dust stays, while quality can still fall.

Delivery, handoff, drop/pick and refill are mutually exclusive actions. Handoff needs an old empty adjacent peer; actual acceptance additionally requires final adjacency and recipient wait. Each donor offers to the lowest-id accepting peer; conflicting offers select the lowest load id. Drop puts a load on its actual cell; pickup takes the lowest-id grounded load there. A courier can carry at most one load. Delivered owner is -2, grounded owner -1.

A station transfers one filter from finite stock when the caller old filter count is below four. Unique occupancy prevents competing refill requests at one station. Sum(portable filters)+sum(station stocks)+filters_consumed remains eight by the update equations. Floor plus body plus unspilled load dust is bounded above by the initial nineteen units; removal and venting cannot create dust.

Reward equals issuer_priority*(delivered load quality/6) for its own issued load, plus 0.5 for the actual delivering courier, plus 0.1 times all current deliveries for each role, minus 0.01 for movement and 0.03 for unfiltered exposure. Quality can reach zero without erasing or declaring a failed load impossible. All three deliveries terminate; otherwise H=96.

## Action and observation permissions

Movement requires in-bounds floor, sufficient old stamina and an old-empty target. Highest-level mask is local: every target is within sensing radius. Sweep needs a filter; deliver needs own carried load at its destination; drop/pick needs a carried or grounded load; handoff needs an adjacent empty peer; spill needs carried dust; refill needs shared stock and filters below four. Handoff acceptance additionally requires the recipient to wait. Invalid actions become wait; terminal roles permit only wait.

Static walls, station coordinates/stocks and cumulative delivery receipts are public. Courier, load and cell dust data are visible within Manhattan radius two with no wall occlusion. Each original issuer alone sees its priority. Hidden numeric fields are zeroed with separate visibility flags. Scene events contain public levels only.

The manifest covers every returned flat observation column and every literal reset-state key. All state leaves are declared as fixed-shape bool/int32/float32. Signed small integer fields use int32 losslessly. The uint32 reset key is trusted provenance and is not copied into state; its literal private-value distribution is preserved. No map or entity is made public just to render it. Private values are omitted from entity attributes and available only through the self observation fields. Local dust events are not broadcast as global scene events.

Declared observation bounds are [0,1]; reward bounds are [-0.1,3.8]. These are interface bounds, not performance gates, and have not been tested with numerical trajectories.

## Partner-dependent strategy hypotheses

A courier can use a scarce filter to protect a shared crossing before a peer reaches it, spend it only on personal contact, or take a longer clean route that costs stamina and time.

Handoff allows a high-priority issuer to rely on a better-positioned carrier. The carrier gains a handling bonus but can expose the issued load to pollution. Drop and reassignment allow logistics changes without destroying load identity.

Rushing a dusty load through a common route can impose quality and stamina costs on later peers. Saving filter stocks, cleaning early, waiting for peers to clear passages and detouring are meaningful alternatives in the rules; their learned use is untested.

## Original scripts and paired intervention

Literal seed 1796047 was chosen without calling reset or searching seeds. `playthrough.json` contains fixed integer joint-action scripts and a one-component paired intervention. Their intended events and legality are source reasoning only.

A single sweep by courier 0 is expected to remove center-crossing dust before courier 1 enters under an identical next action. Compare courier 1 travel stamina, burden and its carried load quality over the suffix. No effect magnitude or replay success is claimed.

A paired suffix may be compared as actual observations, native state and legal masks only after a separately authorized frozen-source replay. This package contains no trace, observed effect magnitude, acceptance score or proof of success.

## Terminal and rendering boundary

A nonterminal raw call advances t exactly once. Natural completion takes priority at H; truncated = (next_t >= H) and not terminated. done equals terminated or truncated. An already-done raw call preserves all state, observations, clock and original causes and returns zero reward with all role dones true. The raw kernel never resets; outer autoreset belongs to trusted integration.

The declarative scene binds actual native entity positions, active/visibility masks, public attributes and event levels to exact source/manifest hashes. It supplies a usable state mapping for later rendering, not recorded animation. Tether midpoints, dust cells, rights and bridge components derive from actual causal state where present. A privileged full-state view must be labelled separately from each role view.

The array interface is GeneratedEnv reset/observe/legal_actions/raw_step. A JaxMARL agent-keyed adapter with __all__, real trajectory capture, animation playback/pause/step, R0 and any learning work remain independent integration responsibilities; none is silently added to these seven source-package files.

## Literal source evidence

- `candidate.py:29`: `walls = ((x == 4) & ((y == 0) | (y == 2) | (y == 4) | (y == 6))).astype(jnp.int32)`
- `candidate.py:43`: `'priority': jax.random.randint(key, (3,), 1, 4, dtype=jnp.int32),`
- `candidate.py:77`: `move = jnp.all(target == clipped, axis=2) & ~wall & ~occupied & (s['stamina'][:, None] >= cost)`
- `candidate.py:112`: `from_load = jnp.minimum(s['load_dust'][c], jnp.where(a == 9, 2, moved.astype(jnp.int32))) * has.astype(jnp.int32)`
- `candidate.py:117`: `deposited = s['dust'].at[s['pos'][:, 1], s['pos'][:, 0]].add(spilled)`
- `candidate.py:123`: `sweep = 3 * jnp.sum(clean[:, None] & reach, axis=0).astype(jnp.int32)`
- `candidate.py:127`: `contact = dust[pos[:, 1], pos[:, 0]] > 0`
- `candidate.py:134`: `damage = jnp.sum(jax.nn.one_hot(c, 3, dtype=jnp.int32) * (has & exposed)[:, None], axis=0)`
- `candidate.py:149`: `handoff = offer & ~duplicate`
- `candidate.py:169`: `reward = s['priority'].astype(jnp.float32) * (quality / 6.0) * delivered_now`
- `candidate.py:21`: `dv = jnp.sum(jnp.abs(s['pos'][:, None, :] - s['tile_pos'][None, :, :]), axis=2) <= 2`
- `candidate.py:192`: `local_agents = jnp.concatenate([av[:, :, None], jnp.where(av[:, :, None], af[None, :, :], 0)], axis=2).reshape(3, 18)`
- `candidate.py:203`: `n, reward = jax.lax.cond(state['done'],`

Method locations from stdlib AST only: carried 12-14, refresh 16-24, reset 26-68, legal_actions 70-93, advance 95-179, observe 181-200, raw_step 202-207.

## Allowed reference use and source-only checks

The supplied JaxMARL README was read for its parallel agent-keyed API and __all__ convention. The supplied PureJaxRL README was read to distinguish end-to-end algorithm implementations from a game library. No upstream implementation or game rules were copied. The Pgx README, Craftax README and Craftax PPO-RNN baseline were byte-read and hashed for integrity only; their contents were not semantically used. Craftax game mechanics, Pgx board APIs and PureJaxRL algorithm code are not claimed as implementations of these games. All exact paths, bytes, hashes and actual use are recorded in author_input_freeze.json.

The supplied AST whitelist accepted this final source, and supplied manifest/scene validators accepted data-only metadata. Class constants, literal reset-key inventory, action vector dimensions/ranges, one-component branch changes and SHA links were checked without importing or executing candidate code. These checks do not establish runtime shape/dtype correctness, eager/JIT/vmap parity, action-mask validity along trajectories, conservation, causality, solvability, strategic richness, non-isomorphism, animation, training, adaptation or transfer. Zero, negative, failed and indeterminate future results remain admissible.
