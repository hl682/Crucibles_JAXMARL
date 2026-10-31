# Beacon Council

AUTHORED_NOT_EXECUTED. Original initial source attempt 00 for game-v3-00212. Requested author gpt-6-astra / max; provider measured identity, tokens and cost unknown.

Three settlements send their physical caravans through a mountain wall with one crossing. The first two travel west-to-east to different markets; the third travels east-to-west. Caravans never disappear after unloading and can still escort or relay for partners. A caravan carries its own single original shipment until it unloads within one tile of its assigned market. It starts with five fuel and can collect two additional two-unit bundles from its own home, with a fuel ceiling of nine.

Reservations spend one fuel near a home/beacon, claim a public five-tick lease only if unclaimed, and govern entry on later ticks. A reservation holder can relinquish it. Escorts spend one fuel and an action to give an adjacent loaded peer three protection ticks; protected entry can cross a closed pass and consumes protection. Thus holding a window, sharing it, escorting and retaining fuel have opposing private/time consequences. Duplicate movement destinations block every contender, and stationary occupants block traffic.

Weather comes only from the supplied step key: when clear, a Bernoulli(0.18) event starts three closure ticks; while closed it counts down. The input-state weather governs movement. The next weather becomes directly visible only near the pass. Actors in range update their own report and its actual observation tick. Relay sends the actor's input-state report unchanged to the beacon chain: delay 2 at the chosen beacon, then +2 per index hop. Packets with at least as recent a stamp can replace a pending packet, potentially postponing delivery; equal-stamp competing relays resolve by rotating actor order. Arrival never changes the original timestamp. Local read actions copy a beacon only if it is no older than the actor's memory; direct sight takes priority when observations are rebuilt. A stale clear report may therefore disagree with hidden current weather without leaking that weather.

## Native interface and source evidence

A/O/U/H = (3, 145, 12, 150). Humans and policies choose the same ordered native integer actions. Any subset of the three roles can be assigned to humans by the shared platform; this package adds no separate browser transition or controller restriction. Legal masks are contextual affordances, not a proof that an interaction will find an object. Out-of-range or mask-false actions become wait. Eligible interactions resolve in rotating actor order (t + k) mod 3; ties among eligible objects use lowest native object index. Movement rejects destinations occupied at the start of the tick and duplicate destinations; swaps and following into a simultaneously vacated cell therefore wait.

- candidate.py:L13-L35: reset.
- candidate.py:L37-L47: visibility.
- candidate.py:L49-L64: observe.
- candidate.py:L66-L81: legal_actions.
- candidate.py:L83-L146: advance.
- candidate.py:L148-L150: raw_step.

## Action legend

- 0: Wait.
- 1: Travel north.
- 2: Travel east.
- 3: Travel south.
- 4: Travel west.
- 5: Reserve an unclaimed five-tick crossing lease near own home or a beacon; spend one fuel.
- 6: Relinquish own crossing lease.
- 7: Escort lowest-index adjacent loaded peer for three ticks; spend one fuel.
- 8: Read nearest adjacent beacon report into own memory if at least as recent.
- 9: Relay own dated report from nearest adjacent beacon; spend one fuel.
- 10: Unload own cargo within one tile of own destination market.
- 11: Collect two fuel from own home's finite stock when fuel <= 7.

## Observations and scene

Public information is the map, settlement/market locations, clock/terminal causes, council reservation/lease and market receipts. Agents, their visible fuel/guard/stock/load, beacons and cargo are observed only within Manhattan radius two. Current pass weather is shown only within radius two of its physical position. Each actor additionally receives its own report and original timestamp. Beacon payloads and timestamps are local; pending packets are never observed. Visibility masks suppress all hidden record data to -1. Legal movement does not reveal hidden actors; weather/lease entry checks only occur when the acting caravan is already adjacent to the pass and can directly see it. All state leaves and exact observation slices are in manifest.json. Every visible attribute is native state; scenes introduce no hidden policy data. Public event channels contain only intentionally public values. Debug/full-state inspection must be separate from role view. Use a parchment valley map, purple mountain spine, blue caravans, ochre settlement flags and amber beacons. Preserve the square grid and physical pass; no artificial hex conversion.

## End states, reward and invariants

A delivery gives its owner 2.0 reward and each peer 0.25, with -0.01 per advancing tick. Equivalently reward_i = 1.75*own_new_receipt + 0.25*sum(new_receipts) - 0.01. Natural completion when all three original shipments are delivered; otherwise truncate at tick 150. Natural termination has priority on the cap step. Already done input is absorbing with unchanged state, observation and clock; rewards become zero and terminal causes persist. No raw autoreset occurs. The declared bounds and invariants are source obligations, not observed numerical results. Three original cargo objects follow their owning caravan until deactivated at the correct market. Each home has two finite fuel bundles. Fuel spends exactly one for an accepted reservation, escort or relay. Pending report time is never rewritten to arrival time. A lease lasts at most five ticks and escort protection at most three.

## Hypotheses and provenance limits

Two literal scripts and one single actor/action paired intervention are supplied in playthrough.json; suffix lengths follow the implemented timers or locations. All have UNEXECUTED_AUTHOR_HYPOTHESIS status, and the seed is literal rather than searched. They do not certify legal trajectories, solvability or mechanism success. Prospective tests could vary weather process, beacon-chain delay, private versus shared delivery rewards, starting fuel and partner lease behavior in a separate prospective source; none has been run. This source does not establish compatibility, numerical parity, strategic richness, learning, generalization or transfer.

The actual allowed-input inventory, full-byte hashes and excerpt extents are frozen in author_input_freeze.json. Current scope overrides historical IDs in copied generic requirements. Four supplied official README excerpts provided API context: JaxMARL parallel dictionaries, Pgx explicit-key/terminal concepts, Craftax spatial world/interface context, PureJaxRL single-file implementation philosophy. No benchmark implementation, layout, upstream code or assets were copied; the supplied PPO implementation was not read. Static author checks parse data/AST only and import only supplied trusted stdlib validators. Candidate/JAX/NumPy import, native calls, numerical replay, rendering, training, seed search and network/model calls were not performed.

## Literal rule evidence

candidate.py:L98: n['beacon_report'] = jnp.where(fresh,s['pending_report'],s['beacon_report'])

candidate.py:L125: n['delay'] = jnp.where(accept,2+2*jnp.abs(jnp.arange(3,dtype=jnp.int32)-b),n['delay'])

candidate.py:L124: n['pending_stamp'] = jnp.where(accept,s['stamp'][i],n['pending_stamp'])

candidate.py:L115: escort = (a==7)&(n['fuel'][i]>0)&jnp.any(neighbors)

candidate.py:L109: reserve = (a==5)&(near|home)&(n['fuel'][i]>0)&(n['reservation']<0)

candidate.py:L137: storm = jax.random.bernoulli(key,0.18)

candidate.py:L45: n['report'] = jnp.where(direct,(s['weather'][0]>0).astype(jnp.int32),s['report'])

candidate.py:L145: reward = 1.75*deliveries+0.25*jnp.sum(deliveries)-0.01

candidate.py:L149: n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(key,state,actions),operand=None)

