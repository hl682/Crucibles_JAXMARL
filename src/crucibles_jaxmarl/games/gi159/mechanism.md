# Sealed Routes — game-v3-00159

Status: AUTHOR_ONLY_NOT_EXECUTED. Source SHA256: `4514da16de62d922be1a440c0a7a41528f409b76bfd83b0316cfe13d7123aa47`.

Native contract: A=3, O=171, U=15, H=128. Fixed-shape dictionary state, public coordinate order xy, map indexing yx.

The public road map is an 11 by 7 grid with a one-cell central crossing at (5,3). Depots are at (1,1), (9,1), (9,5). Couriers start at (3,3), (4,3), (7,3), each holding its own parcel and four credits. The supplied reset key is split into two independent JAX keys: destinations are uniform integer depot IDs 0..2, and values are uniform integers 4..8. The author did not draw these values or inspect any seed outcome. Seed zero in the scripts identifies a future reset, not known private labels.

Every parcel has a public shell and a separately visibility-gated manifest. Original ownership, actual possession and a paid local inspection confer lasting knowledge. The native observation masks every unknown true destination and value with -1. Merely seeing the parcel shell, its carrier, an offer or a claim does not reveal truth. The renderer's private-label group uses the same knowledge matrix and disappears when the physical parcel is inactive. All public message and contract attributes are included in the ordinary observation. No privileged destination is bound to the public shell.

Only the original owner holding an uncommitted own parcel can post. Action 7 or 8 escrows one or three credits, records the current claim, gives five ticks to accept and fixes the delivery deadline at posting tick plus eighteen. Posting consumes a turn and locks money; it does not transfer cargo or reward anyone. An adjacent empty courier must later choose action 9, surrendering one collateral credit. A newly posted offer cannot be accepted in the same transition. Acceptance transfers the parcel and reveals its true manifest to the carrier, while the sponsor gains a free capacity slot. Committed parcels cannot be dropped until the contract ends.

Timely delivery at the true depot mints the private parcel value to its original owner. The contractor receives escrow plus its returned bond. The owner's delivery reward is 0.1 times value minus fee; the carrier receives 0.1 times fee. Self delivery returns any pending escrow without a cancellation charge. A failed accepted deadline refunds escrow to the owner and transfers the bond to it, with rewards +0.1 to the owner and -0.1 to the contractor. The parcel remains with the late courier and can still be delivered, but carries no surviving relay fee. This makes accepting a poor route costly. A pending offer that expires or is cancelled refunds escrow minus one burned credit and gives the owner -0.1 reward. Buying an inspection burns one credit and costs -0.1 reward.

Expiry and deadline checks use the next tick and run before movement and interaction. Delivery exactly on the stated deadline remains eligible; a later delivery does not. Walking rejects walls, all initially occupied destinations and simultaneous destination conflicts. All interaction requests use rotating agent priority. Floor pickup selects the first eligible parcel; acceptance and inspection select eligible parcel IDs by cyclic priority. Public offers can travel with their owner. Only one recipient may consume each offer. Inspection and taking possession are different actions with different capacity and economic effects.

A rapid owner route can avoid fee and negotiation delay. Relay can free capacity, assign a nearby depot route to someone else, or move traffic out of the central crossing. High fees can attract a counterpart but reduce sponsor profit. A cheap unverified claim can save an inspection turn, but a wrong claim can waste travel time and cause bond loss; truth is never automatically inferred from the claim. A receiver may inspect first, refuse, or accept and then adapt to the revealed destination. These are plausible source-based strategies, not measured equilibria or learned behavior.

Credits, escrow and bonds remain nonnegative. The accounting identity is sum(credits)+sum(escrow)+sum(bond)+fees_burned = 12+minted. Delivery is the only source of minted money; cancellation and inspection are the only sinks. Minted value is at most 24. Each parcel has at most one carrier, each courier at most one parcel, and agent positions are distinct. Floor parcels may overlap other objects. The public fee/deadline fields retain the most recent terms after settlement; active offer_by and contractor distinguish live commitments from historical fields. Natural completion means all three delivered, not a positive-profit threshold. Terminal state is absorbing.

The playthrough first frees courier 1's capacity, broadcasts a literal claim, posts a quote and requests acceptance. It does not assume that the chosen claim matches the seed's actual destination and does not claim delivery. A second script deliberately lets a quote expire. Variants of map bottlenecks, information costs, fee levels and acceptance windows could favor different negotiation and relay practices; none are evaluated here.

## Action legend

- 0: Wait.
- 1: North.
- 2: East.
- 3: South.
- 4: West.
- 5: Drop uncommitted cargo or pick up nearby floor parcel.
- 6: Deliver carried parcel at its true depot.
- 7: Post one-credit contingent delivery fee.
- 8: Post three-credit contingent delivery fee.
- 9: Accept adjacent existing offer with one-credit bond.
- 10: Broadcast claimed depot zero.
- 11: Broadcast claimed depot one.
- 12: Broadcast claimed depot two.
- 13: Cancel own pending offer; lose one credit.
- 14: Inspect nearby unknown manifest; pay one credit.

## Native observation layout

- [0, 3): Observer role one-hot in fixed agent order; visibility=self.
- [3, 80): Public road map, row-major yx; visibility=public.
- [80, 86): All courier positions; visibility=public.
- [86, 89): Spendable credits; visibility=public.
- [89, 92): Latest sanitized actions; visibility=public.
- [92, 95): Public current claimed depot or minus one; visibility=public.
- [95, 98): Broadcast lifetime; visibility=public.
- [98, 104): Parcel shell positions; visibility=public.
- [104, 107): Original parcel owners; visibility=public.
- [107, 110): Current parcel carriers or minus one; visibility=public.
- [110, 113): Parcel status: floor zero, carried one, delivered two; visibility=public.
- [113, 116): Pending offer sponsor or minus one; visibility=public.
- [116, 119): Most recently quoted contingent fee; visibility=public.
- [119, 122): Locked sponsor fee; visibility=public.
- [122, 125): Last tick on which pending offer may be accepted; visibility=public.
- [125, 128): Last tick for contracted delivery; visibility=public.
- [128, 131): Bound courier or minus one; visibility=public.
- [131, 134): Locked contractor collateral; visibility=public.
- [134, 137): Claim captured when quote was posted; unverified; visibility=public.
- [137, 140): Latest accepted contracts; visibility=public.
- [140, 143): Latest parcel deliveries; visibility=public.
- [143, 146): Latest deadline failures; visibility=public.
- [146, 149): Latest offer expiration or cancellation; visibility=public.
- [149, 152): Couriers that bought an inspection this transition; visibility=public.
- [152, 158): Public depot positions in depot ID order; visibility=public.
- [158, 162): Clock, joint done, cumulative burned credits and cumulative minted delivery value; visibility=public.
- [162, 165): This observer knows each parcel true manifest; visibility=self.
- [165, 171): True destination and value for each parcel, masked to minus one when unknown to this observer; visibility=local.

## Literal source evidence

- candidate.py:18: `s['known'] = s['known'] | (s['carrier'][None, :] == jnp.arange(3)[:, None])`
- candidate.py:89: `late = (s['contractor'][p] >= 0) & (tick > s['deadline'][p])`
- candidate.py:91: `refund = jnp.where(cancel, s['escrow'][p] - 1, 0) + jnp.where(late, s['escrow'][p] + s['bond'][p], 0)`
- candidate.py:123: `post = ((ac[i] == 7) | (ac[i] == 8)) & (s['carrier'][i] == i) & (s['offer_by'][i] < 0) & (s['contractor'][i] < 0)`
- candidate.py:136: `accept = (ac[i] == 9) & ~jnp.any(s['carrier'] == i) & jnp.any(offers) & (s['credits'][i] >= 1)`
- candidate.py:156: `payout = jnp.where(contracted, s['escrow'][p], 0) * deliver.astype(jnp.int32)`
- candidate.py:188: `private = jnp.where(s['known'][:, :, None], secret[None, :, :], -1).reshape(3, 6)`
- candidate.py:177: `s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)`

Method source extents: refresh lines 12–23, reset lines 25–54, legal_actions lines 56–66, raw_step lines 68–179, observe lines 181–189. These are static source pointers, not executed traces.

## Reference provenance and boundary

Official community references were read only from this batch's supplied snapshots. JaxMARL informed the public multiplayer reset/step and agent-role conventions; Pgx informed explicit legal-action and terminal-state API distinctions. Craftax informed native state-backed game objects and presentation separation. PureJaxRL and the supplied Craftax PPO recurrent baseline were integration references for pure array batching and outer wrappers, not copied game libraries, not executed training code, and not replacements for the experimental learner. No upstream source code or map was copied into this game. Exact snapshot hashes and URLs are preserved in generation_record.json and author_input_freeze.json. No external network search occurred.

AUTHOR_ONLY_NOT_EXECUTED. The candidate source was never imported or evaluated, JAX was not imported by the author tooling, and reset/raw_step/observe/legal_actions were not called. Permitted checks were stdlib AST parsing and compile-only, the frozen static whitelist, data-only manifest/scene validation, literal vector structure and exact file hash bindings. These checks do not establish returned runtime shapes, numerical parity, trajectory legality, completion, strategic richness, learning, community-adapter acceptance, replay or animation. There are no experimental results, scorer outputs, learned-policy evidence or fabricated successes in this package. Three or four agent support is the frozen batch scope. Signed discrete state is represented as int32; bool flags and float32 native observations/rewards follow the contract. The supplied PRNG key is not duplicated in state. An external agent-keyed JaxMARL adapter and actual renderer playback remain separate integration work.
