# Hedgehaven

AUTHORED_NOT_EXECUTED. Original initial source attempt 00 for game-v3-00213. Requested author gpt-6-astra / max; provider measured identity, tokens and cost unknown.

The village contains six labeled crop beds around a sixteen-cell insect perimeter, two standing hedge objects, two loose blankets, one hoe and one four-unit watering can. Plot labels suggest neighbor responsibilities but do not restrict any actor's access. All roles may help or disrupt any plot. Protectors and tools have fixed physical identities and at most one holder; harvested produce activates exactly one predeclared object slot for its bed.

Wind alternates eastward/westward each twelve-tick day. A ground hedge one column upwind and within one row shelters a bed. Its daylight shadow extends one or two columns behind it on the same row; early-day shadow points east, later-day shadow west. A placement can therefore protect a neighbor during the night while stopping a different bed's daytime growth. Standing hedges also block gardener entry and the insect's next route cell. The insect reverses rather than passing through that obstruction, changing which plants it reaches and when. Blankets must be physically placed on a bed; they stop frost but also growth, pollination, tending and harvest. Another gardener can remove and carry the same blanket, releasing those opportunities.

Growth requires sunlight, no shade/cover, moisture and a tilled-soil unit. Growth beyond stage three additionally requires pollination. The hoe restores three soil units, while one watering-can unit restores three moisture. Both cap at six; the can refills to four beside the well, which has renewable water. Daylight evaporates one moisture unit every third tick on uncovered growing beds. Unprotected night ticks accumulate frost, permanently killing a plant at six. Mature stage-six pollinated crops can be harvested with empty hands and then transported to the actual village crate. Hedges, blankets, tool travel and competing uses of one-item hands produce delayed externalities and opportunities for reciprocity; no resulting strategy has been observed.

## Native interface and source evidence

A/O/U/H = (3, 265, 16, 180). Humans and policies choose the same ordered native integer actions. Any subset of the three roles can be assigned to humans by the shared platform; this package adds no separate browser transition or controller restriction. Legal masks are contextual affordances, not a proof that an interaction will find an object. Out-of-range or mask-false actions become wait. Eligible interactions resolve in rotating actor order (t + k) mod 3; ties among eligible objects use lowest native object index. Movement rejects destinations occupied at the start of the tick and duplicate destinations; swaps and following into a simultaneously vacated cell therefore wait.

- candidate.py:L13-L39: reset.
- candidate.py:L41-L50: shelter.
- candidate.py:L52-L54: observe.
- candidate.py:L56-L68: legal_actions.
- candidate.py:L70-L160: advance.
- candidate.py:L162-L164: raw_step.

## Action legend

- 0: Wait.
- 1: Walk north.
- 2: Walk east.
- 3: Walk south.
- 4: Walk west.
- 5: Pick lowest-index loose object within one tile; lifting a blanket uncovers its bed.
- 6: Drop held object at own tile; dropped hedge stands upright.
- 7: Place held hedge or blanket on north adjacent tile.
- 8: Place held hedge or blanket on east adjacent tile.
- 9: Place held hedge or blanket on south adjacent tile.
- 10: Place held hedge or blanket on west adjacent tile.
- 11: Use held hoe on lowest-index eligible adjacent uncovered bed; add three soil.
- 12: Fill held watering can at east well to four units.
- 13: Water lowest-index eligible adjacent uncovered bed; spend one can unit for three moisture.
- 14: Harvest lowest-index mature pollinated adjacent uncovered bed with empty hands.
- 15: Deliver held produce beside village crate.

## Observations and scene

Fully observable village. Every role receives identical flattened clock, ground map, all worker/bed/object/insect/facility positions, plot labels, crop stages/moisture/soil/frost/alive/harvested/pollination/shelter/shade/cover state, object kinds/holders/deployment/blanket assignment/water, insect route index/direction, weather phase and delivery counters. Inactive produce slots remain explicit in the native observation but are not drawn as active objects. All public scene attributes are contained in the same observation. All state leaves and exact observation slices are in manifest.json. Every visible attribute is native state; scenes introduce no hidden policy data. Public event channels contain only intentionally public values. Debug/full-state inspection must be separate from role view. Use warm garden greens, brown village tools, gold insect marker and blue well/crate landmarks. Plants expose native crop and protection attributes rather than a decorative score panel.

## End states, reward and invariants

Each advancing tick pays every gardener the total newly delivered produce, plus 0.2 for their own new deliveries, minus 0.1 per newly frost-killed plant, minus 0.01. Natural end when every bed is harvested or dead and no active produce remains in the world. This includes completed delivery and crop-loss exhaustion; otherwise truncate at tick 180. Natural termination has priority on the cap step. Already done input is absorbing with unchanged state, observation and clock; rewards become zero and terminal causes persist. No raw autoreset occurs. The declared bounds and invariants are source obligations, not observed numerical results. There are always two hedge identities, two blanket identities, one hoe and one watering can. Each of six beds produces at most one harvest object; no reseeding or object duplication occurs. Water supply at the well is renewable, but carried water is bounded 0..4. Frost only accumulates; harvested crops do not regrow.

## Hypotheses and provenance limits

Two literal scripts and one single actor/action paired intervention are supplied in playthrough.json; suffix lengths follow the implemented timers or locations. All have UNEXECUTED_AUTHOR_HYPOTHESIS status, and the seed is literal rather than searched. They do not certify legal trajectories, solvability or mechanism success. Prospective tests could vary wind/sun timing, crop frost tolerance, hedge/blanket counts, garden topology and partner tending schedules under separately authored prospective rules; none has been run. This source does not establish compatibility, numerical parity, strategic richness, learning, generalization or transfer.

The actual allowed-input inventory, full-byte hashes and excerpt extents are frozen in author_input_freeze.json. Current scope overrides historical IDs in copied generic requirements. Four supplied official README excerpts provided API context: JaxMARL parallel dictionaries, Pgx explicit-key/terminal concepts, Craftax spatial world/interface context, PureJaxRL single-file implementation philosophy. No benchmark implementation, layout, upstream code or assets were copied; the supplied PPO implementation was not read. Static author checks parse data/AST only and import only supplied trusted stdlib validators. Candidate/JAX/NumPy import, native calls, numerical replay, rendering, training, seed search and network/model calls were not performed.

## Literal rule evidence

candidate.py:L46: n['sheltered'] = jnp.any(hedges[None,:]&(dx==s['wind'])&(dy<=1),axis=1)

candidate.py:L47: n['shaded'] = jnp.any(hedges[None,:]&(dx*s['sun']>=1)&(dx*s['sun']<=2)&(dy==0),axis=1)&s['daylight']

candidate.py:L48: n['covered'] = jnp.any((s['cover_bed'][None,:]==jnp.arange(6,dtype=jnp.int32)[:,None])&s['item_active'][None,:]&(s['holder'][None,:]==-1),axis=1)

candidate.py:L103: blanketplace = (n['kind'][h]==1)&jnp.any(onbed)&~blanketthere&n['alive'][bed]&~n['harvested'][bed]

candidate.py:L137: n['route_direction'] = jnp.where(blocked,-s['route_direction'],s['route_direction'])

candidate.py:L140: visit = (jnp.sum(jnp.abs(n['bed_pos']-n['insect_pos'][0]),axis=1)<=1)&~n['covered']&n['alive']&~n['harvested']&n['daylight']

candidate.py:L144: growth = growing&n['daylight']&~n['covered']&~n['shaded']&(n['moisture']>0)&(n['soil']>0)&((n['growth']<3)|n['pollinated'])

candidate.py:L149: cold = growing&~n['daylight']&~n['sheltered']&~n['covered']

candidate.py:L154: n['terminated'] = jnp.all(n['harvested']|~n['alive'])&~jnp.any(n['item_active']&(n['kind']==4))

candidate.py:L163: n,r = jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((3,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)

