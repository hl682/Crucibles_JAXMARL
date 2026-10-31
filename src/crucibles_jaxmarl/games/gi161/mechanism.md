# Tetherwake — game-v3-00161

Status: AUTHOR_ONLY_NOT_EXECUTED. Source SHA256: `f58a6fc02ef1f4542e3bd11b49c1914cdcb401f41f51f6e198b518df8cef4b87`.

Native contract: A=3, O=210, U=14, H=160. Fixed-shape dictionary state, public coordinate order xy, map indexing yx.

This is an integer grid puzzle, not an orbital mechanics simulator. The public 11 by 9 map has a vertical bulkhead at x=5, with gates at (5,3) and (5,5). Cargo starts at (7,3) and (8,6) and must reach its assigned dock (2,3) or (2,5). The rigger, drive operator and technician start at (6,2), (6,4), (8,3). Bollards at (8,2), (8,6) provide real optional cargo-anchor edges. The charger is at (7,4), and optional chips are at (9,2), (7,7), (3,5).

Each crew member has at most one cargo cable. The rigger's length limit is three and the others two, in Manhattan distance. Action 5 detaches an existing link or connects the nearest reachable active cargo; exact ties use the lowest cargo index. Walking while attached must preserve that limit. The cargo does not automatically follow an individual move, so advancing the network requires staging crew positions, deliberate detachment or synchronized pull commands. No walking swap is allowed, and all agents targeting a shared cell fail. Cargo and bodies block one another.

A powered pull needs two or more matching compass votes from agents attached to that cargo, including the drive operator's matching vote. Other attached crew who do not vote still constrain the proposed displacement. Every crew cable, any bollard cable, the gate, bodies and other cargo must permit the destination. The cargo moves one cell and the shared battery loses two power; crew do not teleport with it. An unsuccessful pull consumes no power. The single drive operator can support only its attached cargo on that turn, creating a real allocation constraint between modules.

The rigger toggles the nearest active cargo's anchor when within three cells. Attaching requires the rigger to be adjacent to a bollard and cargo within length three of it. Detaching an existing anchor requires cargo proximity but not bollard proximity. Anchoring prevents drift without a power cost, at the price of constraining subsequent pulls to the fixed bollard radius. Anchor coordinates therefore directly affect feasible paths, not merely the rendering.

Every third transition, active unanchored cargo without a successful pull attempts to move east. Walls, gates, bodies and other cargo can block drift. If the proposed drift would overextend any attached crew cable, holding costs one shared power. At zero power all crew cables on that cargo snap and the unobstructed drift proceeds. The snapped agents stay at their own positions. Cargo is processed in index order, so the first module has priority for a last unit of shared holding power. This is an explicit deterministic contention rule. An anchor prevents both the drift and its holding bill.

Gate phase uses the starting clock in four-tick blocks, top gate first. A gate occupied by cargo or a crew member remains passable, avoiding a closing gate overwriting a body. The technician can override the nearest adjacent gate, spending three shared power and setting its timer to three before movement. Timers decrease on subsequent transitions. The returned state recomputes the next clock's visible gate phase. Recharging near the charger adds four up to capacity twenty-four. Chip collection by a nearby agent uses rotating priority, adds two shared power up to that same cap, and gives the collector 0.3 reward. Every active transition has -0.001 reward and each cargo recovery gives +1 to every crew member.

Helping a drive operator can move a module while sacrificing a personal chip detour. Staying attached can permit a later pull but may prevent another crew's desired route or drain the shared battery during drift. Anchoring lets a technician leave to recharge without repeated holding costs, but must be released before moving beyond the bollard radius. Saving power for a gate competes with immediate pulling and emergency holding. These are native cross-agent resource and opportunity effects. Their usefulness and learnability are hypotheses only.

The five cable slots are the three crew links and two cargo-anchor links. cable_start, cable_vector and cable_length bind the actual endpoints used by the constraints. A separate ten-endpoint entity group records both exact positions with cable IDs and side IDs, so the generic scene can display each paired endpoint even without a custom line renderer. Active masks suppress disconnected cables; no fake decorative rope is supplied. The scene also binds actual pull, drift, strain, snap and recovery flags. All information is public and included in or derivable from native observations.

Power remains 0..24. Every active crew cable and anchor cable remains within its length limit after a transition; an otherwise violated unpowered crew link is explicitly removed. Active cargo and agents occupy distinct walkable cells. Recovered cargo stays at its dock, becomes inactive and releases associated links. Natural termination occurs only when both modules are recovered; a battery of zero is not automatically failure because recharge remains possible. The raw terminal state and clock are absorbing. Future rule variations in cable length, gate phase, drive-role allocation and anchor geometry could require different staging, but none are tested here.

## Action legend

- 0: Wait.
- 1: North.
- 2: East.
- 3: South.
- 4: West.
- 5: Attach nearest reachable cargo or detach current cable.
- 6: Vote pull north.
- 7: Vote pull east.
- 8: Vote pull south.
- 9: Vote pull west.
- 10: Technician hold nearest gate open for three ticks.
- 11: Technician recharge four shared power at charger.
- 12: Collect nearby personal salvage chip and add two shared power.
- 13: Rigger toggle nearest cargo anchor.

## Native observation layout

- [0, 3): Observer role one-hot in fixed agent order; visibility=self.
- [3, 102): Public walls and gate tiles, row-major yx; visibility=public.
- [102, 108): Crew positions; visibility=public.
- [108, 111): Equipment roles; visibility=public.
- [111, 114): Cargo attachment IDs or minus one; visibility=public.
- [114, 117): Maximum agent cable lengths; visibility=public.
- [117, 120): Current attached cable Manhattan lengths; visibility=public.
- [120, 123): Latest sanitized commands; visibility=public.
- [123, 126): Collected chips by crew member; visibility=public.
- [126, 130): Cargo positions; visibility=public.
- [130, 132): Cargo delivered flags; visibility=public.
- [132, 134): Anchor IDs or minus one; visibility=public.
- [134, 136): Cargo powered pull this transition; visibility=public.
- [136, 138): Cargo drift this transition; visibility=public.
- [138, 140): Attempted drift strained a crew cable this transition; visibility=public.
- [140, 143): Crew cables snapped this transition; visibility=public.
- [143, 145): Cargo delivered this transition; visibility=public.
- [145, 149): Bollard positions; visibility=public.
- [149, 153): Timed gate positions; visibility=public.
- [153, 155): Current passable gate states; visibility=public.
- [155, 157): Remaining forced-open timers; visibility=public.
- [157, 158): Shared power; visibility=public.
- [158, 162): Assigned cargo dock positions; visibility=public.
- [162, 164): Technician charger position; visibility=public.
- [164, 170): Personal salvage chip positions; visibility=public.
- [170, 173): Uncollected chip flags; visibility=public.
- [173, 183): Crew and anchored-cargo cable starts; visibility=public.
- [183, 193): True displacement to other endpoint for each cable; visibility=public.
- [193, 198): Connected cable flags; visibility=public.
- [198, 203): Cable Manhattan length; visibility=public.
- [203, 208): Active cable at its maximum permitted length; visibility=public.
- [208, 210): Clock and joint done flag; visibility=public.

## Literal source evidence

- candidate.py:32: `s['cable_vector'] = ends - s['cable_start']`
- candidate.py:131: `rope_ok = (s['attached'] < 0) | (jnp.sum(jnp.abs(q - attached_cargo), axis=1) <= s['rope_limit'])`
- candidate.py:148: `crew = (votes[direction] >= 2) & participants[1] & (ac[1] == direction + 6)`
- candidate.py:150: `tether_ok = jnp.all((s['attached'] != c) | (jnp.sum(jnp.abs(s['pos'] - q), axis=1) <= s['rope_limit']))`
- candidate.py:154: `pull = s['cargo_active'][c] & crew & (s['power'][0] >= 2) & tether_ok & anchor_ok & self.clear(s, q) & ~occupied`
- candidate.py:163: `powered_hold = strain & (s['power'][0] > 0)`
- candidate.py:167: `s['attached'] = jnp.where(broken, -1, s['attached'])`
- candidate.py:184: `s = jax.tree.map(lambda new, old: jnp.where(state['episode_done'], old, new), s, state)`

Method source extents: gates lines 12–16, clear lines 18–21, refresh lines 23–43, reset lines 45–78, legal_actions lines 80–90, raw_step lines 92–186, observe lines 188–196. These are static source pointers, not executed traces.

## Reference provenance and boundary

Official community references were read only from this batch's supplied snapshots. JaxMARL informed the public multiplayer reset/step and agent-role conventions; Pgx informed explicit legal-action and terminal-state API distinctions. Craftax informed native state-backed game objects and presentation separation. PureJaxRL and the supplied Craftax PPO recurrent baseline were integration references for pure array batching and outer wrappers, not copied game libraries, not executed training code, and not replacements for the experimental learner. No upstream source code or map was copied into this game. Exact snapshot hashes and URLs are preserved in generation_record.json and author_input_freeze.json. No external network search occurred.

AUTHOR_ONLY_NOT_EXECUTED. The candidate source was never imported or evaluated, JAX was not imported by the author tooling, and reset/raw_step/observe/legal_actions were not called. Permitted checks were stdlib AST parsing and compile-only, the frozen static whitelist, data-only manifest/scene validation, literal vector structure and exact file hash bindings. These checks do not establish returned runtime shapes, numerical parity, trajectory legality, completion, strategic richness, learning, community-adapter acceptance, replay or animation. There are no experimental results, scorer outputs, learned-policy evidence or fabricated successes in this package. Three or four agent support is the frozen batch scope. Signed discrete state is represented as int32; bool flags and float32 native observations/rewards follow the contract. The supplied PRNG key is not duplicated in state. An external agent-keyed JaxMARL adapter and actual renderer playback remain separate integration work.
