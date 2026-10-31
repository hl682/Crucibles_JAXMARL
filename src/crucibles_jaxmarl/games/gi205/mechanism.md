Pollinator Paths

A=3, O=317, U=16, H=168. Six crop patches, two shelters with four bees, two gates, three wells and three strip slots form the orchard. Vertical hedges at x=4 and x=8 separate neighboring gardens; their only crossings are the real gates at (4,2) and (8,6). The static public map identifies gate sites but never reveals remote gate openness.

Native work and object handling

Actions 1-4 move cardinally on soil or through open gates. Old worker occupancy blocks swaps and following; rotating destination priority resolves contention. Invalid or masked requests become wait. Actions 5/6 pick a nearby unheld shelter with an empty produce bag or deploy the held shelter at the current soil cell without another deployed shelter. Shelter and produce are mutually exclusive carried loads; water uses a two-unit flask.

Action 7 toggles a nearby healthy gate. Voluntary closure is prohibited while any worker or bee occupies its cell. Open gates lose one health every fourth tick. A gate reaching zero stays open while occupied, then closes when clear. Action 8 spends one of the worker's two wood units to restore a nearby damaged gate to three health; periodic wear occurs after repair on that same tick. Actions 9/13 spend one of three seeds to plant the actor's inactive flowering-strip slot at the current clear soil cell or to reseed a nearby spent crop. Crop ownership remains fixed even when a neighbor supplies the labor and seed.

Action 11 fills one water unit from the lowest eligible nearby well. Wells hold at most six and regenerate one every fourth tick. Action 10 uses one carried unit on the lowest eligible nearby live crop, increasing moisture by two up to four and restoring one nectar to a blooming crop up to three. If no crop qualifies, it restores two nectar to a nearby active strip, capped at four. Actions 12/14/15 harvest, privately store or donate produce. Harvest chooses the lowest nearby blooming/ripe crop, requires an empty bag and yields one/two units respectively. It immediately converts the crop to harvested stubble and removes its nectar. Own private stall or the common barn must be within one cell to unload the entire batch. Rotating actor order rechecks resource claims. Blocked flags cover mask coercion and walking contention.

Actual bee travel and seasonal production

After worker actions, each bee selects a target using actual current gate passability. Two bounded distance fields use twenty-four fixed relaxation rounds on the nine-by-thirteen walking grid: one selects the nearest reachable nectar target, breaking ties by crop/strip index, and one selects a cardinal step toward it. Bees may share cells without blocking workers. Nectar targets must be within Manhattan distance six of their own shelter; no reachable target means return. A bee has six foraging ticks between nest renewals. Budget is decremented on each non-return tick, including a stationary visit; at zero it returns. Reaching a deployed nest in daylight renews the budget. Night, a carried shelter or lack of nectar sends it toward the shelter's current real position. A closed route can delay that return; carried shelters do not teleport bees.

A bee reaching its selected target consumes one nectar. Crop visits increment to a cap of three and the crop's local visited_now flag records the visit. Strip visits consume strip nectar but do not pollinate a crop. A strip redirects bees and spends a finite seed. Early harvesting likewise removes a target before bee movement, changing which other patch can receive the next visit. Bees use rotating processing order to resolve finite nectar claims.

Initial patches bloom with three moisture and three nectar. Young plants gain growth on each third tick while moist, blooming at growth three. A bloom with at least two bee visits gains one growth per moist tick and ripens at three. Each fifth tick live crops lose moisture; zero moisture increases stress and wet soil reduces it. Stress three kills a live crop. Bloom age eighteen also kills an unharvested bloom; death has priority over ripening on the same tick. Blooming and ripe stages have distinct nectar availability: only bloom supplies nectar. Strips start with four nectar and expire at age twelve. Daylight occupies new-clock residues zero through seventeen of a twenty-four-tick cycle; bees return in the other six ticks. These clocks affect harvesting, watering and neighbor labor.

Observation, payoff and accounting

Dynamic groups use radius-four visibility with explicit bits and -1 for unseen records. Only active strips appear. Work, blocked, seed-use and repairs are self-only and omitted from peer scene attributes. Events expose only observed delivery totals and daylight.

Private storage pays its carrier 0.6 per unit and the patch origin 0.2. Community donation pays every role 0.5 per unit and its carrier another 0.1. Every masked non-wait action costs 0.01; failure costs one. Declared reward bounds [-4,6] are conservative. Per origin, harvested = carried + private + community produce. Per worker, seeds + planted = three and wood + repairs = two. Shelter and bee identities persist; workers have unique positions. Bees and attached objects can share cells.

Natural success requires six donated units. Failure requires all crops spent/dead, no seeds and no carried produce, unless success. Scripts request gate/water work and physical shelter transport. The paired hypothesis changes only the west gate-opening action after one wait. Eight suffix ticks span gate state and bee travel; no successful pollination or measured effect is asserted.

Terminal and interface contract

All three roles remain active until joint termination. A nonterminal call increments int32 t exactly once; natural success/failure takes priority at H. Truncation is t >= H and not terminated. Already-done raw calls preserve every state leaf and observation, the clock and terminal causes, return zero float32 rewards and all dones true. There is no raw autoreset. Reset and step keys are accepted but this authored game is deterministic; the source does not persist a PRNG key or introduce a different distribution. Small signed categories use int32, flags bool, boundary observations and rewards float32. All state dimensions are fixed.

The human or policy supplies the same native joint integer action vector and receives the same observe/legal_actions interface. Any role may be assigned to a human, including multiple roles; scripts are diagnostic inputs, not a separate dynamics implementation or a policy. A future agent-keyed JaxMARL adapter and native replay UI are integration work, not claimed here. Scene coordinates, attributes and visibility are state-bound; public event fields contain only quantities present in each role observation.

Input provenance and reference limits

Only the current sanitized 202-205 authorization/scope, current designs/requests, copied contracts/schema/template/whitelist and four supplied README excerpts were used. Every actually read input is hash-receipted in author_input_freeze.json. The supplied Craftax_Baselines_ppo_rnn.py was not read. The current controlling scope supersedes historical IDs in copied generic requirements. Original mechanics and code were authored here; no upstream implementation, map, benchmark or asset was copied or imported. Craftax README lines 22-55 supplies JAX spatial-game/gymnax API context; JaxMARL README lines 111-138 explains agent-keyed parallel interfaces and __all__; Pgx README lines 54-101 supplies native state, legal-mask and explicit-key API context; PureJaxRL README lines 35-37 identifies single-file algorithm reference philosophy, not a game library or replacement learner.

Validation status and limits

AUTHORED_NOT_EXECUTED. The supplied trusted AST whitelist and manifest/scene data validators were run without importing the candidate, JAX or NumPy. JSON/source hashes, exact file names, fixed declarations, literal action ranges and the single-component paired difference were checked as data only. Requested author identity is gpt-6-astra/max; provider-measured identity, effort, tokens, elapsed model compute and cost remain unknown. Source reasoning and static acceptance do not establish runtime compatibility, numerical parity, solvability, successful play, mechanism magnitude, learning, non-isomorphism or transfer. No training, seed search, rollout, rendering, network, external model call, heldout or outcome evidence was used. Variants in timing, topology and role resources are prospective ideas only and would require separate authorization and source identity.

Literal source evidence

- candidate.py:218: n['crop_stage'] = n['crop_stage'].at[q].set(jnp.where(harvest, 3, n['crop_stage'][q])) — Early harvest removes a real nectar source before bee movement.
- candidate.py:123: return jax.lax.fori_loop(0, 24, relax, d) — Bee path distances use a bounded native grid computation.
- candidate.py:257: eligible = available & (jnp.sum(jnp.abs(flowers - nest), axis=1) <= 6) & (d < 99) & n['daylight'] & ~held — Shelter location and actual reachability govern flower choices.
- candidate.py:279: n['bee_pos'] = n['bee_pos'].at[b].set(pos) — Each pollinator has a real updated position.
- candidate.py:275: n['visits'] = n['visits'].at[c].set(jnp.minimum(n['visits'][c] + cv.astype(jnp.int32), 3)) — Only a native bee visit increments crop pollination.
- candidate.py:300: n['gate_open'] = n['gate_open'] & ((n['gate_health'] > 0) | occupiedgate) — Wear cannot close a gate onto an occupying worker or bee.
- candidate.py:17: s['crop_vis'] = jnp.sum(jnp.abs(s['agent_pos'][:, None, :] - s['crop_pos'][None, :, :]), axis=2) <= 4 — Crop visibility is local to the worker.
- candidate.py:311: n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None) — Terminal raw calls preserve all state and return zero reward.

Static method locations: sync: lines 12-23; reset: lines 25-69; walk_grid: lines 71-74; walk_targets: lines 76-81; legal_actions: lines 83-112; distances: lines 114-123; observe: lines 125-139; advance: lines 141-308; raw_step: lines 310-313.
