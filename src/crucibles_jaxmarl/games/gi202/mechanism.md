Counterweight Foundry

A=3, O=182, U=13, H=160. This is a block workshop with one five-cell beam, two traveling side controls, four distinct counterweights and three conserved crate identities. The stepped beam raster is determined by the two endpoint rows, not by a display animation. Its x coordinates run from three through seven; integer interpolation with midpoint rounding determines each occupied y cell. Workers occupy distinct floor cells and cannot walk into the current beam. Rails and loading/delivery tiles are public floor markings; the actual beam footprint governs obstruction.

Actions and ordering

Actions 0-4 wait or walk one cardinal cell. Actions 5/6 pick the lowest eligible crate/weight within Manhattan distance one; weights may be removed from either mounted end. Action 7 drops at the worker cell. Actions 8/9 request one row of local winch travel from exactly the corresponding side pad, with empty hands. Action 10 braces if free-handed within one cell of any beam segment. Action 11 loads one carried crate beside the centre, with an empty beam and tilt at most one. Action 12 mounts a held weight from a side pad, up to two per end.

The mask describes current geometric availability; out-of-range or masked actions become wait. Walks use old worker occupancy, so swaps and entering a just-vacated cell are disallowed. Equal-destination walkers use rotating priority (agent_id - old_t) modulo three. Other contested object actions are processed in rotating order and recheck object possession; losing an object race has no object effect. The blocked flag covers coercion and lost walking contention, not every context race. Carried objects follow their worker. Bracing is taken from the legal joint action before end motion, so its benefit can enable both colleagues during that same step.

The source first resolves worker motion and object transfers, then computes the two winch requests. A loaded end can move with at least one weight mounted there or an active brace; an empty beam needs neither. Workers cannot crank while holding an object. Both requested endpoints are clipped to rows one through seven. If any proposed beam cell overlaps a worker, the whole beam motion is cancelled and jam is raised. Control pads follow the ends, requiring operators to reposition as the beam travels. Weights attached to ends move with them.

Persistent mechanics and alternatives

On a beam-moving step, positive strain equals absolute endpoint difference plus absolute difference in mounted weight counts. Positive strain is added to tension; otherwise tension decreases by one. A brace subtracts two more, and tension is clipped to zero through eight. A stationary tilted beam therefore relaxes tension by one per tick, while retaining its tilted footprint. An unbraced loaded beam with tilt at least two drops its one attached crate to its current centre cell. The crate remains a real recoverable loose object even beneath the beam footprint; workers may pick it from an adjacent cell. Tension eight marks the loaded crate lost and ends the episode even if empty. A level beam at row seven automatically delivers its attached crate; the empty beam can be returned for the others.

The initial west/east operators each have one installed weight. The guide can brace, clear an obstructing route, carry a spare weight or handle crates. Synchronized motion avoids tilt but occupies both operators. A guide's brace can temporarily replace a removed weight and damp strain, at the cost of a third action and that worker's carrying opportunity. Taking a colleague's mounted weight frees a transportable object while reducing that colleague's feasible loaded motion. Deliberately moving one end can change a passage before the other end follows, with recovery or tension consequences. These are source-defined alternatives, not measured strategies.

Accounting, reward and termination

Every crate is exactly one of floor/carried/beam/delivered/lost, with at most one on the beam. Each weight has one identity and at most one holder or mount. A worker has at most one carried object. Delivered and lost crates are inactive scene entities, with status and final coordinates still observed. Non-wait actions after mask coercion cost 0.01; winching or bracing costs another 0.02. Every shipment pays each role 1 and its owner another 0.5. Each slip/breakage costs each role 0.2; failure costs each role 2. Rewards have declared conservative finite bounds [-4,6]. Natural success requires all three deliveries; any lost crate or tension eight causes failure unless success. These ledgers and position/range declarations are intended source invariants, not numerical validation results.

The first literal script requests a series of level moves while operators follow the pads. The second removes and parks a weight, then requests a braced lift. The paired hypothesis changes only the east operator's one crank action after a common wait, retaining the guide's brace and subsequent waits. Its three-step suffix is intended to expose changed beam geometry and its persistence without claiming shipment or successful play.

Terminal and interface contract

All three roles remain active until joint termination. A nonterminal call increments int32 t exactly once; natural success/failure takes priority at H. Truncation is t >= H and not terminated. Already-done raw calls preserve every state leaf and observation, the clock and terminal causes, return zero float32 rewards and all dones true. There is no raw autoreset. Reset and step keys are accepted but this authored game is deterministic; the source does not persist a PRNG key or introduce a different distribution. Small signed categories use int32, flags bool, boundary observations and rewards float32. All state dimensions are fixed.

The human or policy supplies the same native joint integer action vector and receives the same observe/legal_actions interface. Any role may be assigned to a human, including multiple roles; scripts are diagnostic inputs, not a separate dynamics implementation or a policy. A future agent-keyed JaxMARL adapter and native replay UI are integration work, not claimed here. Scene coordinates, attributes and visibility are state-bound; public event fields contain only quantities present in each role observation.

Input provenance and reference limits

Only the current sanitized 202-205 authorization/scope, current designs/requests, copied contracts/schema/template/whitelist and four supplied README excerpts were used. Every actually read input is hash-receipted in author_input_freeze.json. The supplied Craftax_Baselines_ppo_rnn.py was not read. The current controlling scope supersedes historical IDs in copied generic requirements. Original mechanics and code were authored here; no upstream implementation, map, benchmark or asset was copied or imported. Craftax README lines 22-55 supplies JAX spatial-game/gymnax API context; JaxMARL README lines 111-138 explains agent-keyed parallel interfaces and __all__; Pgx README lines 54-101 supplies native state, legal-mask and explicit-key API context; PureJaxRL README lines 35-37 identifies single-file algorithm reference philosophy, not a game library or replacement learner.

Validation status and limits

AUTHORED_NOT_EXECUTED. The supplied trusted AST whitelist and manifest/scene data validators were run without importing the candidate, JAX or NumPy. JSON/source hashes, exact file names, fixed declarations, literal action ranges and the single-component paired difference were checked as data only. Requested author identity is gpt-6-astra/max; provider-measured identity, effort, tokens, elapsed model compute and cost remain unknown. Source reasoning and static acceptance do not establish runtime compatibility, numerical parity, solvability, successful play, mechanism magnitude, learning, non-isomorphism or transfer. No training, seed search, rollout, rendering, network, external model call, heldout or outcome evidence was used. Variants in timing, topology and role resources are prospective ideas only and would require separate authorization and source identity.

Literal source evidence

- candidate.py:14: y = ((7 - x) * ends[0] + (x - 3) * ends[1] + 2) // 4 — Endpoint geometry gives actual beam cells.
- candidate.py:156: support = (counts >= 1) | ~jnp.any(n['cargo_status'] == 2) | n['braced'] — Other roles' weights or brace change endpoint feasibility.
- candidate.py:159: collision = jnp.any(jnp.all(cells[:, None, :] == n['agent_pos'][None, :, :], axis=2)) — Moving beam occupancy can be blocked by a worker.
- candidate.py:165: change = jnp.where(moved & (strain > 0), strain, -1) - 2 * n['braced'].astype(jnp.int32) — Tension persists and is reduced by bracing.
- candidate.py:169: drop = loaded & (tilt >= 2) & ~n['braced'] — Tilt changes physical cargo status.
- candidate.py:176: n['success'] = jnp.all(n['cargo_status'] == 3) — All three shipments are the natural task.
- candidate.py:187: n, reward = jax.lax.cond(state['done'], lambda _: (state, jnp.zeros(3, jnp.float32)), lambda _: self.advance(state, actions), operand=None) — Already-terminal calls are absorbing.

Static method locations: beam_cells: lines 12-15; sync: lines 17-28; reset: lines 30-59; walk_targets: lines 61-67; legal_actions: lines 69-87; observe: lines 89-96; advance: lines 98-184; raw_step: lines 186-189.
