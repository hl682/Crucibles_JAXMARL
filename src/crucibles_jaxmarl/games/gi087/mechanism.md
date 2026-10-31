# Spillway Orchard

**Status: AUTHORED_NOT_EXECUTED.**

A partially observed renewable-commons game with immediate private harvest/sale incentives and delayed downstream water and pollution costs. Agents may move upstream, invest in another bed, monopolize it temporarily, or preserve it for later shared provisioning.

## Rules and real cross-agent dependencies

Beds at [2,4], [4,4], [6,4], [8,4] are connected in increasing x order. HARVEST requires a bag slot, positive stock and water, and either no active claim or a claim owned by that role. It removes one stock and one water, adds two silt (pollution), and puts one fruit in the harvester's bag. This immediately reduces resources available to other visitors and can withhold the next downstream water transfer.

At each turn, one unit of water flows out of each bed holding at least two units. Incoming water is the previous bed's outgoing water; at most one silt accompanies each outgoing unit. All reaches update simultaneously, so a new packet travels only one bed per turn. The last bed drains off-map. Headwater receives one natural unit on even turns and two extra units per paid RELEASE near [1,4]. Local water is capped at three; silt at six. The caps represent spill/settling capacity rather than conserved unlimited reservoirs.

CLEAN consumes one carried fruit and removes up to two local silt. SOW consumes one and stores one seed, at most two. On turns divisible by four, one silt settles; then a bed with water, silt at most one, space below stock five, and either living stock or an invested seed regrows one fruit while consuming one water and one stored seed if present. An emptied, unseeded bed never revives on its own. Selling the last recoverable fruit can therefore destroy a future harvest for everyone.

STAKE spends one carried fruit to claim the nearest adjacent bed for six turns; another stake can replace the owner. Claims restrict harvesting only, not cleaning, sowing or later staking. Each bed admits only one non-release local work operation per turn. Contenders are ordered by (role - old_t) modulo four; there is no permanent lowest-ID privilege. A claim with one remaining turn still controls that turn's legal mask, then expires. The common market accepts as much of a bag as fits under the individual bank cap of eight. Completion needs each role to bank four, not just a large aggregate yield. This is a game objective, not an experimental reward gate.

## Kernel, information and rewards

A=4, O=73, U=11, H=96. Coordinates are integer [x,y], with y increasing downward. All state leaves are fixed-shape int32, float32 or bool arrays; all observations and rewards are float32.

All roles have identical action capabilities, but their initial upstream/downstream positions create unequal opportunities. To interact, a role uses the nearest bed within Manhattan distance one, breaking equal distance ties by lower bed index. Positions and bed condition are observed within distance three, including through the shallow channel; no wall-based line-of-sight is claimed. The fixed map, headwater and market positions are public. Bags are only local information. Bank amounts, total banked and the paid release bell are public announcements. Rendering visibility matches these observation rules.

Order is: sanitize legal intents; choose one work operation per bed; apply harvest/investment/banking and claim expiry; inject headwater; move water and silt one reach; settle/regrow on fourth turns; resolve actual agent movement; assess terminal state. Movement cannot enter another role's old cell or win a contested common destination. Agents can stand in shallow water and on bed coordinates.

Each live turn costs 0.01. Harvest earns its role 0.2. An accepted investment/release costs its role another 0.05 and one fruit. Each banked fruit earns its depositor 1. Joint completion pays each role 5; collapse costs each role 3. An unsuccessful contested work request spends no fruit and earns no harvest reward. These rules create immediate private returns, collective completion incentives, and both exploitable and investable externalities. The declared [-4,10] reward envelope and [0,1] observation bounds cover the kernel without implying expected performance.

The binding addendum is implemented: a live input advances the scalar int32 clock by one; natural completion takes priority over the horizon. Already-terminal raw calls preserve the complete state, observation, clock, terminal causes and final event flags, return zero rewards and all-true dones. They are not extra gameplay transitions. The only legal terminal action is WAIT. Autoreset belongs to an external wrapper. No individual is removed from the world before joint completion. Invalid or out-of-range actions become WAIT; blocked cardinal directions remain legal attempted movements and may produce no motion.

Each observation slice, entity position, active mask, visibility mask and displayed attribute is declared in manifest.json. scene.json repeats exactly the hash-bound map and declared channels. Its motion comes only from real state positions. A debug full-state view must remain distinct from role views. Public event channels are explicitly present in each role observation; local object/agent attributes are zero-masked together with a visible bit. A hidden entity is not a position hint.

## Literal source evidence

- Local visible state is masked by physical distance: candidate.py:L21 — `ns["bed_visible"] = ap <= 3`
- Claim changes another role legal harvest opportunities: candidate.py:L89 — `own = (s["claim_ttl"][target] == 0) | (s["claim_owner"][target] == jnp.arange(4))`
- Rotating contention priority: candidate.py:L109 — `ranks = jnp.mod(jnp.arange(4) - s["t"], 4)`
- Harvest externality and investment in the same condition variable: candidate.py:L128 — `pollution = jnp.clip(s["pollution"] + 2 * harvested - 2 * cleaned, 0, 6)`
- Water moves only one downstream reach: candidate.py:L140 — `incoming_water = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), flow[:-1]))`
- Pollution travels with actual outgoing water: candidate.py:L141 — `dirt_flow = jnp.minimum(pollution, 1) * flow`
- Persistent regenerative dependence on survival or investment: candidate.py:L148 — `grows = season & (pollution <= 1) & (water > 0) & (stock < 5) & ((stock > 0) | (seedlings > 0))`
- Shared task requires each participant: candidate.py:L164 — `success = jnp.all(banked >= 4)`
- True unrecoverable ecological failure: candidate.py:L165 — `collapse = (jnp.sum(stock) + jnp.sum(seedlings) + jnp.sum(bag) == 0) & ~success`

Method extents in this exact source:

- `direction`: candidate.py:L12-L14.
- `refresh`: candidate.py:L16-L22.
- `reset`: candidate.py:L24-L60.
- `observe`: candidate.py:L62-L80.
- `local_bed`: candidate.py:L82-L85.
- `legal_actions`: candidate.py:L87-L99.
- `advance`: candidate.py:L101-L191.
- `absorb`: candidate.py:L193-L196.
- `raw_step`: candidate.py:L198-L199.

## Fixed witnesses and intended intervention differences

Every script here is a local mechanism hypothesis, not a whole-task provisioning attempt. The harvest/clean pair shares the first action and all post-intervention waits, so later source-level differences can be traced to one investment choice. The claim script inspects a changed legal opportunity without deliberately issuing an illegal harvest. The release script spends a real carried fruit and observes the finite transport delay.

No scripted outcome is an execution receipt. A successful-looking path in prose is only a prediction from the rules. Scripts that exercise local mechanisms intentionally do not assert whole-task completion.

## Invariants, variation and limits

Declared nonnegativity covers stock, water, silt, planted seed, bags, banked fruit, claims, public totals and clock. Bounds from construction are stock<=5, water<=3, silt<=6, seedlings<=2, bag<=4, banked<=8 and claim_ttl<=6. Agent positions remain distinct. No assumption that every policy or seed can maintain the ecosystem is made. Changing natural inflow, reach order, pollution persistence or exclusion duration would change which partnerships and harvesting norms are useful; these are prospective variations, not measured variants. Partial observation is limited to a geometric neighborhood and public announcements; it is not a physically detailed fluid or optical simulator.

Integer sign-bearing fields (for example no-owner=-1) remain exact int32 values. There is no design-specified int8 distribution to adapt. The external uint32 PRNG key is not duplicated as a state leaf. Reset and transition are deterministic in this authored instance; accepted keys are unused.

## Provenance and reference use

This is an original authored kernel for initial window0, attempt00, in the train source stream. No prior candidate, project execution result, heldout layout, evaluation score, ranking or final test was read. No candidate or JAX was imported or run. The action lists are fixed hypotheses, not recorded trajectories. Interface/R0 checks, numerical parity, actual legality along the scripts, intervention replays, animation, training, solvability, optimality, strategic richness and meta-learning effects remain unestablished.

Only supplied official references were used. JaxMARL's README informed the intended simultaneous multi-agent adapter boundary (agent-keyed dictionaries and __all__ belong in the later integration layer). Pgx's README informed the separation of explicit transition keys, legal masks and raw terminal state. Craftax's README informed the distinction between a world kernel, state-based rendering and an external reset wrapper. PureJaxRL is an algorithm implementation reference, not a game library; its performance statements are not evidence about this game. Craftax_Baselines/ppo_rnn.py was read only at its environment/reset wrapper interfaces; no PPO/network code or benchmark mechanics were copied. No upstream implementation or map is incorporated. These references are attributed in generation_record.json with exact supplied file hashes and upstream URLs.

The copied community_delivery_requirements.json still lists IDs 82-85. The exact assigned design/request IDs 86-89 control this batch; the general interaction and presentation requirements are applied without rewriting that input. Actual dispatch identity, backend and usage evidence are retained by the coordinator, not invented here.
