# Orchard pollination and shared habitat

AUTHORED_NOT_EXECUTED. This document describes source rules and author hypotheses; it is not a validation, animation, successful play, learning or transfer receipt.

## Task and state transition order

A public 9 by 9 orchard contains two owned plots per grower, upper and lower rows at y=2 and y=6. The shared tank at (4,4) starts with eighteen water; habitats at (2,4) and (6,4) hold two each. Growers start at those three central locations with one seed each and no carried water. Three fixed pollinators start with three visit credits each. There are no births, rain, hidden water sources, seed creation or random yields.

After simultaneous walking, cyclic grower transactions handle sowing, water withdrawal, irrigation, harvest/uproot, habitat deposits and adjacent gifts. Water may be deposited in empty or neighboring plots and persists across crop clearing. Planting consumes a circulating seed. A wet seedling increments age; at age two it consumes one soil water, becomes a flower and resets age. Pollinators then choose nearest eligible flowering plots, excluding their own last plot, or seek a habitat when no target/credit is available. They move one Manhattan cell and can visit on arrival. A visit spends one credit, updates last-plot/pollen-owner memory and marks cross-pollination if the previous owner differs. Habitat arrival with fewer than three credits consumes one reserve water, restores three credits and clears last_plot. It does not erase pollen. Bees may share cells and fly across the walkable interior independently of human occupancy.

After bee motion, all flowers age, including newly opened flowers. At age at least four, a flower with soil water consumes one and becomes ripe. Its fruit becomes one, or three if cross-pollinated. A flower reaching age six without water wilts instead. A grower may harvest only its own ripe plot, earning its actual fruit and receiving the planted seed back; uprooting any own nonempty plot returns the seed without reward. The sixth accepted harvest closes the season; remaining ripe crops can be left unharvested at that natural terminal. Rewards contain no planting, habitat, gifting or pollination bonus.

## Actions and inactive-role semantics

- 0: Wait.
- 1: Move east.
- 2: Move west.
- 3: Move south.
- 4: Move north.
- 5: Plant one seed in nearest empty own plot within distance one.
- 6: Draw one shared tank water within distance one, if carrying below two.
- 7: Irrigate nearest non-full plot within distance one with one carried water; any owner allowed.
- 8: Harvest nearest eligible ripe own plot within distance one, returning its seed.
- 9: Add one carried water to nearest habitat within distance one, up to reserve four.
- 10: Give one seed to previous owner id modulo three if adjacent.
- 11: Give one seed to next owner id modulo three if adjacent.
- 12: Uproot a nonempty own plot within distance one; return its seed without fruit reward.

All thirteen intent actions are legal before joint termination; missing range, ownership, capacity or inventory prerequisites cause deterministic no-ops. Invalid indices map to 0; terminal calls allow only 0. Simultaneous movement blocks walls, input occupied cells and duplicate destinations; swaps are blocked. Resource transactions use cyclic grower priority starting at t mod 3, with nearest admissible target and lowest id ties. Only own plots can be planted, harvested or uprooted; anyone may water another plot or maintain habitat. Pollinators are processed in id order for finite habitat-water withdrawals. A tick completes its physics even when its sixth harvest has been accepted.

## Information and rendering

The world is fully observable: all positions, owners, seed/water balances, growth clocks, pollination/fruit state, pollinator memory/targets, habitat reserves and tank balance are public native fields. Own role is an explicit one-hot. Every scene entity uses true public visibility; there is no private renderer-only plant or bee state.

All entities and event participants are public. Per-step planting, irrigation, visit, pollination, ripening and harvesting flags are distinct from persistent stages, fruit levels, soil water and pollination flags. Habitat maintenance events denote grower deposits; bee refill consumption is visible as actual reserve changes, not falsely labelled new maintenance.

Plot entities are permanent real bed locations even when empty. Their stage, age, soil moisture, pollination and fruit attributes bind the growing state, and their owner is explicit. Pollinator position, remaining visits, carried pollen owner, last plot and selected target bind actual motion and cross-owner yield effects. Habitat and tank bars represent finite stored water. Irrigation events are actual deposits, and plant growth/visits/harvests are per-step events. The scene must not animate an unexecuted script as if it were a recorded trajectory.

## Native observation columns

- [0, 3): Own grower one-hot identity (self).
- [3, 84): Public 9 by 9 tile map in yx row-major order divided by 4 (public).
- [84, 90): Three grower xy positions divided by 8 (public).
- [90, 93): Unplanted seed counts divided by 3 (public).
- [93, 96): Carried water divided by 2 (public).
- [96, 99): Own harvest counts divided by 6 (public).
- [99, 111): Six plot xy positions divided by 8 (public).
- [111, 117): Plot owner ids divided by 2 (public).
- [117, 123): Plot stages divided by 4 (public).
- [123, 129): Plot stage ages divided by 6 (public).
- [129, 135): Soil water divided by 2 (public).
- [135, 141): Cross-owner pollination flags (public).
- [141, 147): Fruit values divided by 3 (public).
- [147, 153): Three pollinator xy positions divided by 8 (public).
- [153, 156): Pollinator remaining visits divided by 3 (public).
- [156, 159): Pollen owner ids divided by 2; -0.5 means none (public).
- [159, 162): Last plot ids divided by 5; negative means none (public).
- [162, 165): Target ids divided by 7; 0..5 plots and 6..7 habitats (public).
- [165, 169): Two habitat xy positions divided by 8 (public).
- [169, 171): Habitat water reserves divided by 4 (public).
- [171, 173): Tank xy position divided by 8 (public).
- [173, 174): Tank water divided by 18 (public).
- [174, 177): Clock/96, total harvest count/6, permanently consumed water/22 (public).

## Invariants and conservation

Let S=sum(seeds)+count(stage!=0). Reset gives S=3; planting transfers one from inventory to a nonempty plot, clearing returns exactly one, and gifts subtract/add the same unit. Thus S=3 without clipping. Let W=tank_water+sum(water)+sum(soil_water)+sum(habitat_water)+water_spent. Reset W=18+0+0+4+0=22. Drawing, irrigation and maintenance are exact transfers; blooming, ripening and bee refills subtract one from storage and add one to water_spent. Hence W=22. Capacity and positive-balance checks keep stores nonnegative and bounded. Three bee slots remain active; visit credits stay 0..3. Growth clocks stay within 0..6. The harvest admission check caps the season at six, so normalized counts stay within bounds. These are source-level conservation arguments; no simulated trajectory or invariant measurement has been produced.

## Partner-dependent strategies and bounded variation

Staggering or synchronizing flowering changes the shared bees travel and carried pollen. A nearby new flower can divert a bee away from a partner, but a sequence of different owners can improve later yields for several growers. A grower can irrigate a neighbor to retain useful flowers or put water into habitat so shared bees keep moving, sacrificing water that could mature its own crop. Tank depletion and six season slots favor early claims, while larger cross-pollinated fruit can favor waiting. Seed gifts permit another owner to plant two plots at once while the donor gives up an immediate planting option. Upper/lower planting choices change distances and pollen routes. Variations in plot geometry, flowering clocks, initial water, habitat capacities or bee tie-breaking would change these tradeoffs; no adaptation, optimality or transfer result is claimed.

## Terminal, RNG and numeric representation contract

A nonterminal call advances scalar int32 t once. Natural termination takes priority at H: truncated=(t>=H)&~terminated; done=terminated|truncated. An already-done input returns the same entire state and observation, zero float32[A] rewards, all dones true and the preserved causes. No raw reset occurs. All roles continue to receive the joint done flag; an arrived or stunned role is not independently terminated.
Reset is deterministic and does not consume its supplied key. Each live step draws phases=jax.random.randint(step_key,(3,),0,6,dtype=int32). Each phase breaks equal-distance flower ties by cyclic plot rank and chooses x-first versus y-first movement when both are possible. No other random source or stored key is used. Paired interventions require the same literal per-step keys so differences are not caused by reseeding.
The author declares seed 121 with reset_key=jax.random.PRNGKey(seed) and raw_step_key=jax.random.fold_in(reset_key, step_index+1), step_index starting at zero. The same keys are required for both intervention branches, including suffix steps. This schedule has not been executed.
All signed integer state uses int32, boolean state uses bool, and the observation/reward boundary is float32. No smaller signed integer design field or stored reset key is required. Probability draws use only the supplied explicit key. Token counts, compute cost and elapsed model-compute cost are unknown.

## Literal source evidence

- Scarce water transferred into plots: candidate.py:105; literal source: `s['soil_water'] = s['soil_water'].at[watered].add(irrigate.astype(jnp.int32))`.
- Seeds actually leave and enter inventories: candidate.py:135; literal source: `s['seeds'] = s['seeds'].at[receiver].add(give.astype(jnp.int32))`.
- Flowering changes shared destination set: candidate.py:148; literal source: `candidates = (s['stage'] == 2) & (jnp.arange(6) != s['bee_last_plot'][b])`.
- Cross-owner pollination causal test: candidate.py:167; literal source: `cross = visit & (s['bee_pollen'][b] >= 0) & (s['bee_pollen'][b] != s['plot_owner'][flower])`.
- Habitat capacity consumes conserved water: candidate.py:174; literal source: `s['habitat_water'] = s['habitat_water'].at[habitat].add(-recharge.astype(jnp.int32))`.
- Pollination changes actual fruit amount: candidate.py:184; literal source: `s['fruit'] = jnp.where(ripen, 1 + 2 * s['pollinated'].astype(jnp.int32), s['fruit'])`.
- Reward comes only from harvest: candidate.py:111; literal source: `rewards = rewards.at[i].add(harvest.astype(jnp.float32) * s['fruit'][picked].astype(jnp.float32))`.
- Finite season admission: candidate.py:110; literal source: `harvest = (a[i] == 8) & jnp.any(ripe) & (s['harvest_count'] < 6)`.

## Method line map

- reset: candidate.py:12-52.
- observe: candidate.py:54-63.
- legal_actions: candidate.py:65-66.
- advance: candidate.py:68-196.
- absorb: candidate.py:198-200.
- raw_step: candidate.py:202-204.

## Reference attribution

- JaxMARL_README.md: Reviewed the copied parallel agent-keyed API description and reset/step call sites. The native author kernel remains a distinct interface requiring a trusted adapter. Supplied SHA-256 6649d04f5ed05d620e78e8416a00d4db841cf2633ae60fdb3c9729e037a71c15; origin https://github.com/bold-lab-ai/JaxMARL/blob/main/README.md.
- Pgx_README.md: Reviewed the copied legal-action-mask and explicit-key board-game API discussion. No board rule, implementation, layout or evaluation was copied. Supplied SHA-256 153f44999e55d5df3e51e13dee6ef418cc9704ddfa26206f1dc5789b9cbbf8df; origin https://github.com/sotetsuk/pgx/blob/main/README.md.
- PureJaxRL_README.md: Reviewed the copied description of a JAX-native PPO training pipeline. This is algorithm integration background, not a game library and not a learner replacement. Supplied SHA-256 d5f40182c924a42cec820931e61a7d5070e48c4c8413c7d16455abc6ea0ef6b0; origin https://github.com/luchris429/purejaxrl/blob/main/README.md.
- Craftax_Baselines_ppo_rnn.py: Read-only search of reset/step/scan call sites in the supplied recurrent PPO integration code. No upstream source was copied or executed. Supplied SHA-256 5f93e6a7c3a1e0c551831d75be311735ba495dfdecae51787a1c03a27b1cd011; origin https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py.
- Craftax_README.md: Reviewed copied JAX spatial game, gymnax-interface and optional auto-reset descriptions. Original mechanics are authored here; raw terminal handling follows the assigned contract. Supplied SHA-256 7f9757a2047143cf183f7e2e84851a740f386af80dbee050c89b316d66a3d3ff; origin https://github.com/MichaelTMatthews/Craftax/blob/main/README.md.

No upstream game code is copied. JaxMARL API ideas, Pgx board APIs, Craftax game/environment ideas and PureJaxRL/Craftax PPO algorithm integration are distinct sources of conceptual context. No literature claim is treated as evidence that this game works.

## Acceptance limits

Only the supplied stdlib AST whitelist and data-only manifest/scene validators are used. Static checks do not test runtime array shapes, eager/JIT/vmap parity, terminal behavior, conservation over rollouts, paired causal outcomes, solvability, rich strategy, trainability, animation, optimality, non-isomorphism or meta-learning. Source authoring alone is not community delivery. The coordinator must retain independent dispatch provenance and perform any later numeric work only under its separately authorized, hash-bound execution boundary.
All scripts and paired branches are UNEXECUTED_AUTHOR_HYPOTHESIS. Negative, failed and indeterminate later outcomes must remain visible; no performance threshold or scientific acceptance is introduced.
