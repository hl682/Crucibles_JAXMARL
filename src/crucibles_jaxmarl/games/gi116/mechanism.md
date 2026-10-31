# Bridge builders with unequal tool kits

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00116`, source attempt 0, source split train, final test unused.

## Spatial rules and cross-agent dependence

The 9 by 5 archipelago has a western island, a one-column middle island and a far bank, with four fixed bridge sockets across two channels. Northern and southern socket rows offer different construction routes. Four named structural parts exist: three on the west island and one reset-randomized between the two ends of the middle island. A part is always on ground, held by one crew member or installed at one socket. The same identity and real position are retained through pickup, handoff, installation, recovery and collapse.

The three unequal tool kits are fixed. The stabilizer cannot carry material and alone can brace a nearby socket. The builder carries one part and alone can install or retrieve a span. The porter carries up to two parts and can offer one to an adjacent agent that simultaneously chooses receive. Carrying capacity is checked on actual ownership; a give command without consent does not transfer. There is no abstract credit balance that substitutes for the parts.

Crossing mass equals one plus carried part count. A span with no live support carries total mass one; live support raises capacity to two. Therefore a builder carrying a part or a porter carrying one part needs support, and a porter carrying two parts cannot enter any span. Land allows co-location, while bridge movements are resolved in rotating order against actual mass already at the destination. One actor's entry can thus consume the remaining capacity available to another.

A brace renews the first nearby socket in fixed socket order to ten ticks; it is not a directional action or an arbitrary site selector. A newly installed span starts with lifetime twelve before that transition's maintenance. Unsupported life immediately loses one per transition. While supported, life does not decay and is at least six. Expiration or overload collapses a span. Its part and every falling crew member, including held parts, return to that socket's western bank. This explicit recovery rule conserves material while imposing time and fall penalties; it is not a realistic structural-engineering model. The far bank must contain all three crew simultaneously for natural completion.

Every role receives two units per first crew arrival, five for joint completion, minus one half per falling person and 0.01 per tick. First-arrival flags prevent farming the arrival reward by returning. A returning agent can still provide useful support or transport before joint completion. The map, part owners, load, tool role, span life, support and capacity are all public.

## Partner-dependent strategies and limits

One construction plan supports a northern span while the porter carries a second part across and consents to hand it to the builder for the second channel. Another plan can use the middle-island part after the first crossing, retrieve an unused span, or build along the southern sockets. Maintaining a span at the expense of moving the stabilizer, reducing load through a handoff, and sequencing bridge entry all depend on the other roles. Shared rewards do not eliminate a non-cooperating porter's withholding or a stabilizer's departure.

The main seed-11601 witness proposes a supported first crossing, physical porter-to-builder handoff, second-span assembly and staged completion. It remains unexecuted. Fixed tie rules, support expiration, capacity arbitration or an implementation defect may prevent the intended crossing. The paired test has an identical pickup/build prefix, keeps the loaded porter's east action fixed and varies only the stabilizer's brace. It is intended to produce a permitted mass-two entry in A and a blocked entry in B. Record actual load, support, capacity and positions rather than inferring interaction from shared reward.

No gain in the short suffix is required: neither branch may yet reach the goal, and extra support can consume time without a corresponding later benefit. A wrong prefix or depleted lifetime can erase the intended contrast. Collapse and negative returns remain valid outcomes.

Changing toolkit allocation, support duration, socket placement, material starting positions or the mass rule could alter which ordering or route works. Except for the declared middle-island part reset variation, these are unimplemented future hypotheses. Static contracts do not establish multiple successful routes or adaptive transfer.


## Literal source evidence

- Material ownership determines installed spans: `candidate.py:19` contains `n['built'] = jnp.any(s['part_owner'][:, None] == (3 + jnp.arange(4))[None, :], axis=0)`.
- Actual installed span tile rendering: `candidate.py:20` contains `n['render_tiles'] = s['tiles'].at[s['bridge_pos'][:, 1], s['bridge_pos'][:, 0]].set(jnp.where(n['built'], 4, 2))`.
- Total crossing mass: `candidate.py:79` contains `mass = jnp.sum(jnp.where(occupants, 1 + s['load'], 0)) + 1 + s['load'][i]`.
- Physical movement admissibility: `candidate.py:81` contains `allowed = (tile == 0) | (tile == 3) | (jnp.any(at_bridge) & s['built'][bi] & (mass <= s['capacity'][bi]))`.
- Unequal carrying kits: `candidate.py:84` contains `carry_limit = jnp.array([0, 1, 2], jnp.int32)`.
- Two-sided handoff consent: `candidate.py:102` contains `receivers = (a == 9) & (jnp.arange(3) != i) & (n['load'] < carry_limit) & (`.
- Real part transfer: `candidate.py:108` contains `give = (a[i] == 8) & jnp.any(held) & jnp.any(receivers)`.
- Support renewal: `candidate.py:114` contains `brace = (a[0] == 7) & jnp.any(brace_near)`.
- Builder-only material installation: `candidate.py:121` contains `build = (a[1] == 6) & jnp.any(empty_site) & jnp.any(held)`.
- Expiry or overload collapse: `candidate.py:136` contains `collapsed = n['built'] & ((n['life'] <= 0) | (mass > n['capacity']))`.
- Conserved span recovery: `candidate.py:143` contains `lost_span = deployed & collapsed[part_site]`.
- Current-position natural completion: `candidate.py:157` contains `n['terminated'] = jnp.all(n['agent_pos'][:, 0] >= 6)`.
- Absorbing terminal projection: `candidate.py:162` contains `n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}`.

Method ranges: refresh at candidate.py:12-25; reset at candidate.py:27-50; observe at candidate.py:52-61; legal_actions at candidate.py:63-64; raw_step at candidate.py:66-165.

## Literal interface and execution boundary

The author supplied three-agent flat float32 observations, int32 joint actions, fixed-shape dictionary state and homogeneous discrete command attempts. The exact dimensions, every leaf, every observation column, reward bounds, actions and spatial bindings are in manifest.json. Unmet command prerequisites produce no effect; out-of-range integers are converted to wait. Each live raw transition advances t once. Natural completion takes precedence over an equal-tick timeout. Already-terminal raw calls preserve every state leaf, observation and clock, return zero reward and preserve actual terminal causes. There is no raw autoreset.

No candidate was imported or called. No JAX/NumPy import, compilation, numerical game call, training, model API, installation, SSH or remote work occurred during authoring. Standard-library AST and supplied data validators may establish only their literal static checks. They do not establish interface execution, numerical parity, conservation over actual traces, strategic richness, solvability, policy learning or transfer. AUTHORED_NOT_EXECUTED is the package status. The JaxMARL-compatible public dictionary adapter, real replay, RGB animation, interactive playback and role-view audit belong to the separately authorized implementation/validation stage and are not delivered or claimed here.

The playthrough JSON uses literal seeds and joint integer vectors. A trusted runner should use jax.random.PRNGKey(seed) for reset, then jax.random.fold_in(reset_key, step_index+1) for raw-step keys; these are prospective instructions and were not executed. All four kernels use step keys only as interface inputs, so their transition stochasticity is reset-derived. Scripts stop at joint done. They are author hypotheses, not successful trajectories or acceptance gates. Paired tests freeze the shared prefix and the non-intervened actions; they must retain zero, negative, mixed, failure and indeterminate outcomes.

The source uses int32 for small integer categories, owners and counters and bool for masks/terminal causes. The uint32 PRNG key remains trusted trajectory provenance rather than a duplicated state leaf. Render data is declarative. Map and entity positions bind real state. An omniscient debug state, where provided later, must be labelled separately and never substituted for a role observation.

## Official ecosystem inspiration and source use

Only this batch's physically supplied official README excerpts were read for broad ecosystem context: JaxMARL separates MARL environments and algorithms and exposes multi-agent interfaces; Pgx exemplifies JAX-native board states, legal action masks and explicit step randomness; Craftax illustrates persistent spatial game state, inventories and separate reset/render interfaces. These are conceptual influences. No upstream source code, environment layout, asset, baseline or algorithm was copied into this original kernel.

The supplied JaxMARL README identifies Apache-2.0, the Pgx README Apache-2.0, and the Craftax README MIT. Those are descriptions of upstream snapshots, not a new license grant for this package; an original-package release license remains the coordinator's decision. PureJaxRL describes algorithm implementations rather than an environment library. Craftax_Baselines_ppo_rnn.py was supplied and hash-accounted for but was not opened or reused. No baseline replaces the frozen experimental learner. Exact source snapshot URLs and hashes are in generation_record.json; no batch-external source or candidate feedback was consulted.

Generating or replaying these rules would not establish large-scale meta-RL adaptation. Rule/role/map variations discussed above are future research hypotheses requiring their own frozen source, cost and information controls, and genuine train/develop/confirm/final-transfer evidence. They are not implemented variants or measured generalization.
