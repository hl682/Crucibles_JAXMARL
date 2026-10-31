# Lantern rescue and moving shadows

Status: **AUTHORED_NOT_EXECUTED**. Candidate `game-v3-00114`, source attempt 0, source split train, final test unused.

## Spatial rules and cross-agent dependence

This is a 9 by 7 cooperative rescue map with three roles, two evacuees, two portable lanterns, two relay posts, one finite battery depot and two moving shadows. The eastern evacuees must physically reach the shelter at (1,3). A rescuer carrying an evacuee cannot also carry a lantern. Role 0 may handle lanterns but cannot lift evacuees; role 1 may lift evacuees but cannot pick up lanterns; role 2 may do either. Agents can share cells, so the essential dependency is illumination and inventory rather than incidental body blocking.

A burning lantern or charged relay illuminates Manhattan radius three subject to the implemented 17-sample rounded-cell ray. This is the exact discrete sight rule, not a continuous ray-tracing claim. Loaded-carrier movement checks illumination in the old state. Consequently an escort moving the light changes the carrier's next feasible destination, and moving a light away can strand a carrier. A lamp may also be grounded and left burning. Lighting and position are not a scalar reward ledger: they enter movement permission, visual access and damage.

A live lamp burns one charge unit per tick, including while grounded. Near the depot, a holder transfers up to eight of its remaining sixteen stock units into a lamp, capped at sixty-four. A holder may instead spend four lamp units to charge a nearby relay; the next displayed state has twelve relay ticks. These actions trade portable coverage against a stationary safe route and finite fuel. A zero-fuel switched-on lamp does not illuminate. Fuel is decremented before damage, so entering on the last unit is a real timing risk.

Two shadow positions advance along the six-cell eastern loop every three ticks. The random initial phase belongs to reset state and is not a policy field. An agent co-located with a shadow after the phase/fuel update loses one health if the cell is unlit. At zero health the role becomes inactive and drops its object. Unrescued evacuees remain recoverable; all roles disabled is natural failure. Sheltered evacuees are removed from the active entity set and counted once. Each live transition pays all roles five per new rescue, minus one quarter per hit and a one-hundredth time cost.

## Partner-dependent strategies and limits

A moving escort can accompany one carrier while the runner establishes the southern relay or returns with spare fuel. Alternatively, grounding a lantern and charging a post can free the runner's single bulky-item slot for the second evacuee. Splitting the lamps may expose more cells but leaves less redundant coverage. Waiting for a partner's light, spending scarce stock and choosing which evacuee to retrieve first all depend on the partner's actions. Shared rewards alone do not enforce cooperation, and an inactive or unhelpful escort can prevent useful carrier progress.

The role view is deliberately incomplete: each living observer sees radius-one cells or illuminated cells within radius four, always through its own sight rays. Distant lantern charge, evacuee carrier and shadow positions are not supplied. A shadow sprite also requires its cell to be illuminated. Current visibility has no automatic memory. Time and rescue totals are public; private shadow phase is omitted. Dynamic render tiles mark illuminated floor using the same state illumination, and the map visibility binding prevents a role renderer from revealing distant light or geometry.

Changing relay placement, wall apertures, shadow timing or the role permitted to lift either object would change which escorts and routes are useful. Those are unimplemented variation proposals. Neither this source nor an eventual successful rescue would establish transfer to them.

## Prospective witness and causal comparison

The main seed-11401 script aims to escort the northern evacuee toward the shelter while the runner carries the spare lamp to the southern beacon. It is a finite action witness hypothesis, not a claim that either event occurred or that both evacuees were rescued. The paired script freezes the same pickup prefix and the carrier's actions, then changes only the lighter's toggle action. Its intended contrast is a later carrier movement succeeding under continued light and failing after darkness. Record actual carrier coordinates, light grids, visibility and damage. The runner's light, occlusion, accidental prerequisite failures or an implementation defect could make an intended contrast zero or indeterminate. No reward improvement or survival difference is required.


## Literal source evidence

- Discrete line-of-sight sampling: `candidate.py:15` contains `ray = jnp.round(sources[:, None, None, :] + delta *`.
- Light depends on actual source charge and switch: `candidate.py:30` contains `lit = jnp.any(lamp_rays & lamp_near & (s['lamp_on'] & (s['lamp_charge'] > 0))[:, None], axis=0)`.
- Renderer floor illumination binding: `candidate.py:48` contains `n['render_tiles'] = jnp.where((s['tiles'] == 0) & n['illumination'], 4, s['tiles']).astype(jnp.int32)`.
- Role-local map visibility: `candidate.py:51` contains `visible = self.sight(n['agent_pos'], cells, n['tiles']) & (`.
- Loaded carrier depends on pre-step light: `candidate.py:131` contains `can_walk = (s['tiles'][target[:, 1], target[:, 0]] != 1) & ((s['agent_evac'] < 0) |`.
- Complementary lifting capability: `candidate.py:151` contains `pick_evac = (a[i] == 6) & (i != 0) & (held_evac < 0) & (held_lamp < 0) & jnp.any(free_evac)`.
- Finite shared battery transfer: `candidate.py:158` contains `amount = jnp.where((a[i] == 8) & (held_lamp >= 0) & near_depot,`.
- Relay battery expenditure: `candidate.py:164` contains `pulse = (a[i] == 9) & (held_lamp >= 0) & n['lamp_on'][safe_li] & (n['lamp_charge'][safe_li] >= 4) & jnp.any(near_beacon)`.
- Dark-shadow damage: `candidate.py:175` contains `hit = touched & ~n['illumination'][n['agent_pos'][:, 1], n['agent_pos'][:, 0]] & n['agent_active']`.
- Natural success/failure: `candidate.py:185` contains `n['terminated'] = jnp.all(n['evac_rescued']) | ~jnp.any(n['agent_active'])`.
- Absorbing terminal projection: `candidate.py:190` contains `n = {k: jnp.where(s['done'], s[k], n[k]) for k in s}`.

Method ranges: sight at candidate.py:12-20; lighting at candidate.py:22-33; refresh at candidate.py:35-62; reset at candidate.py:64-93; observe at candidate.py:95-118; legal_actions at candidate.py:120-122; raw_step at candidate.py:124-193.

## Literal interface and execution boundary

The author supplied three-agent flat float32 observations, int32 joint actions, fixed-shape dictionary state and homogeneous discrete command attempts. The exact dimensions, every leaf, every observation column, reward bounds, actions and spatial bindings are in manifest.json. Unmet command prerequisites produce no effect; out-of-range integers are converted to wait. Each live raw transition advances t once. Natural completion takes precedence over an equal-tick timeout. Already-terminal raw calls preserve every state leaf, observation and clock, return zero reward and preserve actual terminal causes. There is no raw autoreset.

No candidate was imported or called. No JAX/NumPy import, compilation, numerical game call, training, model API, installation, SSH or remote work occurred during authoring. Standard-library AST and supplied data validators may establish only their literal static checks. They do not establish interface execution, numerical parity, conservation over actual traces, strategic richness, solvability, policy learning or transfer. AUTHORED_NOT_EXECUTED is the package status. The JaxMARL-compatible public dictionary adapter, real replay, RGB animation, interactive playback and role-view audit belong to the separately authorized implementation/validation stage and are not delivered or claimed here.

The playthrough JSON uses literal seeds and joint integer vectors. A trusted runner should use jax.random.PRNGKey(seed) for reset, then jax.random.fold_in(reset_key, step_index+1) for raw-step keys; these are prospective instructions and were not executed. All four kernels use step keys only as interface inputs, so their transition stochasticity is reset-derived. Scripts stop at joint done. They are author hypotheses, not successful trajectories or acceptance gates. Paired tests freeze the shared prefix and the non-intervened actions; they must retain zero, negative, mixed, failure and indeterminate outcomes.

The source uses int32 for small integer categories, owners and counters and bool for masks/terminal causes. The uint32 PRNG key remains trusted trajectory provenance rather than a duplicated state leaf. Render data is declarative. Map and entity positions bind real state. An omniscient debug state, where provided later, must be labelled separately and never substituted for a role observation.

## Official ecosystem inspiration and source use

Only this batch's physically supplied official README excerpts were read for broad ecosystem context: JaxMARL separates MARL environments and algorithms and exposes multi-agent interfaces; Pgx exemplifies JAX-native board states, legal action masks and explicit step randomness; Craftax illustrates persistent spatial game state, inventories and separate reset/render interfaces. These are conceptual influences. No upstream source code, environment layout, asset, baseline or algorithm was copied into this original kernel.

The supplied JaxMARL README identifies Apache-2.0, the Pgx README Apache-2.0, and the Craftax README MIT. Those are descriptions of upstream snapshots, not a new license grant for this package; an original-package release license remains the coordinator's decision. PureJaxRL describes algorithm implementations rather than an environment library. Craftax_Baselines_ppo_rnn.py was supplied and hash-accounted for but was not opened or reused. No baseline replaces the frozen experimental learner. Exact source snapshot URLs and hashes are in generation_record.json; no batch-external source or candidate feedback was consulted.

Generating or replaying these rules would not establish large-scale meta-RL adaptation. Rule/role/map variations discussed above are future research hypotheses requiring their own frozen source, cost and information controls, and genuine train/develop/confirm/final-transfer evidence. They are not implemented variants or measured generalization.
