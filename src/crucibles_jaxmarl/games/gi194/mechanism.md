# Windlift Quarry

Status: AUTHORED_NOT_EXECUTED. Train-only initial source, attempt_00. No runtime, replay, solvability, adapter compatibility, mechanism success, learning or transfer acceptance is claimed.

## Native interface and human operation

A=3, O=186, U=12, H=160. Humans and future policies use exactly GeneratedEnv.reset, raw_step, observe and legal_actions. Presentation adapter and human controls are separate integration work, not implemented or verified here. No alternative browser dynamics exist.

The quarry is a 9 by 7 block world, with upper and lower terraces separated by a chasm. Edge columns x=0 and x=8 are continuous safe ramps. The shaft at x=4 is traversable only where the actual platform is located. The platform starts at (4,5), with one ore block and the courier. Controls stay on the lower terrace and remotely control either dock; they are not fictional coordinates on unrelated quantities.

There are exactly six ore blocks (two per owner), two mass-two counterweights and two repair kits. Each worker carries at most one block. Pickup uses only the same cell and lowest eligible block index. Loose stacks are allowed. Drop, installation, delivery and consumption change one block's status; nothing spawns or duplicates. A weight may be installed, lent through a drop, retrieved, transported or sold at its owner's depot. A kit is consumed once to restore at most three rope units, capped at eight.

Interaction order is old-state action masking, simultaneous movement with contention, rotating-order item transactions, load calculation, launch, one-cell lift motion, accounting and terminal evaluation. A launch requires both native control actions, a clear opposite dock, positive installed mass, absolute load-minus-counterweight at most two, and enough rope for one plus that imbalance. Load counts each aboard worker once and every aboard block once, including held cargo. A launch immediately moves one cell and reaches the other dock after four total moving steps. No new crank or brake is needed in transit. Arrival cells are reserved while travelling. A moving rider cannot step away, and carried/loose aboard blocks are translated with the platform.

This creates a direct peer dependency: the brake keeper's choice determines whether the courier travels this tick, and lending or selling a weight changes future feasible loads. Holding the brake, fetching ore, earning private weight-sale reward, walking a safe ramp and carrying a repair kit consume different worker time and scarce objects. A team can specialize around the lift, bypass it for some loads, or alternate those strategies. These are source-supported alternatives, not measured strategic richness.

Reward is -0.01 per role per active step, plus 0.2 to every role for each ore delivered, plus 1 to that ore's owner, plus 0.5 to a sold weight's owner, plus 2 to each role on natural success. No reward is paid merely for repeatedly moving the platform, picking up or dropping objects.

## Action legend

- 0: Wait
- 1: North
- 2: East
- 3: South
- 4: West
- 5: Pick up loose block on own cell
- 6: Drop held block on own cell
- 7: Crank: role 0 at crank, hands empty
- 8: Brake: role 1 at brake, hands empty
- 9: Socket held weight or retrieve a socketed weight at brake
- 10: Deliver ore or sell weight at its owner's depot
- 11: Patch rope using held kit within one cell of a control

## Information and boundary semantics

Fully observable. Every policy receives its role one-hot and all dynamic/game-relevant public state leaves listed by obs_keys, including the complete tile map, object status, carrier, owner, load mass, controls, platform, rope, depots and events. All active entities are visible to all roles; inactive consumed/delivered block coordinates remain in the public observation for accounting. Rendering uses the same state and cannot add information.

Invalid or masked actions become wait. Contention may make a legal action have no effect. Agent movement forbids entering another role old cell, including swaps, then resolves a shared destination by rotating priority (role - old_t) mod 3. All roles remain active until joint termination; a role completing its own deliveries can continue to help. Terminal masks expose only wait.

State leaves are fixed-shape int32/bool arrays. Observations/rewards are float32. Signed IDs use int32 losslessly, and the reset key is not copied into state. The reset layout and all transitions are deterministic; key arguments are accepted but unused.

Natural termination: All six ore blocks have been delivered. There is no early loss terminal: depleted or unavailable lift service still leaves the physical edge ramps. Nonterminal calls increment t once. Natural completion takes precedence over the cap. Already-terminal calls preserve state, observation, clock and actual terminal causes, return zero reward/all dones true and never reset. H is an episode cap, not a scientific acceptance gate.

## Invariants and limits

Agent positions are unique, in [0..8,0..6]. Capacity is exactly ten persistent block identities; all are either loose, held once, installed, delivered/sold or consumed. A role holds at most one. Only the two weight identities can have installed status; installed mass is their actual sum. Ore delivered counts sum to the number of ore identities in delivered state and never exceed six. Rope remains 0..8. Platform x remains 4, y remains 1..5; its target is 1 or 5. Each raw input already marked done is absorbing. These conservation and range statements are reasoned invariants, not executed test results.

Observation bounds [-5,256] and reward bounds [-1, 6] are declared source-level bounds, not measured extrema. Static shape/schema checks do not establish numerical correctness, eager/JIT/vmap parity, arbitrary-trajectory validity or successful play. Those require a separate frozen-source dispatch.

## Literal source evidence

- Physical terrain: candidate.py:32, literal 'tiles = tiles.at[3, 1:8].set(1)'.
- No disembarkation in transit: candidate.py:70, literal 'ok = ok & ~(travelling & rider[:, None])'.
- Installed counterweight accounting: candidate.py:27, literal 's["countermass"] = jnp.array([jnp.sum(jnp.where(status == -2, s["block_mass"], 0))], dtype=jnp.int32)'.
- Mutual control and load rule: candidate.py:153, literal 'launch = docked & (actions[0] == 7) & (actions[1] == 8) & (s["countermass"][0] > 0) & (imbalance <= 2) & (s["rope"][0] >= 1 + imbalance) & ~occupied'.
- Actual cargo transport: candidate.py:158, literal 's["block_pos"] = s["block_pos"] + aboard[:, None].astype(jnp.int32) * shift'.
- Owner-specific returns: candidate.py:169, literal 'reward = -0.01 + 0.2 * jnp.sum(s["delivered_now"]) + s["delivered_now"].astype(jnp.float32) + 0.5 * s["sale_now"].astype(jnp.float32) + 2.0 * success'.
- Natural terminal: candidate.py:164, literal 'success = jnp.sum(s["delivered"]) == 6'.
- Absorbing dispatch: candidate.py:177, literal 'return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))'.

- refresh: candidate.py:18 through line 28.
- reset: candidate.py:30 through line 56.
- movement: candidate.py:58 through line 71.
- legal_actions: candidate.py:73 through line 93.
- observe: candidate.py:95 through line 97.
- advance: candidate.py:99 through line 170.
- absorb: candidate.py:172 through line 174.
- raw_step: candidate.py:176 through line 177.

## Hypotheses and sources

playthrough.json contains original literal scripts and one paired hypothesis differing in exactly one actor-action component. Seed zero is literal and unsearched. Suffix timing follows the source mechanism; no branch has run and no script is a success receipt.

The supplied JaxMARL README informed parallel agent-dictionary API context; Pgx informed mask/terminal and external-key context; Craftax informed spatial mechanics and raw/autoreset separation; PureJaxRL is an algorithm reference, not a new game library. No upstream implementation, benchmark layout, learner, commercial asset or code was imported or copied. The supplied Craftax PPO source was not read. Public performance claims and incidentally displayed score rows were not used. All actual input paths, bytes, hashes and uses are frozen in author_input_freeze.json. Current 194-197 designs and family principles control this independent game; historical IDs in copied requirements do not assign work.
