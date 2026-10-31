# Canopy Commons

AUTHORED_NOT_EXECUTED. Current blind train-only initial attempt_00, no numerical or selection feedback. A=3; O=443; U=13; H=160.

Canopy Commons is a warm orchard village on an 11 by 7 board. Six physical plant identities comprise two young trees, one mature tree and three loose saplings. Six finite beds, two shade screens, one water cart, two regenerating springs, three private stalls and one community barn are native entities. Low hedges shape the walking routes; loose screens and the cart are real obstacles that gardeners can carry aside.

Each gardener has one slot for a sapling, tool or fruit. Saplings and tools are picked within one Manhattan cell and dropped on the carrier's cell; a tool cannot be dropped onto another loose tool. A carried sapling can be planted in the lowest-index adjacent empty bed. Its original owner persists, and it starts young with moisture one. Planted trees cannot be carried, and dead trees keep their bed, making planting capacity finite.

A deployed screen shades exactly the column one cell east or west of its position, within two rows vertically. Carrying it removes its footprint. Rotation changes its sign; at most one rotation per screen is applied per tick. The explicit shade_map drives both visible tile shading and the plant shaded flags used in ecology. Initially one screen shades the central young tree and a different owner's mature tree. Turning it changes their heat protection and light access together; protecting one patch is not a cost-free universal improvement.

After gardener transactions, t advances through morning, hot, late-day and night phases, four ticks each. A live planted tree consumes one moisture each third tick and one additional moisture each hot unshaded tick. Drought or unshaded hot exposure of a young tree increases stress; otherwise stress recovers by one, bounded 0..3. Stress three causes death. With daylight, no shade, positive remaining water and nonfatal stress, young and mature trees gain one growth tick. Four ticks turn young into mature or mature into ripe, resetting progress. Shade prevents young heat exposure and extra evaporation but also stops growth, including the mature neighbor's fruit development.

Harvest takes a real ripe portion into empty hands, keeps the plant's original owner and returns the tree to mature. A ripe portion can instead be lost if its tree dies. Cart watering transfers up to four stored units into an adjacent live plant, capped at moisture five. Springs hold up to six servings and regenerate one each third tick. The gardener carrying the only cart cannot simultaneously collect produce, move a screen or plant a sapling. Handoffs therefore require real drops and movement.

Fruit can go to its owner's private stall or the community barn. Either counts toward six total deliveries. Each private portion pays all roles 0.1 plus its original owner 1; a community portion pays every role 0.6. Every active step costs 0.01, with +2 each on natural completion and -1 each on natural total crop failure. No free resource is created by switching destinations or rehandling an object. Possible strategies include moving shade between phases, watering to buy growth time, specialized cart labor, relocating a spare screen and choosing between owner reward and shared reward. None has been run or accepted as successful.

## Native controls

- 0: Wait
- 1: North
- 2: East
- 3: South
- 4: West
- 5: Take adjacent loose sapling or put held sapling down
- 6: Take adjacent screen/cart or put held tool down
- 7: Rotate adjacent deployed shade screen east/west
- 8: Plant carried sapling in adjacent empty bed
- 9: Water adjacent live plant using carried water cart
- 10: Harvest adjacent ripe fruit with empty hands
- 11: Deliver carried fruit at matching owner stall or community barn
- 12: Fill carried water cart from adjacent nonempty spring

Humans and policies must call the same native reset/raw_step/observe/legal_actions interface. This source package does not implement or verify a public adapter, browser controls or a separate dynamics approximation. Invalid/masked actions become wait. Movement rejects old occupied agent cells, including simultaneous swaps, then uses rotating priority (role - old_t) mod 3 for shared targets. Contextual transactions are rechecked in rotating role order. Nearest-range interactions select the lowest eligible identity; later contenders may wait if a resource is taken or a bed is filled.

## Information and terminal semantics

Fully observable. All fields in obs_keys are public after the role one-hot, including actual base/derived terrain, explicit shade_map footprint, screen positions and orientations, carriers, planting identities/owners/beds/stages/water/stress/growth, spring stocks, fruit inventory and destination accounting. Derived shaded tiles and plant flags come from the same native footprint used by heat and growth transitions. No independent browser shade model or hidden crop planner exists.

Success when private plus community deliveries total at least six. Failure only when every plant identity is dead and no gardener carries fruit, with success taking priority if already achieved. Natural causes take priority over H. Each nonterminal raw call advances t once; an already terminal input preserves all state, observation, clock and terminal causes, returns zero rewards/all dones true, and never resets. Terminal masks permit only wait. All three roles remain present until joint termination.

Reset layout, phase transitions and all dynamics are deterministic. Required key arguments are accepted but unused.

## Invariants and evidence limits

There are exactly six plant and three tool identities, with at most one plant per bed and at most one total carried object/fruit per gardener. All gardener and entity coordinates stay within the board. Moisture stays 0..5, stress 0..3, growth 0..3, spring stock 0..6 and cart charge 0..4. Screen directions are +/-1 and cart direction zero. Total fruit_generated equals ripe fruit currently on trees + carried fruit + private deliveries + community deliveries + fruit_lost. Death only adds lost fruit for a previously ripe tree. Resource creation is explicit: spring regeneration and live-tree fruit growth. These are source-reasoned invariants, not runtime verification.

All state leaves are fixed-shape int32/bool arrays; signed identifiers are represented losslessly as int32 and the external reset key is not copied into state. Observations and rewards are float32. Declared observation bounds [-5,256] and reward bounds [-2, 6] are reasoned bounds, not measured extrema. Static AST/schema/shape bookkeeping cannot establish runtime compatibility, eager/JIT/vmap parity, solvability, meaningful effect size, successful play, learning or transfer.

## Literal source evidence

- Explicit shade footprint: candidate.py:32, literal 'footprint = (xx[None, :, :] == (s["tool_pos"][:, 0] + s["orientation"])[:, None, None]) & (jnp.abs(yy[None, :, :] - s["tool_pos"][:, 1, None, None]) <= 2)'.
- Shade drives native scene map: candidate.py:34, literal 's["tiles"] = jnp.where(s["shade_map"] & (s["base_tiles"] < 2), s["base_tiles"] + 2, s["base_tiles"])'.
- Single carrying slot: candidate.py:80, literal 'empty = (s["seed_carry"] < 0) & (s["tool_carry"] < 0) & (s["fruit_carry"] == 0)'.
- Real planting capacity: candidate.py:148, literal 'beds = (jnp.sum(jnp.abs(s["bed_pos"] - s["agent_pos"][i]), axis=1) <= 1) & (s["bed_occupant"] < 0)'.
- Actual peer watering: candidate.py:159, literal 'amount = jnp.where(act == 9, jnp.minimum(s["water_cart"][h], 5 - s["moisture"][j]), 0)'.
- Heat/water tradeoff: candidate.py:189, literal 'trouble = (s["moisture"] == 0) | (hot & ~s["shaded"] & (s["stage"] == 1))'.
- Delayed light-dependent growth: candidate.py:196, literal 'grow = live & (s["stage"] <= 2) & light & (s["moisture"] > 0) & ~s["wilted_now"]'.
- Individual/community return choice: candidate.py:211, literal 'reward = -0.01 + 0.1 * jnp.sum(s["private_now"]) + s["private_now"].astype(jnp.float32) + 0.6 * jnp.sum(s["community_now"]) + 2.0 * success - failure.astype(jnp.float32)'.
- Natural task completion: candidate.py:206, literal 'success = jnp.sum(s["private"] + s["community"]) >= 6'.
- Raw absorption: candidate.py:219, literal 'return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))'.

- update_geometry: candidate.py:20 through line 39.
- reset: candidate.py:41 through line 66.
- walking: candidate.py:68 through line 75.
- legal_actions: candidate.py:77 through line 102.
- observe: candidate.py:104 through line 106.
- transition: candidate.py:108 through line 212.
- absorb: candidate.py:214 through line 216.
- raw_step: candidate.py:218 through line 219.

## Blind scripts and attribution

Two original literal seed-zero scripts and one equal-length, single-actor-action paired hypothesis are in playthrough.json. They are unexecuted author hypotheses; timing follows the implemented mechanism and no seed search occurred. JaxMARL README supplied parallel agent-dictionary API context; Pgx supplied state/mask/terminal and external-key context; Craftax supplied JAX spatial-environment context; PureJaxRL supplied the distinction between algorithm implementations and game libraries. No upstream implementation, benchmark layout, learner or commercial asset was copied or imported. The supplied PPO baseline source was not read. Actual current input paths, bytes, hashes, displayed reference ranges and uses are in author_input_freeze.json. Historical IDs in generic copied requirements are background only. A separate explicit repair dispatch is required after this source is frozen.
