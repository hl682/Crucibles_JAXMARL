# Icehouse Service

AUTHORED_NOT_EXECUTED. Current blind train-only initial attempt_00, no numerical or selection feedback. A=3; O=209; U=14; H=180.

Icehouse Service is an 11 by 7 kitchen with a cold room, hot stove and two service hatches. A wall at x=5 leaves only (5,3) as the service corridor. The cold room occupies x<=2,y<=2 and its only opening is the door at (3,1). Three tray identities and two portable ice boxes are native objects. Two trays begin with one raw hot portion and one raw dessert; the pantry holds two more of each type. The three cooks begin at hot prep, cold prep and the pantry.

Every cook has one carrying slot. Trays are picked from the same cell; processing trays cannot be lifted. Boxes are picked within one Manhattan cell. A loose ice box blocks worker movement even though a tray does not. A box may be put on the carrier's own cell and the carrier may then leave, so deliberately blocking the narrow corridor is a real native action. Putting down an item requires that cell have no other loose item.

Starting work leaves the held raw tray at its matching station. Hot work increases temperature by two, capped at eight, and completes in four ticks; a ready hot dish can also be returned for reheating. Cold work requires applied cooling and temperature at most one for three consecutive ticks; an insufficiently cold tick resets its progress. A powered cold room cools all trays inside. Each charged ice box, loose or carried, cools trays within one Manhattan cell. Cooling subtracts up to two temperature; hot processing takes priority. Otherwise, each third tick moves a tray temperature one unit toward ambient three. Ready hot food needs at least five at its hatch, and dessert at most one at its own hatch.

Boxes spend one charge per tick when providing nearby food or warm-cookware load. The cold reserve loses one for an open door, one per actively processing dessert inside, and one per inside tray still above temperature three after the thermal update. Refilling a held box inside transfers up to eight units from that same reserve. An empty-handed cook at the compressor can spend one of four fuel units for up to five cooling, capped at twenty. Thus opening access or moving warm cookware can reduce another cook's future cooling opportunities; moving a box can improve dessert life while harming hot service or blocking passage.

Hot ingredients age every tick; desserts age only when not cooled. Age thirty spoils a portion. Serving happens before ageing, removes food and leaves dirty cookware with its current temperature. Washing resets food, stage, age and temperature and records spoiled food as waste. Ingredient identities are accounted by loaded portions, pantry stock, service and waste; nothing is replenished by washing.

Transition order is old-state masks and simultaneous walking, rotating rechecked item/door/station actions, then thermal/processing/freshness/cooling costs and terminal evaluation. One door toggle is applied per step. Alternatives include a moving cold escort, stationary cooling stations, hand-carried hot batches before cold batches, or keeping the room shut between access windows. Reward is shared: -0.01 each active tick, +1 per served portion, +2 on all-demand completion or -1 on irrecoverable food failure.

## Native controls

- 0: Wait
- 1: North
- 2: East
- 3: South
- 4: West
- 5: Take nonprocessing tray here or put held tray down
- 6: Take adjacent ice box or put held box down
- 7: Load hot ingredient onto clean tray at pantry
- 8: Load dessert ingredient onto clean tray at pantry
- 9: Start/reheat matching tray at hot stove or cold-prep station
- 10: Serve ready held course at matching hatch and temperature
- 11: Toggle adjacent cold-room door when closure is clear
- 12: Wash held dirty/spoiled tray at sink, or crank compressor empty-handed
- 13: Fill held ice box inside cold room using stored room cooling

Humans and policies must call the same native reset/raw_step/observe/legal_actions interface. This source package does not implement or verify a public adapter, browser controls or a separate dynamics approximation. Invalid/masked actions become wait. Movement rejects old occupied agent cells, including simultaneous swaps, then uses rotating priority (role - old_t) mod 3 for shared targets. Contextual actions are rechecked in rotating role order; a changed item or doorway can make a later action wait.

## Information and terminal semantics

Fully observable. The flat public obs_keys vector includes every cook/item position, carrier, type, food, processing state, temperature, freshness, clock, portable charge, map, door state, cooling/fuel store, station, ingredient and service counter, and per-step event. The chilled flag reports cooling actually applied on the just-finished step, not a promise of cooling after reserves deplete. All active entities are visible to all roles; human and policy information is identical.

Success when both served counts reach two. Failure when remaining pantry stock plus raw/processing/ready portions of either type cannot supply that type's remaining demand. Spoiled food is irrecoverable. Natural causes take priority over H. Each nonterminal raw call advances t once; an already terminal input preserves all state, observation, clock and terminal causes, returns zero rewards/all dones true, and never resets. Terminal masks permit only wait. All three roles remain present until joint termination.

Reset and transition rules are deterministic; key arguments are accepted but unused.

## Invariants and evidence limits

There are exactly three trays and two boxes. Each item is held once or loose; each role holds at most one. For each food type, stock + trays containing that type + served + waste equals three. Spoiled food remains on a tray until washing. Temperatures stay 0..8, freshness 0..30, box charge 0..8, shared cooling 0..20, compressor fuel 0..4, service 0..2 per type. Boxes do not turn into trays or create ingredients. Cook positions remain unique and within the map. These are source-level invariants, not runtime evidence.

All state leaves are fixed-shape int32/bool arrays; signed identifiers are represented losslessly as int32 and the external reset key is not copied into state. Observations and rewards are float32. Declared observation bounds [-5,256] and reward bounds [-2, 6] are reasoned bounds, not measured extrema. Static AST/schema/shape bookkeeping cannot establish runtime compatibility, eager/JIT/vmap parity, solvability, meaningful effect size, successful play, learning or transfer.

## Literal source evidence

- Actual box obstruction: candidate.py:52, literal 'blocked_box = jnp.any(jnp.all(p[:, :, None, :] == s["item_pos"][None, None, :, :], axis=3) & boxes[None, None, :], axis=2)'.
- Thermal service requirement: candidate.py:79, literal 'serving = ((food == 1) & station[:, 4] & (s["temperature"][h] >= 5)) | ((food == 2) & station[:, 5] & (s["temperature"][h] <= 1))'.
- Finite recharge transfer: candidate.py:148, literal 'transfer = jnp.where(act == 13, jnp.minimum(8 - s["box_charge"][h], s["room_cooling"][0]), 0)'.
- Temperature priority: candidate.py:160, literal 'temp = jnp.where(hot_work, jnp.minimum(8, s["temperature"] + 2), jnp.where(cooled, jnp.maximum(0, s["temperature"] - 2), ambient))'.
- Consecutive cold processing: candidate.py:164, literal 'clocks = jnp.where(cold_work & ~progress, 0, s["clock"] + progress.astype(jnp.int32))'.
- Warm cookware externality: candidate.py:175, literal 'warm_ware = room & (s["item_kind"] == 0) & (s["temperature"] > 3)'.
- Shared cooling depletion: candidate.py:176, literal 'room_load = s["door_open"][0].astype(jnp.int32) + jnp.sum((room & cold_work).astype(jnp.int32)) + jnp.sum(warm_ware.astype(jnp.int32))'.
- Food-availability natural failure: candidate.py:180, literal 'failure = jnp.any(s["stock"] + jnp.sum(usable.astype(jnp.int32), axis=1) < 2 - s["served"])'.
- Absorbing dispatch: candidate.py:193, literal 'return jax.lax.cond(state["done"], self.absorb, self.transition, (key, state, actions))'.

- arrange: candidate.py:17 through line 23.
- reset: candidate.py:25 through line 44.
- walking: candidate.py:46 through line 54.
- legal_actions: candidate.py:56 through line 89.
- observe: candidate.py:91 through line 93.
- transition: candidate.py:95 through line 186.
- absorb: candidate.py:188 through line 190.
- raw_step: candidate.py:192 through line 193.

## Blind scripts and attribution

Two original literal seed-zero scripts and one equal-length, single-actor-action paired hypothesis are in playthrough.json. They are unexecuted author hypotheses; timing follows the implemented mechanism and no seed search occurred. JaxMARL README supplied parallel agent-dictionary API context; Pgx supplied state/mask/terminal and external-key context; Craftax supplied JAX spatial-environment context; PureJaxRL supplied the distinction between algorithm implementations and game libraries. No upstream implementation, benchmark layout, learner or commercial asset was copied or imported. The supplied PPO baseline source was not read. Actual current input paths, bytes, hashes, displayed reference ranges and uses are in author_input_freeze.json. Historical IDs in generic copied requirements are background only. A separate explicit repair dispatch is required after this source is frozen.
