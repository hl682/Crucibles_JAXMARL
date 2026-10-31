# Clockwork Banquet

Status: AUTHORED_NOT_EXECUTED. Train-only initial source, attempt_00. No runtime, replay, solvability, adapter compatibility, mechanism success, learning or transfer acceptance is claimed.

## Native interface and human operation

A=3, O=150, U=11, H=160. Humans and future policies use exactly GeneratedEnv.reset, raw_step, observe and legal_actions. Presentation adapter and human controls are separate integration work, not implemented or verified here. No alternative browser dynamics exist.

Clockwork Banquet uses a warm copper-and-tile kitchen, not the quarry's block palette or the frontier's territorial board. All cells are walkable. Seven distinct physical stations identify two pantries, two course-specific stoves, a sink, a conveyor control and the hatch. Cooks occupy unique cells, but a tray and a cook may share a cell so a cook can retrieve it. The belt is the actual seven-cell horizontal segment (1..7,3).

Three tray identities persist throughout the game. Empty hands can take a loose non-cooking tray on the same cell; a held tray can be put down only without another loose tray there. Pantry loading consumes one of that pantry's three ingredients and places that course on a clean held tray. Starting a matching raw course at its stove places the tray on the stove, freeing the cook. The start tick is cooking tick one; tick three converts it to ready with ready-age zero. A ready portion ages even while carried, and becomes spoiled at age twelve. It cannot be served after spoilage. Serving precedes ageing and leaves food=0 on a dirty tray. Washing a dirty or spoiled held tray returns it to clean, recording any spoiled ingredient as waste.

Order is action masking, agent movement, direction reversal, rotating-order physical tray transactions, automatic cooking/ready ageing, the even-tick conveyor pulse, and terminal evaluation. Loose belt trays move with the one global direction. Motion checks old loose-tray occupancy, so a trailing tray waits for a gap rather than entering a cell vacated on that pulse. There is no belt wrap or endpoint deletion. Carried trays do not move independently. A direction change before a pulse is effective on that pulse.

An early reversal can bring a raw course toward one teammate while moving an urgently needed clean or dirty tray away from another. A cook can carry a tray directly, use the belt, specialize at a stove, or leave a tray for another cook. The next-course requirement makes a fast protein dish wait if vegetables are not ready, while washing ties future ingredient access to another person's cleanup. All of these use the same physical object identities and native action masks. No hidden planning or scripted teammate behavior is implemented.

Every role receives -0.01 per active step, plus 1 when the next requested course is served, plus 2 on success, or minus 1 on irrecoverable next-course failure. There is no private reward or bonus for repeated washing, reversing or carrying.

## Action legend

- 0: Wait
- 1: North
- 2: East
- 3: South
- 4: West
- 5: Take a non-cooking tray here, or put held tray on a free floor cell
- 6: Load clean held tray at a stocked pantry
- 7: Start matching raw course at its stove, leaving the tray there
- 8: Reverse conveyor at control (4,5)
- 9: Wash held dirty/spoiled tray at sink (1,5)
- 10: Serve the requested ready held course at hatch (7,1)

## Information and boundary semantics

Fully observable. A role one-hot precedes the complete public obs_keys vector: map, all cooks, all tray positions/holders/food/stages/ages, station locations, pantry counts, conveyor direction, next course index, served/waste accounting and per-step events. All entities are visible to all roles. Labels and native state provide the same information to human and policy users.

Invalid or masked actions become wait. Contention may make a legal action have no effect. Agent movement forbids entering another role old cell, including swaps, then resolves a shared destination by rotating priority (role - old_t) mod 3. All roles remain active until joint termination; a role completing its own deliveries can continue to help. Terminal masks expose only wait.

State leaves are fixed-shape int32/bool arrays. Observations/rewards are float32. Signed IDs use int32 losslessly, and the reset key is not copied into state. Reset and transition rules are deterministic. The required key arguments are accepted but unused.

Natural termination: Success when course reaches four. Failure when the next required type has no pantry stock and no raw/cooking/ready portion left on any tray; spoiled portions cannot be recovered. Nonterminal calls increment t once. Natural completion takes precedence over the cap. Already-terminal calls preserve state, observation, clock and actual terminal causes, return zero reward/all dones true and never reset. H is an episode cap, not a scientific acceptance gate.

## Invariants and limits

Exactly three trays exist; each has one holder or a loose position, and each cook holds at most one. Loose tray positions are unique; held trays can share coordinates with a distinct loose tray, so no blanket tray-position uniqueness invariant is declared. For each type, pantry stock + number of trays containing that type + served + waste equals its original three ingredients. A spoiled tray still contains its ingredient until washing. Pantry stock stays 0..3, food 0..2, stage 0..5, timer 0..11, course 0..4, and direction is -1 or +1. All agent and tray positions are within the board. These are source-reasoned statements, not numerical checks.

Observation bounds [-5,256] and reward bounds [-2, 4] are declared source-level bounds, not measured extrema. Static shape/schema checks do not establish numerical correctness, eager/JIT/vmap parity, arbitrary-trajectory validity or successful play. Those require a separate frozen-source dispatch.

## Literal source evidence

- Distinct physical stations: candidate.py:24, literal 'stations = jnp.array([[1, 1], [7, 5], [3, 1], [5, 1], [1, 5], [4, 5], [7, 1]], jnp.int32)'.
- Contextual serve mask: candidate.py:71, literal 'mask = mask.at[:, 10].set(~free & at_station[:, 6] & (s["stage"][h] == 3) & (food == demand) & (s["course"] < 4))'.
- Shared direction reversal: candidate.py:94, literal 's["direction"] = jnp.where(s["reversed"], -s["direction"], s["direction"])'.
- Ingredient conservation debit: candidate.py:109, literal 's["stock"] = s["stock"].at[pantry].add(-load.astype(jnp.int32))'.
- Cooking delay: candidate.py:132, literal 'cooked = cooking & (timers >= 3)'.
- Freshness window: candidate.py:133, literal 'spoiled = ready & (timers >= 12)'.
- Real belt collision and timing: candidate.py:140, literal 'belt_move = on_belt & (jnp.mod(s["t"], 2) == 0) & (target[:, 0] >= 1) & (target[:, 0] <= 7) & ~blocked'.
- Irrecoverable natural failure: candidate.py:146, literal 'failure = ~success & (s["stock"][demand - 1] == 0) & ~jnp.any(usable)'.
- Raw absorption: candidate.py:160, literal 'return jax.lax.cond(state["done"], self.absorb, self.advance, (key, state, actions))'.

- refresh: candidate.py:16 through line 21.
- reset: candidate.py:23 through line 43.
- movement: candidate.py:45 through line 51.
- legal_actions: candidate.py:53 through line 72.
- observe: candidate.py:74 through line 76.
- advance: candidate.py:78 through line 153.
- absorb: candidate.py:155 through line 157.
- raw_step: candidate.py:159 through line 160.

## Hypotheses and sources

playthrough.json contains original literal scripts and one paired hypothesis differing in exactly one actor-action component. Seed zero is literal and unsearched. Suffix timing follows the source mechanism; no branch has run and no script is a success receipt.

The supplied JaxMARL README informed parallel agent-dictionary API context; Pgx informed mask/terminal and external-key context; Craftax informed spatial mechanics and raw/autoreset separation; PureJaxRL is an algorithm reference, not a new game library. No upstream implementation, benchmark layout, learner, commercial asset or code was imported or copied. The supplied Craftax PPO source was not read. Public performance claims and incidentally displayed score rows were not used. All actual input paths, bytes, hashes and uses are frozen in author_input_freeze.json. Current 194-197 designs and family principles control this independent game; historical IDs in copied requirements do not assign work.
