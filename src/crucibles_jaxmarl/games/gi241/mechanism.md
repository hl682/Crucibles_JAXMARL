# Skyport Banquet Kitchens

Status: AUTHORED_NOT_EXECUTED. All gameplay, scripts and causal predictions below are source-derived author hypotheses, not measured results.

## Rules and spatial task

Eight eight-person kitchens share a 48 by 24 terminal. Tools 1/2 are the cooking knife and counterweight handle; tools can be traded. Each initial runner carries a prepared recipe-1 meal and each initial cook carries one raw ingredient. These are explicit reset endowments. Lockers contain 24 ingredients. Every action targets the nearest appropriate station within Manhattan distance one.

Actions 5/6/7/8 take, chop, tend and plate. Chopping creates bench stage 1; tending or old washing steam advances stages to 4. At heat 4, food aged at least seven ticks spoils. Four reservoirs physically between kitchens heat benches at distance at most six on the following step. Washing consumes a water unit and creates three steam units; reservoirs refill four units every 16 ticks up to eight. Each contested station action selects the lowest actor index.

Actions 9/10/11 load a tray, hold its counterweight and receive a tray. An already-loaded lift moves only with a same-step counterweight holder and no receiver. Newly loading does not move it immediately. The upper/lower platform alternation opens one passage tile and closes the other. Consenting adjacent partners exchange tools (15) or cargo and its message (16).

Orders alternate every 12 ticks from a seeded reset code, only visible locally. Reading (14) requires empty hands and carries the code. A chopped dish inherits that message, defaulting to recipe 1; plating/receiving carry its physical tag. Serving (12) accepts only a matching tag while the local departure clock is less than eight; any service returns a dirty tray. The actual departure counter alternates between upper and lower positions.

## Strategies, rewards and termination

Dedicated manual tending competes with coordinated washing that accelerates several benches but risks overheating their old food. Lift staffing competes with foot routes and cooking labor. Runners must time changing local orders and communicate by physically carrying them. These alternatives remain unmeasured.

Every nonrest action costs 0.01. Accepted service pays its actor 1 and every home-kitchen worker 0.05 per meal at that kitchen. Rejected service costs the actor 0.25. Spoiling costs home workers 0.05; moving a lift costs them 0.01. Visitors keep personal service rewards while station ownership fixes shared recipients. Natural completion occurs at 32 cumulative accepted meals; H=896 is the hard horizon.

## Literal diagnostics

Script one loads with runner 2, holds with operator 1, approaches the lower platform and receives. Script two chops, tends through readiness and plates with cook 0. The paired script changes only operator 1's hold versus rest after loading; identical later rest actions are intended to reveal platform and passage differences. Neither script has been executed.


## Native boundary and local information

Every listed actor is an independently selectable row of the same int32 action vector. Human or AI control uses that unchanged row; a frontend, keyboard binding, animation, camera, agent-keyed adapter and human usability have not been implemented or verified by this author. All actors remain active. Rest restores one stamina unit; exertion consumes one, with documented game-specific additions. Movement is cardinal; destinations occupied at the start of a step cannot be entered. Simultaneous claims for an empty destination favor the lowest actor index. Spatial interactions use start-of-step positions and Manhattan distance at most one to the nearest relevant object, unless a specific rule says otherwise. Tool exchange requires both adjacent actors to choose exchange and mutually select each other. Invalid or out-of-range actions become rest deterministically.

Each role receives the own-state prefix plus a 5x5 radius-two patch. The manifest lists every channel and normalization; the renderer's entity masks and map mask are derived from the same radius. Out-of-world patch cells are all -1. There is no omniscient policy feed. Agent IDs, fixed role and home crew are public metadata; current tools, cargo, stamina, messages and world attributes are visible only in the patch. Entity positions are used by movement, proximity, ecology or construction; none are layout-only proxies.

A nonterminal call advances the int32 clock once. Natural completion has priority over horizon truncation. A terminal raw call preserves every state leaf, observation and terminal cause, returns zero reward and all dones true, and does not reset. All boundary/state integers are int32; the supplied uint32 key is used only through JAX randomness and is not copied into state. State keys/shapes remain fixed. Rewards declare conservative [-8,8] bounds and observations [-1,1]; these are interface ranges, not scientific performance gates. Concurrency, bounds, terminal absorption, JIT/vmap, causal effects, throughput and numerical parity remain unverified.

## Source-line evidence

- candidate.py:115 'moved =': loaded-lift counterweight dependency
- candidate.py:102 'heat =': delayed shared steam changes nearby benches
- candidate.py:126 'accepted =': dish tag and departure timing govern service
- candidate.py:37 's["order"]': seeded local order
- candidate.py:122 'n["lift_pos"]': physical lift position changes

## Provenance and limits

The source is original for this batch and train split. Only current batch authority, four assigned design/request pairs, population/family direction, human-playability addendum and copied interface contracts/templates were read. No prior candidate, IR, result, training, heldout or memory was read; no remote query was made. Official repository references were allowed but not read or copied. The family inspiration comes solely from the supplied design direction: no commercial assets or upstream game implementation is included. PureJaxRL/Craftax baseline code is not used as a game or learner replacement.

The two literal scripts and the paired single-action hypothesis in playthrough.json are unexecuted diagnostics, not a controller restriction or proof of task success. Source-authored invariants and bounds await the independent sealed validator. No candidate import, Python candidate execution, JAX/NumPy computation, validator, training, replay, browser session, human play, accepted R0/M0, solvability, learning, strategic richness, non-isomorphism or transfer result is claimed. Zero, negative, failed, missing and indeterminate outcomes remain admissible.
