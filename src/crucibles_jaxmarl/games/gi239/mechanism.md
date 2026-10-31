# Floodplain Seed Cooperative

Status: AUTHORED_NOT_EXECUTED. Source SHA256: 88c80f02733cafd84a9a6ec94b88c35b218f754b8acd8a799af10423a545881b. All scripts/interventions are UNEXECUTED_AUTHOR_HYPOTHESIS.

Native contract: A=96, O=238, U=19, H=896; scalar int32 actions, float32 observations, fixed int32/bool state. Geometry 48 by 24, x right/y down. Family rural_community.

Twenty-four households follow a serpentine order across six columns and four rows. Each has a plot, slower orchard, nursery, granary, sluice/levee and barn. The directed watershed is a coarse channel network between real sites, not fluid simulation. Source water and salt enter three FIFO slots: an emission on tick t reaches the downstream field three subsequent transitions later. A tray advances only on wet/open source ticks and may stall.

Four tools are exchangeable. The boat tool also crosses drainage strips without intact levees. A sickle needs another actor's simultaneous Assist within two Manhattan cells of its field; cooperation reads original positions while handoffs use serial current positions. Plantings need both care bits and positive soil. Wet variety needs water at least three and salt at most two; dry variety needs water one through three and salt at most one. Orchard growth ticks every eight, fields every four. Flood and high salt erode growth.

A harvest bundle becomes three private food units, three shared seeds, or three levee integrity. Returning an unused seed earns no reward, avoiding seed-withdrawal reward cycling. Bounded rain, evaporation, eight-tick flood phases, levee weathering, pantry consumption and fallow recovery are native rules. Herding moves an animal one cardinal step toward its barn; sheltered living animals support soil recovery, while unsafe animals can lose health in floods.

Work costs 0.002. Harvest pays 0.08. Private storage pays actor 0.15 and household actors 0.025 each. Donating a harvest bundle pays 0.06; donating a seed pays zero. Crop flood/salt stress costs home actors 0.005 per tick; harmful arriving flow additionally charges its recorded real sluice operator 0.03. Empty granaries on thirty-two-tick consumption cost household actors 0.03. Rewards are clipped to [-4,4]. Food versus seeds, upstream relief versus exported salt, and herding versus fieldwork are untested author hypotheses.

## Interface and visibility

Public village/drainage geometry; only selected nearby entity rows with presence flags are observed. Downstream pipes and remote weather are hidden; local field-water relay requires an actual action. Off-view events are debug-only.

Source entry lines: reset 31, legal_actions 67, observe 89, raw_step 123. Exact columns, normalizers, shapes and scene bindings are in manifest.json. Every selected entity slot ends with presence; hidden slots are zero. Crew identities remain fixed, while tools may change. A legal intent can still be blocked by earlier actors, missing simultaneous help, material shortages or geometry. Masked or out-of-range actions become Wait. Actor priority rotates as (loop index + t) mod A.

No early natural terminal; farming season truncates at 896 ticks.

Already-done raw calls preserve every state leaf and observation, return zero rewards and true per-agent dones, and preserve terminal causes. Nonterminal calls advance t exactly once. No autoreset or external state exists. Signed identities and sentinels use int32; reset keys are not copied into state. Observation bounds [0,1] and explicit reward clipping [-4,4] are source declarations, not numerical verification.


## Literal source evidence

- candidate.py:167: assist = jnp.any((jnp.sum(jnp.abs(original - w['field_pos'][f]), axis=1) <= 2) & (actions == 18) & (jnp.arange(96, dtype=jnp.int32) != i))
- candidate.py:168: harvest = (a == 7) & close & w['planted'][f] & (w['growth'][f] >= 8) & (w['tool'][i] == 2) & (w['cargo'][i] == 0) & assist
- candidate.py:183: launch = (a == 11) & jnp.any(sluices) & ((w['cargo'][i] == 1) | (w['cargo'][i] == 2)) & (w['tool'][i] == 3) & ~w['tray_active'][c] & (c < 23)
- candidate.py:203: donation_harvest = donate & (w['cargo'][i] == 3)
- candidate.py:215: herd = (a == 14) & (ad[animal] <= 1) & w['animal_active'][animal]
- candidate.py:242: incoming = s['water_pipe'][:, 0]
- candidate.py:253: charged = (incoming > 0) & bad[::2] & (s['flow_owner'][:, 0] >= 0)
- candidate.py:260: growing = s['planted'] & (s['care'] == 3) & (s['soil'] > 0) & (s['salt'] <= jnp.where(s['variety'] == 0, 2, 1))
- candidate.py:268: animal_hurt = flood & ~s['animal_safe'] & (s['water'][::2] > 5) & s['animal_active']
- candidate.py:275: drift = s['tray_active'] & s['gate'] & (s['tray_progress'] < 3) & (wet >= 2)


## Human operation, provenance and limits
Scene data binds actual positions, stock, tools and events. Human single-role/multiple-role control must call the unchanged native joint action interface. No UI has been implemented, opened or played by this author. Role rendering must restrict event presentation to visible entities; global event inspection is debug-only.


Four independent world families are authored, without upstream code or assets. Supplied external reference files were permitted but not read. No prior game, results, IR, selection, training, heldout, memory, network, API, local model or numerical package was read/invoked. Current population/family240 and dispatch scope override inherited counts. Parentless initial window: 240 used/reserved of 512, feedback 0.


Only bytes/SHA256, JSON/AST and literal structure are checked here. Candidate import/reset/step/observe/legal_actions and provided validators were never executed. Runtime shapes/dtypes, JIT/vmap, terminal replay, mask reachability, script success, finite numerical bounds, resource cost, animation, human playability, non-isomorphism, learning and transfer remain unverified. Independent authorized frozen-source integration must preserve actual failures. No acceptance, deployment or experimental result is authored.
