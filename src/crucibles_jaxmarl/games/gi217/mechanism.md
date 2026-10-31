# Seedwind Orchard

AUTHORED_NOT_EXECUTED. Current initial attempt 00. Requested author gpt-6-astra / max; provider measured model, effort, tokens, elapsed time and cost are unknown.

This original orchard uses twenty-four actual beds, eight physical seed baskets, four shared drying slots, eight windbreak capacities and sixteen airborne seed capacities. Reset seeds the first sixteen crop traits and soil moisture from the supplied key; the remaining beds begin empty. Eight gardeners form two spatially separated sets of four complementary abilities. Any role may harvest, plant, carry/share seed or move a windbreak, while pruning, construction, grafting and reservoir work have explicit specialist masks. Baskets can be borrowed, but one gardener can carry only one basket or windbreak at a time.

Mature crops contain up to two wet seeds. Harvesting yields trait-dependent fruit, returns the parent to stage two and removes its seedbank; without a carried basket those seeds are not saved. Leaving it ripe instead permits timed autonomous seed shedding. Seeds inherit the parent's actual trait, move with public wind, expire after eight ticks, and plant only in an empty moist bed. A windbreak's orientation determines which wind direction it blocks within one tile. Its standing location also blocks walking. Neighboring stage-five/six canopies and windbreaks shade eastward beds. Shade slows hardy crops but retains soil moisture and leaves lush/hybrid growth possible. A gardener's immediate harvest or shield placement therefore changes another plot's resources and future work.

Wet seed traits dry after two/four/six usable ticks at an occupied shared slot; roof level two extends usability into rain. Two dry base traits can be consumed to graft a growing crop into a hybrid with a different fruit value and drying requirement. Foragers prune to create timber or clear young plants to reopen soil; timber builds windbreaks, upgrades the common shed or restores fertility. Waterkeepers withdraw from a common twenty-unit reservoir and carry at most three units to beds. Rain replenishes water/fertility. Crop growth consumes water and nutrients, while ripe seed renewal consumes nutrients. These cycles allow several continuing goals without resetting on an achievement.

## Native roles, controls and observations

A/O/U/H = (8, 1192, 24, 512). U=24 combines selected-trait planting/gifting/release, holding and placing seed baskets or windbreaks, orientation, pruning/clearing, building, mulch, watering and grafting. A=8 is the assigned count for this source and is not a claimed framework maximum. Human and AI controllers use exactly the same scalar int32 joint-action interface and native state transition. The shared platform can assign any subset of roles to humans. No separate browser dynamics are provided.

The static ground map, facility/drying-slot locations, clock, wind/rain/daylight, wind offset, reservoir and roof are public. Dynamic gardeners, bed/canopy/soil/seed states, basket contents/holders, windbreaks and seed particles are visible only within Manhattan radius four. Every hidden record is -1 with an explicit visibility bit; inactive seed/windbreak capacities are also hidden. Each actor additionally sees only its own trait-achievement flags and five reward components. Shared establishment reward reveals an aggregate count, without a remote crop identity, trait or location. No distant current crop is exposed through public scene events or HUD fields.

0: Wait.
1: Walk north and face north.
2: Walk east and face east.
3: Walk south and face south.
4: Walk west and face west.
5: Pick lowest-index adjacent unheld seed basket; hands must be free.
6: Put held seed basket on current tile.
7: Harvest adjacent ripe crop for fruit; collect wet seeds if carrying a basket with room.
8: Plant selected dry seed into adjacent empty moist bed.
9: Set held basket on adjacent empty shared drying slot.
10: Fill carried water to three from shared reservoir at well; waterkeeper.
11: Water adjacent needy bed with one carried unit; waterkeeper.
12: Prune mature growth for two timber or clear young growth for one; forager.
13: Build a front windbreak for two timber; builder.
14: Rotate adjacent standing windbreak orientation.
15: Lift adjacent windbreak with free hands.
16: Place held windbreak on clear front cell.
17: Graft nearby growing crop to hybrid using one dry seed of each base trait; propagator.
18: Upgrade shared drying roof with two timber; builder.
19: Select next seed trait: hardy, lush, hybrid.
20: Give one selected dry seed to adjacent peer's carried basket.
21: Mulch adjacent depleted soil with one timber.
22: Dismantle adjacent windbreak and recover one timber; builder.
23: Release one selected dry seed from held basket into the current wind.

## Resolution, reward and time

Invalid or mask-false commands become wait. Movement rejects old-occupied or duplicate destinations and native pond/standing-windbreak cells. Interactions rotate by (t+k) mod 8; eligible objects, beds, peers and drying slots use lowest index. Construction/interactions precede weather, drying and growth. Autonomous seed emission uses bed index order when fixed capacities compete, followed by motion, blocking/expiry and landing. Hidden objects never affect a remote interaction mask; actionable targets are adjacent.

Five native components are own fruit value 1.5/2.5/4 for hardy/lush/hybrid harvest; own +1.5 first harvest of each trait; shared +0.05 per newly established plant, whether hand-planted or landed; own -0.1 per timber unit consumed by building/roof/mulch; and source-owner -0.05 per lost airborne seed plus plot-owner -0.2 when another gardener harvests or prunes its plant. A single actor harvests at most once per tick, at most 24 empty beds can establish, at most 16 particles are lost and at most eight neighbor removals occur. Thus reward is conservatively within [-4,8]. All five components are explicit float32 state, and unequal private versus shared consequences are intentional.

No mandatory harvest, construction or biodiversity threshold terminates this world. terminated remains false and tick 512 truncates; each harvest, dry cycle and planting leaves the world persistent. Natural causes take priority over H, and already-terminal input is absorbing: unchanged state/observation/clock, zero reward and preserved terminal causes. Raw transitions never reset. Basket wet+dry seed totals never exceed six; plant/graft/gift/release consume exact selected seeds. Original basket identities persist. Seed particles and built windbreaks have fixed capacities. Timber caps at six, carried water at three, soil and growth at six, seedbanks at two, reservoir at twenty. Explicit rain/growth/pruning/seed-regeneration rules define sources and sinks. Physical crop removal and replanting change the same finite beds. The world is neither infinite nor cross-process persistent.

## Literal source evidence

candidate.py:L33: variation=jax.random.randint(key,(24,),0,2,dtype=jnp.int32)

candidate.py:L49: n['canopy']=(s['growth']>=5)&(s['trait']>=0)

candidate.py:L51: n['shade']=jnp.any(hedges[None,:]&(dx>=1)&(dx<=3)&(dy<=1),axis=1)|jnp.any(n['canopy'][None,:]&(tx>=1)&(tx<=3)&(ty==0),axis=1)

candidate.py:L95: harvest=(a==7)&jnp.any(ripe);room=6-jnp.sum(n['wet'][b]+n['dry'][b],dtype=jnp.int32);collected=jnp.where(harvest&has,jnp.minimum(room,n['seedbank'][crop]),0)

candidate.py:L125: graft=(a==17)&has&jnp.any(growing)&(n['dry'][b,0]>0)&(n['dry'][b,1]>0)

candidate.py:L127: roof=(a==18)&(n['wood'][i]>=2)&(n['roof']<2);n['wood']=n['wood'].at[i].add(-2*roof.astype(jnp.int32));n['roof']=jnp.minimum(2,n['roof']+roof.astype(jnp.int32))

candidate.py:L147: ready=n['dry_timer'][:,None]>=jnp.array([2,4,6],jnp.int32)[None,:];transfer=jnp.where(ready,n['wet'],0);n['dry']=n['dry']+transfer;n['wet']=n['wet']-transfer

candidate.py:L124: growing=near&(n['trait']>=0)&(n['growth']<6);bed=jnp.argmax(growing).astype(jnp.int32)
candidate.py:L150: growing=(n['trait']>=0)&(n['growth']<6)&n['daylight']&(n['t']%2==0)&(n['moisture']>=demand)&(n['fertility']>0)&(~n['shade']|(n['trait']!=0))

candidate.py:L76: shed=jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['shed_pos'][None,:,:]),axis=2)<=1,axis=1)
candidate.py:L83: n=dict(s);n['reward_parts']=jnp.zeros((8,5),jnp.float32);established=jnp.array(0,jnp.int32)
candidate.py:L103: n['dry']=n['dry'].at[b,selected].add(-plant.astype(jnp.int32));n['trait']=n['trait'].at[bed].set(jnp.where(plant,selected,n['trait'][bed]));n['growth']=n['growth'].at[bed].set(jnp.where(plant,1,n['growth'][bed]));established=established+plant.astype(jnp.int32)
candidate.py:L154: free=~n['seed_active'];seed=jnp.argmax(free).astype(jnp.int32);shed=(n['growth'][bed]==6)&(n['seedbank'][bed]>0)&(n['t']%6==0)&jnp.any(free)
candidate.py:L166: n['trait']=n['trait'].at[bed].set(jnp.where(land,n['seed_trait'][seed],n['trait'][bed]));n['growth']=n['growth'].at[bed].set(jnp.where(land,1,n['growth'][bed]));n['seed_active']=n['seed_active'].at[seed].set(n['seed_active'][seed]&~land);established=established+land.astype(jnp.int32)

candidate.py:L159: blocked=jnp.any(nearbreak&(n['break_active']&n['deployed']&facingwind)[None,:],axis=1)

candidate.py:L165: bed=jnp.argmax(possible).astype(jnp.int32);land=n['seed_active'][seed]&jnp.any(possible)

candidate.py:L168: n['reward_parts']=n['reward_parts'].at[:,2].set(0.05*established.astype(jnp.float32)).at[:,4].add(-0.05*losses)

candidate.py:L169: n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated'];n=self.visibility(n)

## Presentation and evidence boundary

Warm orchard greens, timber baskets and shelters, gold airborne seed circles, blue well and distinct living canopy/soil attributes. Public map shows only ground; remote dynamic crop and inventory details remain masked. All scene geometry, entity attributes and visibility refer to declared native leaves. Public scene events are restricted to actually public information. Debug/full-state view must be separate from role view.

Two literal scripts and a single actor/action paired hypothesis are supplied, with fixed integer seed and source-derived suffix lengths. All remain UNEXECUTED_AUTHOR_HYPOTHESIS; neither legality of the full trajectory nor any effect, solvability, strategic richness, learning or transfer is certified. Prospective variation could examine trait inheritance, shelter orientation, reservoir supply, gardening roles, drying occupancy or partner harvest/replant timing under future separately frozen source. No such variation or candidate has been run.

Current author checks use stdlib AST/literal shape arithmetic, byte hashes and the supplied trusted manifest/scene/whitelist validators only. No candidate/JAX/NumPy import, native method call, numerical rollout, rendering, training, network/model call or seed search occurred. Four official README excerpts supplied API context only: JaxMARL parallel dictionaries, Pgx explicit-key and terminal interfaces, Craftax spatial game framing, PureJaxRL single-file implementation philosophy. No upstream implementation, layout or asset was copied. The supplied PPO implementation was not read. Exact current input paths, bytes, hashes and excerpt extents are recorded in author_input_freeze.json. Current diversity addendum controls A/H and the scope of open worlds.
