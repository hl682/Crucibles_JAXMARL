"""Seedwind Orchard: eight gardeners, physical seed dispersal and shared drying."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents=8
    observation_size=1192
    num_actions=24
    max_steps=512
    public_keys=('t','done','terminated','truncated','terrain','facility_pos','shed_pos','wind','rain','daylight','wind_offset','reservoir','roof')

    def reset(self,key):
        s={
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.zeros((24,24),jnp.int32),
            'agent_pos':jnp.array([[4,5],[7,5],[10,5],[13,5],[16,5],[19,5],[10,13],[13,13]],jnp.int32),
            'agent_active':jnp.ones((8,),jnp.bool_),'agent_visible':jnp.ones((8,8),jnp.bool_),
            'role':jnp.array([0,1,2,3,0,1,2,3],jnp.int32),'facing':jnp.ones((8,),jnp.int32),'wood':jnp.array([0,0,2,0,0,0,2,0],jnp.int32),'carrying':jnp.full((8,),-1,jnp.int32),'carry_break':jnp.full((8,),-1,jnp.int32),'water':jnp.zeros((8,),jnp.int32),'selected':jnp.zeros((8,),jnp.int32),'achieved':jnp.zeros((8,3),jnp.bool_),
            'bed_pos':jnp.array([[4,4],[7,4],[10,4],[13,4],[16,4],[19,4],[4,8],[7,8],[10,8],[13,8],[16,8],[19,8],[4,14],[7,14],[10,14],[13,14],[16,14],[19,14],[4,18],[7,18],[10,18],[13,18],[16,18],[19,18]],jnp.int32),
            'bed_active':jnp.ones((24,),jnp.bool_),'bed_visible':jnp.ones((8,24),jnp.bool_),
            'bed_owner':jnp.array([0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7],jnp.int32),
            'trait':jnp.full((24,),-1,jnp.int32),'growth':jnp.array([6,6,6,6,6,6,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0],jnp.int32),'canopy':jnp.zeros((24,),jnp.bool_),'moisture':jnp.full((24,),4,jnp.int32),'fertility':jnp.full((24,),5,jnp.int32),'seedbank':jnp.array([2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],jnp.int32),'shade':jnp.zeros((24,),jnp.bool_),
            'basket_pos':jnp.array([[4,5],[7,5],[10,5],[13,5],[16,5],[19,5],[10,13],[13,13]],jnp.int32),
            'basket_active':jnp.ones((8,),jnp.bool_),'basket_visible':jnp.ones((8,8),jnp.bool_),'basket_holder':jnp.full((8,),-1,jnp.int32),'basket_slot':jnp.full((8,),-1,jnp.int32),'wet':jnp.array([[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0]],jnp.int32),'dry':jnp.zeros((8,3),jnp.int32),'dry_timer':jnp.zeros((8,),jnp.int32),
            'break_pos':jnp.array([[8,4],[15,8],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],jnp.int32),'break_active':jnp.array([True,True,False,False,False,False,False,False]),'break_visible':jnp.zeros((8,8),jnp.bool_),'break_holder':jnp.full((8,),-1,jnp.int32),'break_axis':jnp.array([1,0,0,0,0,0,0,0],jnp.int32),'deployed':jnp.array([True,True,False,False,False,False,False,False]),
            'seed_pos':jnp.zeros((16,2),jnp.int32),'seed_active':jnp.zeros((16,),jnp.bool_),'seed_visible':jnp.zeros((8,16),jnp.bool_),'seed_trait':jnp.zeros((16,),jnp.int32),'seed_owner':jnp.zeros((16,),jnp.int32),'seed_age':jnp.zeros((16,),jnp.int32),
            'facility_pos':jnp.array([[12,12],[10,10]],jnp.int32),'facility_active':jnp.ones((2,),jnp.bool_),'facility_visible':jnp.ones((8,2),jnp.bool_),
            'shed_pos':jnp.array([[10,10],[11,10],[10,11],[11,11]],jnp.int32),'shed_active':jnp.ones((4,),jnp.bool_),'shed_visible':jnp.ones((8,4),jnp.bool_),
            'wind':jnp.array([1,0],jnp.int32),'rain':jnp.array(False),'daylight':jnp.array(True),'wind_offset':jnp.array(0,jnp.int32),'reservoir':jnp.array(20,jnp.int32),'roof':jnp.array(1,jnp.int32),
            'reward_parts':jnp.zeros((8,5),jnp.float32)
        }
        variation=jax.random.randint(key,(24,),0,2,dtype=jnp.int32)
        s['trait']=jnp.where(jnp.arange(24,dtype=jnp.int32)<16,variation,-1)
        s['moisture']=s['moisture']+variation;s['wind_offset']=variation[23]
        s['terrain']=s['terrain'].at[20:23,20:23].set(1)
        s=self.visibility(s)
        return self.observe(s),s

    def visibility(self,s):
        n=dict(s);p=s['agent_pos']
        n['agent_visible']=jnp.sum(jnp.abs(p[:,None,:]-p[None,:,:]),axis=2)<=4
        n['bed_visible']=jnp.sum(jnp.abs(p[:,None,:]-s['bed_pos'][None,:,:]),axis=2)<=4
        n['basket_visible']=jnp.sum(jnp.abs(p[:,None,:]-s['basket_pos'][None,:,:]),axis=2)<=4
        n['break_visible']=(jnp.sum(jnp.abs(p[:,None,:]-s['break_pos'][None,:,:]),axis=2)<=4)&s['break_active'][None,:]
        n['seed_visible']=(jnp.sum(jnp.abs(p[:,None,:]-s['seed_pos'][None,:,:]),axis=2)<=4)&s['seed_active'][None,:]
        hedges=s['break_active']&s['deployed']&(s['break_holder']==-1)
        dx=s['bed_pos'][:,None,0]-s['break_pos'][None,:,0];dy=jnp.abs(s['bed_pos'][:,None,1]-s['break_pos'][None,:,1])
        n['canopy']=(s['growth']>=5)&(s['trait']>=0)
        tx=s['bed_pos'][:,None,0]-s['bed_pos'][None,:,0];ty=jnp.abs(s['bed_pos'][:,None,1]-s['bed_pos'][None,:,1])
        n['shade']=jnp.any(hedges[None,:]&(dx>=1)&(dx<=3)&(dy<=1),axis=1)|jnp.any(n['canopy'][None,:]&(tx>=1)&(tx<=3)&(ty==0),axis=1)
        return n

    def record(self,data,visible):
        return jnp.concatenate((visible[:,None],jnp.where(visible[:,None],data,-1)),axis=1).ravel().astype(jnp.float32)

    def observe(self,s):
        public=jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.public_keys])
        agents=jnp.concatenate((s['agent_pos'],s['role'][:,None],s['facing'][:,None],s['wood'][:,None],s['carrying'][:,None],s['carry_break'][:,None],s['water'][:,None],s['selected'][:,None]),axis=1)
        beds=jnp.concatenate((s['bed_pos'],s['bed_owner'][:,None],s['trait'][:,None],s['growth'][:,None],s['canopy'][:,None],s['moisture'][:,None],s['fertility'][:,None],s['seedbank'][:,None],s['shade'][:,None]),axis=1)
        baskets=jnp.concatenate((s['basket_pos'],s['basket_holder'][:,None],s['basket_slot'][:,None],s['wet'],s['dry'],s['dry_timer'][:,None]),axis=1)
        breaks=jnp.concatenate((s['break_pos'],s['break_holder'][:,None],s['break_axis'][:,None],s['deployed'][:,None]),axis=1)
        seeds=jnp.concatenate((s['seed_pos'],s['seed_trait'][:,None],s['seed_owner'][:,None],s['seed_age'][:,None]),axis=1)
        rows=[]
        for i in range(8):
            own=jnp.concatenate((jnp.array([i],jnp.float32),s['achieved'][i],s['reward_parts'][i]))
            rows=rows+[jnp.concatenate((public,own,self.record(agents,s['agent_visible'][i]),self.record(beds,s['bed_visible'][i]),self.record(baskets,s['basket_visible'][i]),self.record(breaks,s['break_visible'][i]),self.record(seeds,s['seed_visible'][i]))).astype(jnp.float32)]
        return jnp.stack(rows)

    def legal_actions(self,s):
        p=s['agent_pos'];d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32);q=p[:,None,:]+d[None,:,:]
        solid=s['break_active']&s['deployed']&(s['break_holder']==-1)
        blocked=jnp.any(jnp.all(q[:,:,None,:]==s['break_pos'][None,None,:,:],axis=3)&solid[None,None,:],axis=2)
        move=jnp.all((q>=0)&(q<24),axis=2)&~blocked&(s['terrain'][jnp.clip(q[:,:,1],0,23),jnp.clip(q[:,:,0],0,23)]==0)
        bed=jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['bed_pos'][None,:,:]),axis=2)<=1,axis=1)
        shed=jnp.any(jnp.sum(jnp.abs(p[:,None,:]-s['shed_pos'][None,:,:]),axis=2)<=1,axis=1)
        well=jnp.sum(jnp.abs(p-s['facility_pos'][0]),axis=1)<=1
        basket=s['carrying']>=0;empty=~basket&(s['carry_break']<0);yes=jnp.ones((8,),jnp.bool_);role=s['role']
        extra=jnp.stack((empty,basket,bed,bed&basket,shed&basket,(role==3)&well,(role==3)&bed,(role==1)&bed,role==2,yes,empty,s['carry_break']>=0,(role==0)&bed&basket,(role==2)&shed,yes,basket,bed,role==2,basket),axis=1)
        return (jnp.concatenate((move,extra),axis=1)&~s['done']).at[:,0].set(True)

    def advance(self,s,actions):
        n=dict(s);n['reward_parts']=jnp.zeros((8,5),jnp.float32);established=jnp.array(0,jnp.int32)
        ac=jnp.clip(actions,0,23);ac=jnp.where((actions>=0)&(actions<24)&self.legal_actions(s)[jnp.arange(8,dtype=jnp.int32),ac],ac,0)
        d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32);target=s['agent_pos']+d[jnp.minimum(ac,4)]*((ac>0)&(ac<5))[:,None];others=~jnp.eye(8,dtype=jnp.bool_)
        block=jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&others,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&others,axis=1)
        n['agent_pos']=jnp.where(block[:,None],s['agent_pos'],target);n['facing']=jnp.where((ac>0)&(ac<5),ac,s['facing'])
        for k in range(8):
            i=(s['t']+k)%8;p=n['agent_pos'][i];a=ac[i];has=n['carrying'][i]>=0;b=jnp.clip(n['carrying'][i],0,7);freehands=~has&(n['carry_break'][i]<0)
            loose=(n['basket_holder']==-1)&(jnp.sum(jnp.abs(n['basket_pos']-p),axis=1)<=1);pick=jnp.argmax(loose).astype(jnp.int32);take=(a==5)&freehands&jnp.any(loose)
            n['basket_holder']=n['basket_holder'].at[pick].set(jnp.where(take,i,n['basket_holder'][pick]));n['basket_slot']=n['basket_slot'].at[pick].set(jnp.where(take,-1,n['basket_slot'][pick]))
            drop=(a==6)&has;n['basket_holder']=n['basket_holder'].at[b].set(jnp.where(drop,-1,n['basket_holder'][b]));n['basket_pos']=n['basket_pos'].at[b].set(jnp.where(drop,p,n['basket_pos'][b]))
            near=jnp.sum(jnp.abs(n['bed_pos']-p),axis=1)<=1
            ripe=near&(n['growth']>=6)&(n['trait']>=0);crop=jnp.argmax(ripe).astype(jnp.int32);trait=jnp.clip(n['trait'][crop],0,2);owner=n['bed_owner'][crop]
            harvest=(a==7)&jnp.any(ripe);room=6-jnp.sum(n['wet'][b]+n['dry'][b],dtype=jnp.int32);collected=jnp.where(harvest&has,jnp.minimum(room,n['seedbank'][crop]),0)
            n['wet']=n['wet'].at[b,trait].add(collected);n['dry_timer']=n['dry_timer'].at[b].set(jnp.where(collected>0,0,n['dry_timer'][b]))
            n['growth']=n['growth'].at[crop].set(jnp.where(harvest,2,n['growth'][crop]));n['seedbank']=n['seedbank'].at[crop].set(jnp.where(harvest,0,n['seedbank'][crop]))
            value=jnp.array([1.5,2.5,4.0],jnp.float32)[trait]
            n['reward_parts']=n['reward_parts'].at[i,0].add(value*harvest.astype(jnp.float32)).at[i,1].add(1.5*(harvest&~n['achieved'][i,trait]).astype(jnp.float32)).at[owner,4].add(-0.2*(harvest&(owner!=i)).astype(jnp.float32))
            n['achieved']=n['achieved'].at[i,trait].set(n['achieved'][i,trait]|harvest)
            empty=near&(n['trait']<0)&(n['moisture']>0);bed=jnp.argmax(empty).astype(jnp.int32);selected=n['selected'][i]
            plant=(a==8)&has&jnp.any(empty)&(n['dry'][b,selected]>0)
            n['dry']=n['dry'].at[b,selected].add(-plant.astype(jnp.int32));n['trait']=n['trait'].at[bed].set(jnp.where(plant,selected,n['trait'][bed]));n['growth']=n['growth'].at[bed].set(jnp.where(plant,1,n['growth'][bed]));established=established+plant.astype(jnp.int32)
            slots=(jnp.sum(jnp.abs(n['shed_pos']-p),axis=1)<=1)&~jnp.any(n['basket_slot'][None,:]==jnp.arange(4,dtype=jnp.int32)[:,None],axis=1)
            slot=jnp.argmax(slots).astype(jnp.int32);dry=(a==9)&has&jnp.any(slots)
            n['basket_slot']=n['basket_slot'].at[b].set(jnp.where(dry,slot,n['basket_slot'][b]));n['basket_holder']=n['basket_holder'].at[b].set(jnp.where(dry,-1,n['basket_holder'][b]));n['basket_pos']=n['basket_pos'].at[b].set(jnp.where(dry,n['shed_pos'][slot],n['basket_pos'][b]))
            fill=(a==10)&(n['reservoir']>0);amount=jnp.where(fill,jnp.minimum(3-n['water'][i],n['reservoir']),0)
            n['water']=n['water'].at[i].add(amount);n['reservoir']=n['reservoir']-amount
            needy=near&(n['moisture']<6);bed=jnp.argmax(needy).astype(jnp.int32);water=(a==11)&jnp.any(needy)&(n['water'][i]>0)
            n['water']=n['water'].at[i].add(-water.astype(jnp.int32));n['moisture']=n['moisture'].at[bed].set(jnp.minimum(6,n['moisture'][bed]+3*water.astype(jnp.int32)))
            woody=near&(n['trait']>=0);bed=jnp.argmax(woody).astype(jnp.int32);prune=(a==12)&jnp.any(woody)&(n['wood'][i]<=4);young=n['growth'][bed]<3
            n['wood']=n['wood'].at[i].add(jnp.where(young,1,2).astype(jnp.int32)*prune.astype(jnp.int32));n['growth']=n['growth'].at[bed].set(jnp.where(prune,jnp.where(young,0,2),n['growth'][bed]));n['seedbank']=n['seedbank'].at[bed].set(jnp.where(prune,0,n['seedbank'][bed]));n['trait']=n['trait'].at[bed].set(jnp.where(prune&young,-1,n['trait'][bed]))
            n['reward_parts']=n['reward_parts'].at[n['bed_owner'][bed],4].add(-0.2*(prune&(n['bed_owner'][bed]!=i)).astype(jnp.float32))
            front=p+d[n['facing'][i]];inside=jnp.all((front>=0)&(front<24));f=jnp.clip(front,0,23)
            obstruction=jnp.any(jnp.all(n['agent_pos']==front,axis=1))|jnp.any(jnp.all(n['bed_pos']==front,axis=1))|jnp.any(jnp.all(n['break_pos']==front,axis=1)&n['break_active']&n['deployed'])|(n['terrain'][f[1],f[0]]!=0)
            available=~n['break_active'];h=jnp.argmax(available).astype(jnp.int32);build=(a==13)&inside&~obstruction&jnp.any(available)&(n['wood'][i]>=2)
            n['break_pos']=n['break_pos'].at[h].set(jnp.where(build,front,n['break_pos'][h]));n['break_active']=n['break_active'].at[h].set(n['break_active'][h]|build);n['deployed']=n['deployed'].at[h].set(n['deployed'][h]|build);n['break_axis']=n['break_axis'].at[h].set(jnp.where(build,(n['facing'][i]%2==0).astype(jnp.int32),n['break_axis'][h]))
            n['wood']=n['wood'].at[i].add(-2*build.astype(jnp.int32))
            nearby=n['break_active']&(n['break_holder']==-1)&(jnp.sum(jnp.abs(n['break_pos']-p),axis=1)<=1);h=jnp.argmax(nearby).astype(jnp.int32)
            rotate=(a==14)&jnp.any(nearby);lift=(a==15)&freehands&jnp.any(nearby)
            n['break_axis']=n['break_axis'].at[h].set(jnp.where(rotate,1-n['break_axis'][h],n['break_axis'][h]));n['break_holder']=n['break_holder'].at[h].set(jnp.where(lift,i,n['break_holder'][h]));n['deployed']=n['deployed'].at[h].set(n['deployed'][h]&~lift)
            h=jnp.clip(n['carry_break'][i],0,7);place=(a==16)&(n['carry_break'][i]>=0)&inside&~obstruction
            n['break_holder']=n['break_holder'].at[h].set(jnp.where(place,-1,n['break_holder'][h]));n['break_pos']=n['break_pos'].at[h].set(jnp.where(place,front,n['break_pos'][h]));n['deployed']=n['deployed'].at[h].set(n['deployed'][h]|place)
            growing=near&(n['trait']>=0)&(n['growth']<6);bed=jnp.argmax(growing).astype(jnp.int32)
            graft=(a==17)&has&jnp.any(growing)&(n['dry'][b,0]>0)&(n['dry'][b,1]>0)
            n['dry']=n['dry'].at[b,0].add(-graft.astype(jnp.int32)).at[b,1].add(-graft.astype(jnp.int32));n['trait']=n['trait'].at[bed].set(jnp.where(graft,2,n['trait'][bed]))
            roof=(a==18)&(n['wood'][i]>=2)&(n['roof']<2);n['wood']=n['wood'].at[i].add(-2*roof.astype(jnp.int32));n['roof']=jnp.minimum(2,n['roof']+roof.astype(jnp.int32))
            n['selected']=n['selected'].at[i].set(jnp.where(a==19,(selected+1)%3,selected))
            peers=(jnp.sum(jnp.abs(n['agent_pos']-p),axis=1)<=1)&(jnp.arange(8,dtype=jnp.int32)!=i)&(n['carrying']>=0);peer=jnp.argmax(peers).astype(jnp.int32);otherbasket=jnp.clip(n['carrying'][peer],0,7)
            gift=(a==20)&has&jnp.any(peers)&(n['dry'][b,selected]>0)&(jnp.sum(n['wet'][otherbasket]+n['dry'][otherbasket],dtype=jnp.int32)<6)
            n['dry']=n['dry'].at[b,selected].add(-gift.astype(jnp.int32)).at[otherbasket,selected].add(gift.astype(jnp.int32))
            tired=near&(n['fertility']<6);bed=jnp.argmax(tired).astype(jnp.int32);mulch=(a==21)&jnp.any(tired)&(n['wood'][i]>0)
            n['wood']=n['wood'].at[i].add(-mulch.astype(jnp.int32));n['fertility']=n['fertility'].at[bed].set(jnp.minimum(6,n['fertility'][bed]+3*mulch.astype(jnp.int32)))
            h=jnp.argmax(nearby).astype(jnp.int32);reclaim=(a==22)&jnp.any(nearby)&(n['wood'][i]<6)
            n['break_active']=n['break_active'].at[h].set(n['break_active'][h]&~reclaim);n['deployed']=n['deployed'].at[h].set(n['deployed'][h]&~reclaim);n['wood']=n['wood'].at[i].add(reclaim.astype(jnp.int32))
            free=~n['seed_active'];seed=jnp.argmax(free).astype(jnp.int32);release=(a==23)&has&jnp.any(free)&(n['dry'][b,selected]>0)
            n['dry']=n['dry'].at[b,selected].add(-release.astype(jnp.int32));n['seed_active']=n['seed_active'].at[seed].set(n['seed_active'][seed]|release);n['seed_pos']=n['seed_pos'].at[seed].set(jnp.where(release,p,n['seed_pos'][seed]));n['seed_trait']=n['seed_trait'].at[seed].set(jnp.where(release,selected,n['seed_trait'][seed]));n['seed_owner']=n['seed_owner'].at[seed].set(jnp.where(release,i,n['seed_owner'][seed]));n['seed_age']=n['seed_age'].at[seed].set(jnp.where(release,0,n['seed_age'][seed]))
            n['reward_parts']=n['reward_parts'].at[i,3].add(-0.1*(2*build.astype(jnp.float32)+2*roof.astype(jnp.float32)+mulch.astype(jnp.float32)))
            match=n['basket_holder'][None,:]==jnp.arange(8,dtype=jnp.int32)[:,None];n['carrying']=jnp.where(jnp.any(match,axis=1),jnp.argmax(match,axis=1),-1).astype(jnp.int32)
            bm=n['break_holder'][None,:]==jnp.arange(8,dtype=jnp.int32)[:,None];n['carry_break']=jnp.where(jnp.any(bm,axis=1),jnp.argmax(bm,axis=1),-1).astype(jnp.int32)
        n['basket_pos']=jnp.where((n['basket_holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['basket_holder'],0,7)],n['basket_pos']);n['break_pos']=jnp.where((n['break_holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['break_holder'],0,7)],n['break_pos'])
        n['t']=s['t']+1;n['rain']=(n['t']%24)>=18;n['daylight']=(n['t']%12)<8
        winds=jnp.array([[1,0],[0,1],[-1,0],[0,-1]],jnp.int32);n['wind']=winds[(n['t']//16+n['wind_offset'])%4]
        n['reservoir']=jnp.minimum(20,n['reservoir']+jnp.where(n['rain']&(n['t']%4==0),4,0))
        n=self.visibility(n)
        drying=(n['basket_slot']>=0)&(~n['rain']|(n['roof']>=2));n['dry_timer']=jnp.minimum(6,n['dry_timer']+drying.astype(jnp.int32))
        ready=n['dry_timer'][:,None]>=jnp.array([2,4,6],jnp.int32)[None,:];transfer=jnp.where(ready,n['wet'],0);n['dry']=n['dry']+transfer;n['wet']=n['wet']-transfer
        rainfeed=n['rain']&(n['t']%4==0);n['moisture']=jnp.minimum(6,n['moisture']+rainfeed.astype(jnp.int32));n['fertility']=jnp.minimum(6,n['fertility']+(n['rain']&(n['t']%8==0)).astype(jnp.int32))
        demand=jnp.where(n['trait']==1,2,1).astype(jnp.int32)
        growing=(n['trait']>=0)&(n['growth']<6)&n['daylight']&(n['t']%2==0)&(n['moisture']>=demand)&(n['fertility']>0)&(~n['shade']|(n['trait']!=0))
        n['growth']=n['growth']+growing.astype(jnp.int32);n['moisture']=jnp.maximum(0,n['moisture']-demand*growing.astype(jnp.int32)-((n['t']%8==0)&~n['shade']&~n['rain']).astype(jnp.int32));n['fertility']=n['fertility']-growing.astype(jnp.int32)
        replenish=(n['growth']==6)&(n['seedbank']<2)&(n['fertility']>0)&(n['t']%12==0);n['seedbank']=n['seedbank']+replenish.astype(jnp.int32);n['fertility']=n['fertility']-replenish.astype(jnp.int32)
        for bed in range(24):
            free=~n['seed_active'];seed=jnp.argmax(free).astype(jnp.int32);shed=(n['growth'][bed]==6)&(n['seedbank'][bed]>0)&(n['t']%6==0)&jnp.any(free)
            n['seed_active']=n['seed_active'].at[seed].set(n['seed_active'][seed]|shed);n['seed_pos']=n['seed_pos'].at[seed].set(jnp.where(shed,n['bed_pos'][bed],n['seed_pos'][seed]));n['seed_trait']=n['seed_trait'].at[seed].set(jnp.where(shed,n['trait'][bed],n['seed_trait'][seed]));n['seed_owner']=n['seed_owner'].at[seed].set(jnp.where(shed,n['bed_owner'][bed],n['seed_owner'][seed]));n['seed_age']=n['seed_age'].at[seed].set(jnp.where(shed,0,n['seed_age'][seed]));n['seedbank']=n['seedbank'].at[bed].add(-shed.astype(jnp.int32))
        proposed=n['seed_pos']+n['wind'][None,:];step=n['t']%2==0
        nearbreak=jnp.sum(jnp.abs(proposed[:,None,:]-n['break_pos'][None,:,:]),axis=2)<=1
        facingwind=jnp.where(n['break_axis']==1,n['wind'][0]!=0,n['wind'][1]!=0)
        blocked=jnp.any(nearbreak&(n['break_active']&n['deployed']&facingwind)[None,:],axis=1)
        inside=jnp.all((proposed>=0)&(proposed<24),axis=1);lost=n['seed_active']&step&(~inside|blocked)
        n['seed_pos']=jnp.where((n['seed_active']&step&~lost)[:,None],proposed,n['seed_pos']);n['seed_age']=jnp.minimum(8,n['seed_age']+n['seed_active'].astype(jnp.int32));lost=lost|(n['seed_active']&(n['seed_age']>=8))
        n['seed_active']=n['seed_active']&~lost
        for seed in range(16):
            possible=(jnp.sum(jnp.abs(n['bed_pos']-n['seed_pos'][seed]),axis=1)<=1)&(n['trait']<0)&(n['moisture']>0)
            bed=jnp.argmax(possible).astype(jnp.int32);land=n['seed_active'][seed]&jnp.any(possible)
            n['trait']=n['trait'].at[bed].set(jnp.where(land,n['seed_trait'][seed],n['trait'][bed]));n['growth']=n['growth'].at[bed].set(jnp.where(land,1,n['growth'][bed]));n['seed_active']=n['seed_active'].at[seed].set(n['seed_active'][seed]&~land);established=established+land.astype(jnp.int32)
        losses=jnp.sum((lost[None,:]&(n['seed_owner'][None,:]==jnp.arange(8,dtype=jnp.int32)[:,None])).astype(jnp.float32),axis=1)
        n['reward_parts']=n['reward_parts'].at[:,2].set(0.05*established.astype(jnp.float32)).at[:,4].add(-0.05*losses)
        n['terminated']=jnp.array(False);n['truncated']=n['t']>=self.max_steps;n['done']=n['truncated'];n=self.visibility(n)
        return n,jnp.sum(n['reward_parts'],axis=1,dtype=jnp.float32)

    def raw_step(self,key,state,actions):
        n,r=jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((8,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((8,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
