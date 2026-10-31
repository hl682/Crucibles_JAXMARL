"""Washwater Bistro: a two-worker kitchen with conserved bowls and dirty flow."""
import jax
import jax.numpy as jnp


class GeneratedEnv:
    num_agents=2
    observation_size=190
    num_actions=16
    max_steps=256
    obs_keys=('t','done','terminated','truncated','terrain','agent_pos','carrying','tool_pos','tool_holder','station_pos','item_pos','item_active','kind','holder','clean','prepared','filled','age','spoiled','channel_pos','flow','tank','sink_dirt','draining','divert','basin','filter_installed','filter_charge','pan_dirty','grease','order_pos','order_status','patience','served','wasted','contaminations','reward_parts')

    def reset(self,key):
        s={
            't':jnp.array(0,jnp.int32),'done':jnp.array(False),'terminated':jnp.array(False),'truncated':jnp.array(False),
            'terrain':jnp.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1,1,1,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],jnp.int32),
            'agent_pos':jnp.array([[5,3],[2,3]],jnp.int32),'agent_active':jnp.ones((2,),jnp.bool_),'agent_visible':jnp.ones((2,2),jnp.bool_),'carrying':jnp.full((2,),-1,jnp.int32),
            'tool_pos':jnp.array([[5,3],[2,3]],jnp.int32),'tool_active':jnp.ones((2,),jnp.bool_),'tool_visible':jnp.ones((2,2),jnp.bool_),'tool_holder':jnp.array([0,1],jnp.int32),
            'station_pos':jnp.array([[5,2],[2,2],[1,4],[5,4],[4,2],[1,1],[4,4]],jnp.int32),'station_active':jnp.ones((7,),jnp.bool_),'station_visible':jnp.ones((2,7),jnp.bool_),
            'item_pos':jnp.array([[4,3],[3,3],[6,1],[6,2],[6,3],[6,4],[1,3]],jnp.int32),
            'item_active':jnp.ones((7,),jnp.bool_),'item_visible':jnp.ones((2,7),jnp.bool_),
            'kind':jnp.array([0,0,1,1,1,1,2],jnp.int32),'holder':jnp.full((7,),-1,jnp.int32),
            'clean':jnp.zeros((7,),jnp.bool_),'prepared':jnp.zeros((7,),jnp.bool_),'filled':jnp.zeros((7,),jnp.bool_),'age':jnp.zeros((7,),jnp.int32),'spoiled':jnp.zeros((7,),jnp.bool_),
            'channel_pos':jnp.array([[3,2],[4,2],[5,2],[6,2]],jnp.int32),'channel_active':jnp.ones((4,),jnp.bool_),'channel_visible':jnp.ones((2,4),jnp.bool_),'flow':jnp.zeros((4,),jnp.int32),
            'tank':jnp.array(8,jnp.int32),'sink_dirt':jnp.array(0,jnp.int32),'draining':jnp.array(False),'divert':jnp.array(False),'basin':jnp.array(0,jnp.int32),'filter_installed':jnp.array(False),'filter_charge':jnp.array(3,jnp.int32),'pan_dirty':jnp.array(False),'grease':jnp.array(0,jnp.int32),
            'order_pos':jnp.array([[2,5],[3,5],[4,5],[5,5]],jnp.int32),'order_active':jnp.ones((4,),jnp.bool_),'order_visible':jnp.ones((2,4),jnp.bool_),'order_status':jnp.zeros((4,),jnp.int32),'patience':jnp.array([55,95,135,175],jnp.int32),
            'served':jnp.array(0,jnp.int32),'wasted':jnp.array(0,jnp.int32),'contaminations':jnp.array(0,jnp.int32),'reward_parts':jnp.zeros((2,4),jnp.float32)
        }
        return self.observe(s),s

    def observe(self,s):
        flat=jnp.concatenate([s[k].ravel().astype(jnp.float32) for k in self.obs_keys])
        return jnp.broadcast_to(flat,(2,self.observation_size))

    def legal_actions(self,s):
        p=s['agent_pos'];d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32);q=p[:,None,:]+d[None,:,:]
        move=jnp.all((q>=0)&(q<=jnp.array([6,5],jnp.int32)),axis=2)&(s['terrain'][jnp.clip(q[:,:,1],0,5),jnp.clip(q[:,:,0],0,6)]==0)
        near=jnp.sum(jnp.abs(p[:,None,:]-s['station_pos'][None,:,:]),axis=2)<=1
        cook=s['tool_holder'][0]==jnp.arange(2,dtype=jnp.int32)
        adj=jnp.sum(jnp.abs(p[0]-p[1]))<=1
        extra=jnp.stack((s['carrying']<0,s['carrying']>=0,(cook&near[:,0])|(~cook&(near[:,0]|near[:,1])),near[:,2],near[:,1],near[:,4],near[:,4],cook&near[:,0],near[:,3],near[:,5],jnp.full((2,),adj,jnp.bool_)),axis=1)
        return (jnp.concatenate((move,extra),axis=1)&~s['done']).at[:,0].set(True)

    def advance(self,s,actions):
        n=dict(s);ac=jnp.clip(actions,0,15);ac=jnp.where((actions>=0)&(actions<16)&self.legal_actions(s)[jnp.arange(2,dtype=jnp.int32),ac],ac,0)
        d=jnp.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]],jnp.int32);target=s['agent_pos']+d[jnp.minimum(ac,4)]*((ac>0)&(ac<5))[:,None]
        other=~jnp.eye(2,dtype=jnp.bool_);block=jnp.any(jnp.all(target[:,None,:]==s['agent_pos'][None,:,:],axis=2)&other,axis=1)|jnp.any(jnp.all(target[:,None,:]==target[None,:,:],axis=2)&other,axis=1)
        n['agent_pos']=jnp.where(block[:,None],s['agent_pos'],target)
        swap=jnp.any(ac==15)&(jnp.sum(jnp.abs(n['agent_pos'][0]-n['agent_pos'][1]))<=1)
        n['tool_holder']=jnp.where(swap,1-s['tool_holder'],s['tool_holder'])
        served=jnp.array(0,jnp.int32);waste=jnp.array(0,jnp.int32)
        for k in range(2):
            i=(s['t']+k)%2;p=n['agent_pos'][i];a=ac[i];cook=n['tool_holder'][0]==i
            held=n['holder']==i;has=jnp.any(held);h=jnp.argmax(held).astype(jnp.int32)
            loose=n['item_active']&(n['holder']==-1)&(jnp.sum(jnp.abs(n['item_pos']-p),axis=1)<=1);j=jnp.argmax(loose).astype(jnp.int32)
            pick=(a==5)&~has&jnp.any(loose);drop=(a==6)&has
            n['holder']=n['holder'].at[j].set(jnp.where(pick,i,n['holder'][j])).at[h].set(jnp.where(drop,-1,jnp.where(pick&(j==h),i,n['holder'][h])))
            n['item_pos']=n['item_pos'].at[h].set(jnp.where(drop,p,n['item_pos'][h]))
            n['filter_installed']=n['filter_installed']&~(pick&(j==6))
            near=jnp.sum(jnp.abs(n['station_pos']-p),axis=1)<=1
            prep=(a==7)&cook&near[0]&has&(n['kind'][h]==1)&~n['spoiled'][h]&~n['prepared'][h]&~n['pan_dirty']&(n['grease']<3)&(n['tank']>0)
            wash=(a==7)&~cook&near[1]&has&(n['kind'][h]==0)&~n['filled'][h]&(n['tank']>0)&(n['sink_dirt']<3)
            scrub=(a==7)&~cook&near[0]&~has&(n['tank']>=2)&(n['pan_dirty']|(n['grease']>0))
            n['prepared']=n['prepared'].at[h].set(n['prepared'][h]|prep)
            n['clean']=n['clean'].at[h].set(n['clean'][h]|wash)
            n['sink_dirt']=jnp.minimum(3,n['sink_dirt']+wash.astype(jnp.int32)+scrub.astype(jnp.int32))
            n['pan_dirty']=n['pan_dirty']&~scrub;n['grease']=jnp.where(scrub,0,n['grease'])
            n['tank']=n['tank']-prep.astype(jnp.int32)-wash.astype(jnp.int32)-2*scrub.astype(jnp.int32)
            cleanfilter=(a==8)&near[2]&has&(h==6)&(n['tank']>=2)
            pump=(a==8)&near[2]&~(has&(h==6))
            n['filter_charge']=jnp.where(cleanfilter,3,n['filter_charge'])
            n['tank']=jnp.minimum(12,n['tank']+2*pump.astype(jnp.int32)-2*cleanfilter.astype(jnp.int32))
            n['draining']=n['draining']^((a==9)&near[1]);n['divert']=n['divert']^((a==10)&near[4])
            install=(a==11)&near[4]&has&(h==6)
            n['filter_installed']=n['filter_installed']|install
            n['holder']=n['holder'].at[6].set(jnp.where(install,-1,n['holder'][6]));n['item_pos']=n['item_pos'].at[6].set(jnp.where(install,n['station_pos'][4],n['item_pos'][6]))
            ingredients=n['item_active']&(n['kind']==1)&n['prepared']&~n['spoiled']&(n['holder']==-1)&(jnp.sum(jnp.abs(n['item_pos']-n['station_pos'][0]),axis=1)<=1)
            ing=jnp.argmax(ingredients).astype(jnp.int32)
            assemble=(a==12)&cook&near[0]&has&(n['kind'][h]==0)&n['clean'][h]&~n['filled'][h]&~n['pan_dirty']&(n['tank']>0)&jnp.any(ingredients)
            n['item_active']=n['item_active'].at[ing].set(n['item_active'][ing]&~assemble);n['holder']=n['holder'].at[ing].set(jnp.where(assemble,-2,n['holder'][ing]))
            n['filled']=n['filled'].at[h].set(n['filled'][h]|assemble);n['age']=n['age'].at[h].set(jnp.where(assemble,0,n['age'][h]));n['spoiled']=n['spoiled'].at[h].set(n['spoiled'][h]&~assemble)
            n['tank']=n['tank']-assemble.astype(jnp.int32);n['grease']=jnp.minimum(3,n['grease']+assemble.astype(jnp.int32))
            pending=(n['order_status']==0)&(n['patience']>0);order=jnp.argmax(pending).astype(jnp.int32)
            deliver=(a==13)&near[3]&has&n['filled'][h]&n['clean'][h]&~n['spoiled'][h]&jnp.any(pending)
            n['order_status']=n['order_status'].at[order].set(jnp.where(deliver,1,n['order_status'][order]));served=served+deliver.astype(jnp.int32)
            discard=(a==14)&near[5]&has&(n['kind'][h]<2)
            waste=waste+(discard&((n['kind'][h]==1)|n['filled'][h])).astype(jnp.int32)
            resetbowl=(deliver|discard)&(n['kind'][h]==0)
            n['filled']=n['filled'].at[h].set(n['filled'][h]&~resetbowl);n['clean']=n['clean'].at[h].set(n['clean'][h]&~resetbowl);n['spoiled']=n['spoiled'].at[h].set(n['spoiled'][h]&~resetbowl)
            n['holder']=n['holder'].at[h].set(jnp.where(deliver,-1,jnp.where(discard&(n['kind'][h]==1),-2,n['holder'][h])))
            n['item_pos']=n['item_pos'].at[h].set(jnp.where(deliver,n['station_pos'][3],n['item_pos'][h]))
            n['item_active']=n['item_active'].at[h].set(n['item_active'][h]&~(discard&(n['kind'][h]==1)))
        emission=n['draining']&(n['sink_dirt']>0);n['sink_dirt']=n['sink_dirt']-emission.astype(jnp.int32)
        shifted=jnp.concatenate((emission.astype(jnp.int32)[None],s['flow'][:3]))
        filtered=(shifted[1]>0)&n['filter_installed']&(n['filter_charge']>0)
        n['filter_charge']=n['filter_charge']-filtered.astype(jnp.int32)
        diverted=(shifted[1]>0)&~filtered&n['divert']&(n['basin']<3)
        n['basin']=jnp.maximum(0,n['basin']+diverted.astype(jnp.int32)-(s['t']%12==11).astype(jnp.int32))
        n['flow']=shifted.at[1].set(jnp.where(filtered|diverted,0,shifted[1]))
        contamination=(n['flow'][2]>0)&~n['pan_dirty'];n['pan_dirty']=n['pan_dirty']|(n['flow'][2]>0)
        n['contaminations']=s['contaminations']+contamination.astype(jnp.int32)
        aging=n['item_active']&((n['kind']==1)|n['filled']);n['age']=jnp.minimum(90,n['age']+aging.astype(jnp.int32))
        spoiled=n['item_active']&(((n['kind']==1)&(n['age']>=90))|(n['filled']&(n['age']>=18)))
        waste=waste+jnp.sum((spoiled&~n['spoiled']).astype(jnp.int32),dtype=jnp.int32);n['spoiled']=n['spoiled']|spoiled
        n['patience']=jnp.maximum(0,n['patience']-(n['order_status']==0).astype(jnp.int32));n['order_status']=jnp.where((n['patience']==0)&(n['order_status']==0),-1,n['order_status'])
        n['order_active']=n['order_status']==0;n['order_visible']=jnp.broadcast_to(n['order_active'],(2,4))
        n['served']=s['served']+served;n['wasted']=s['wasted']+waste
        n['item_pos']=jnp.where((n['holder']>=0)[:,None],n['agent_pos'][jnp.clip(n['holder'],0,1)],n['item_pos']);n['tool_pos']=n['agent_pos'][n['tool_holder']]
        match=n['holder'][None,:]==jnp.arange(2,dtype=jnp.int32)[:,None];n['carrying']=jnp.where(jnp.any(match,axis=1),jnp.argmax(match,axis=1),-1).astype(jnp.int32)
        n['item_visible']=jnp.broadcast_to(n['item_active'],(2,7));n['t']=s['t']+1
        n['terminated']=jnp.all(n['order_status']!=0);n['truncated']=(n['t']>=self.max_steps)&~n['terminated'];n['done']=n['terminated']|n['truncated']
        n['reward_parts']=jnp.stack((jnp.full((2,),6.0*served,jnp.float32),jnp.full((2,),-0.75*contamination.astype(jnp.float32),jnp.float32),jnp.full((2,),-waste.astype(jnp.float32),jnp.float32),-0.02*(ac>=7).astype(jnp.float32)),axis=1)
        return n,jnp.sum(n['reward_parts'],axis=1,dtype=jnp.float32)

    def raw_step(self,key,state,actions):
        n,r=jax.lax.cond(state['done'],lambda _: (state,jnp.zeros((2,),jnp.float32)),lambda _: self.advance(state,actions),operand=None)
        return self.observe(n),n,r,jnp.full((2,),n['done'],jnp.bool_),{'done_all':n['done'],'terminated':n['terminated'],'truncated':n['truncated']}
