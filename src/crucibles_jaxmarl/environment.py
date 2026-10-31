import jax
import jax.numpy as jnp

class Discrete:
    def __init__(self,n): self.n,self.shape,self.dtype=n,(),jnp.int32
    def sample(self,key): return jax.random.randint(key,(),0,self.n,dtype=self.dtype)
    def contains(self,x):
        x=jnp.asarray(x)
        return jnp.asarray(x.shape==()) & jnp.issubdtype(x.dtype,jnp.integer) & jnp.all((x>=0)&(x<self.n))

class Box:
    def __init__(self,low,high,shape):
        self.low=-jnp.inf if low is None else low
        self.high=jnp.inf if high is None else high
        self.shape,self.dtype=shape,jnp.float32
    def sample(self,key):
        lo,hi=jnp.asarray(self.low),jnp.asarray(self.high)
        # Unbounded dimensions use a standard normal, rather than uniform infinity.
        uniform=jax.random.uniform(key,self.shape,minval=jnp.where(jnp.isfinite(lo),lo,-1),maxval=jnp.where(jnp.isfinite(hi),hi,1))
        return jnp.where(jnp.isfinite(lo)&jnp.isfinite(hi),uniform,jax.random.normal(key,self.shape))
    def contains(self,x):
        x=jnp.asarray(x)
        return jnp.asarray(x.shape==self.shape)&jnp.all(jnp.isfinite(x)&(x>=self.low)&(x<=self.high))

class CrucibleEnv:
    """The original array kernel plus JaxMARL-style dictionary facade.

    raw_step/step_env preserve the original terminal state. step implements
    explicit outer autoreset. No PRNG, global mutable state or hidden RNG.
    """
    def __init__(self,kernel,manifest,metadata,scene=None):
        self.kernel,self.manifest,self.metadata=kernel,manifest,metadata
        self.scene=scene or {}
        self.name=metadata['key']
        for k in ['num_agents','observation_size','num_actions','max_steps']:setattr(self,k,manifest[k])
        self.agents=[a['id'] for a in manifest['agents']]
        self.agent_classes={}
        for a in manifest['agents']:self.agent_classes.setdefault(a['role'],[]).append(a['id'])
        self.action_spaces={a:Discrete(self.num_actions) for a in self.agents}
        self.observation_spaces={a:Box(manifest.get('observation_low'),manifest.get('observation_high'),(self.observation_size,)) for a in self.agents}
    def _dict(self,x):return {a:x[i] for i,a in enumerate(self.agents)}
    def action_space(self,a):return self.action_spaces[a]
    def observation_space(self,a):return self.observation_spaces[a]
    def reset(self,key):
        obs,state=self.kernel.reset(key)
        return self._dict(obs),state
    def observe(self,state):return self.kernel.observe(state)
    def get_obs(self,state):return self._dict(self.observe(state))
    def legal_actions(self,state):return self.kernel.legal_actions(state)
    def get_avail_actions(self,state):return self._dict(self.legal_actions(state))
    def sample_actions(self,key,state):
        # Gumbel-max over legal actions; avoids sampling illegal discrete commands.
        mask=self.legal_actions(state)
        logits=jnp.where(mask,0.,-jnp.inf)
        return self._dict(jax.random.categorical(key,logits,axis=-1).astype(jnp.int32))
    def raw_step(self,key,state,actions):return self.kernel.raw_step(key,state,actions)
    def step_env(self,key,state,actions):
        array=jnp.stack([actions[a] for a in self.agents]).astype(jnp.int32)
        obs,state,rewards,dones,info=self.raw_step(key,state,array)
        return self._dict(obs),state,self._dict(rewards),dict(self._dict(dones),__all__=info['done_all']),info
    def step(self,key,state,actions,reset_state=None):
        transition_key,reset_key=jax.random.split(key)
        obs,raw,rewards,dones,info=self.step_env(transition_key,state,actions)
        fresh_obs,fresh=self.reset(reset_key) if reset_state is None else (self.get_obs(reset_state),reset_state)
        choose=lambda new,old:jax.lax.select(dones['__all__'],new,old)
        return (jax.tree.map(choose,fresh_obs,obs),jax.tree.map(choose,fresh,raw),rewards,dones,
                dict(info,raw_state=raw,raw_observation=obs))
