# 强化学习训练接口

接口采用按角色索引的多智能体约定，是独立库，不代表上游认证。每款说明书另有具体目标、动作、机制假说和训练注意事项。

## 重置与一步转移

```python
import jax
from crucibles_jaxmarl import make

env = make("gi072")
key = jax.random.PRNGKey(0)
key, rk, ak, sk = jax.random.split(key, 4)
obs, state = jax.jit(env.reset)(rk)
actions = env.sample_actions(ak, state)
obs, state, rewards, dones, info = jax.jit(env.step_env)(sk, state, actions)
done = dones["__all__"]
```

观察为角色字典，每个值是原生浮点向量；动作是每人一个整型编号，掩码长度等于动作数。所有角色同一步提交。合法采样只是工程参照，替换为自己的策略时仍须保持角色观察权限。

## 终止语义

`step_env` 不自动重置，保留原终止观察和状态。`step` 明确拆分转移与重置密钥，提供外层自动重置，在 `info['raw_state']` 和 `info['raw_observation']` 留存终止数据。用整局结束标记划分轨迹，同时区分自然完成、失败与时限。空间内核保留吸收终止，抽象内核保留历史行为。

## 编译与批处理

```python
keys = jax.random.split(jax.random.PRNGKey(1), 4)
obs, states = jax.jit(jax.vmap(env.reset))(keys)
keys = jax.random.split(jax.random.PRNGKey(2), 4)
actions = jax.vmap(env.sample_actions)(keys, states)
keys = jax.random.split(jax.random.PRNGKey(3), 4)
obs, states, rewards, dones, info = jax.jit(jax.vmap(env.step))(keys, states, actions)
```

只批处理同身份同形状，示例批量不替代原始人口或时限。不同形状明确分组，或声明学习器侧填充方案。重置、策略采样和转移使用独立密钥。渲染在主机侧，位于训练编译图外。

## 空间、掩码与原生数组

```python
env.agents
env.agent_classes
env.action_space("agent_0")
env.observation_space("agent_0")
env.get_avail_actions(state)
env.get_obs(state)
env.observe(state)
env.legal_actions(state)
env.raw_step(key, state, action_array)
env.manifest
env.metadata
```

固定数据划分和选择规则，保留原奖励，报告逐角色结果及成本。集中价值网络、记忆、通信、奖励塑形和伙伴适应须作为明确实验因素。冒烟检查不能证明收敛、迁移、元梯度正确或离散动力学可微。
