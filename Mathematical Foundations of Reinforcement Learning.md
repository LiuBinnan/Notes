# Ch1 Basic Concepts
A *trajectory* is a state-action-reward chain.

The *return* of this trajectory is defined as the sum of all the rewards collected along the trajectory. Returns are also called *total rewards* or *cumulative rewards*.

为了避免无限长的 trajectory 使得 return 发散，因此引入折扣回报 *discounted return*，并称 $\gamma \in (0, 1)$ 为折扣因子 *discount rate*。

根据 policy 与环境进行交互时，agent 可能会停止在一些 terminal states，对应的 trajectory 被称为 *episode*。

马尔科夫决策过程包含状态空间 $\mathcal{S}$，动作空间 $\mathcal{A}(s)$，奖励集合 $\mathcal{R}(s, a)$，状态转移概率 $p(s^\prime \vert s, a)$，奖励概率 $p(r \vert s, a)$，策略 $\pi(a \vert s)$。其具有马尔可夫性质：下一个状态或奖励仅取决于当前的状态和动作，与之前的状态与动作无关。

Q: Is the reward a function of the next state?

A: “We mentioned that the reward r depends only on s and a but not the next state s′.  However, this may be counterintuitive since it is the next state that determines the  reward in many cases.”。奖励 $r$ 取决于 $s, a$ 和 $s^\prime$，因为 $s^\prime$ 也取决于 $s, a$，因此可以将 $r$ 改写为 $s$ 和 $a$ 的函数：$p(r \vert s, a) = \sum_{s^\prime}p(r \vert s, a, s^\prime)p(s^\prime \vert s, a)$

# Ch2 State Values and Bellman Equation
## State values
记时间序列 $t = 0, 1, 2, \cdots$，在 $t$ 时刻，agent 处于状态 $S_t$，根据策略 $\pi$ 采取的动作为 $A_t$，下一个状态为 $S_{t+1}$，获得的即时奖励为 $R_{t+1}$，从 $t$ 时刻起，可以得到 state-action-reward trajectory：

$$S_t \xrightarrow{A_t} S_{t+1}, R_{t+1} \xrightarrow{A_{t+1}} S_{t+2}, R_{t+2} \xrightarrow{A_{t+2}} \cdots$$

得到的折扣回报为

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

由于 $G_t$ 是一个随机变量，因此可以计算其期望：

$$v_{\pi}(s) = \mathbb{E}[G_t \vert S_t = s]$$

其中 $v_{\pi}(s)$ 称为状态价值函数 *state-value function*，或者说状态 $s$ 的状态价值 *state value*。

## Bellman equation
$G_t$ 可以写为

$$\begin{aligned}
&G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
& = R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots) \\
& = R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$

然后状态价值也可以重写为

$$\begin{aligned}
&v_{\pi}(s) = \mathbb{E}[G_t \vert S_t = s] \\
&~~~~~~~~~ = \mathbb{E}[R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&~~~~~~~~~ = \mathbb{E}[R_{t+1} \vert S_t = s] + \gamma\mathbb{E}[G_{t+1} \vert S_t = s] 
\end{aligned}
$$

按顺序来分析这两项。$\mathbb{E}[R_{t+1} \vert S_t = s]$ 是即时奖励的期望，可以计算为

$$\begin{aligned}
&\mathbb{E}[G_t \vert S_t = s] = \sum_{a \in \mathcal{A}}\pi(a \vert s)\mathbb{E}[G_t \vert S_t = s, A_t = a]\\
&~~~~~~~~~~~~~~~~~~~~~ = \sum_{a \in \mathcal{A}}\pi(a \vert s)\sum_{r \in \mathcal{R}}p(r \vert s, a) r \\
\end{aligned}
$$

$\mathbb{E}[G_{t+1} \vert S_t = s]$ 是未来奖励的期望，可以写为

$$\begin{aligned}
&\mathbb{E}[G_{t+1} \vert S_t = s]  = \sum_{s^\prime \in \mathcal{S}}\mathbb{E}[G_{t+1} \vert S_t = s, S_{t+1} = s^\prime]p(s^\prime \vert s)\\
&~~~~~~~~~~~~~~~~~~~~~~~~ = \sum_{s^\prime \in \mathcal{S}}\mathbb{E}[G_{t+1} \vert S_{t+1} = s^\prime]p(s^\prime \vert s) \\
&~~~~~~~~~~~~~~~~~~~~~~~~ = \sum_{s^\prime \in \mathcal{S}} v_{\pi}(s^\prime)p(s^\prime \vert s) \\
&~~~~~~~~~~~~~~~~~~~~~~~~ = \sum_{s^\prime \in \mathcal{S}} v_{\pi}(s^\prime) \sum_{a \in \mathcal{A}}p(s^\prime \vert s, a)\pi(a \vert s)
\end{aligned}
$$

将上面两项带入式子中就有

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/12/jhNOguG8IRotXBK.png"  width="80%" />
</div>    

上式就是贝尔曼方程 *Bellman equation*。

## Matrix-vector form of the Bellman equation
可以将贝尔曼方程重写为

$$v_{\pi}(s) = r_{\pi}(s) + \gamma\sum_{s^\prime \in \mathcal{S}}p_{\pi}(s^\prime \vert s)v_{\pi}(s^\prime)$$

其中

$$r_{\pi}(s)  = \sum_{a \in \mathcal{A}}\pi(a \vert s)\sum_{r \in \mathcal{R}}p(r \vert s, a)r$$

$$p_{\pi}(s^\prime \vert s) = \sum_{a \in \mathcal{A}}\pi(a \vert s)p(s^\prime \vert s, a)$$

将每个状态的状态价值都写为以上形式即可得到矩阵形式的贝尔曼方程

$$v_\pi = r_\pi + \gamma P_\pi v_\pi$$

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/12/jIeO6Fb9q2pQmVW.png"  width="100%" />
</div>    

贝尔曼方程一定存在解析解

$$v_\pi = (I  - \gamma P_\pi)^{-1}r_\pi$$

但通常会使用迭代的方法计算数值解

## From state value to action value
动作价值 *action value* 可以定义为

$$q_\pi(s, a) = \mathbb{E}[G_t \vert S_t = s, A_t = a]$$

可以得到 

$$v_\pi(s) = \sum_{a \in \mathcal{A}}\pi(a \vert s)q_\pi(s, a)$$

# Ch3 Optimal State Values and Bellman  Optimality Equation

强化学习的核心目标是找到最优策略，最优策略意味着在每个状态 $s$，其状态价值均大于等于其他策略的状态价值。因此只要为每个状态 $s$，为其找到策略 $\pi(s)$ 使得 $v(s)$ 最大，就能得到最优策略。求解这一问题的工具就是贝尔曼最优方程 Bellman optimality equation (BOE)

$$v(s) = \max_{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}}\pi(a \vert s)q(s, a)$$

上式有 2 个未知变量 $v(s), \pi(a \vert s)$，可以写求解 $\pi(a \vert s)$ 再求解 $v(s)$ 解决这一问题。对于 $\pi(s)$，我们可以证明当其选择具有最大值的 $q(s, a)$ 时其就是最优策略。求解 $\pi(s)$ 后上述等式就变为了 $v = f(v)$ 的形式，根据 contraction mapping theorem，这一等式的解存在且唯一，并且可以通过迭代的方式得到。
因此，寻找最优策略 -> 每个状态的状态价值最大 -> 求解 BOE -> 选择拥有最大动作价值的策略 $\pi(s)$ -> 求解 $v = f(v)$ -> 解存在且唯一。

# Ch4 Value Iteration and Policy Iteration
## Value iteration
value iteration 的思想基本与上一章的思想一致，首先初始化 $v(s)$ 的值，然后计算动作价值，并选取产生最大动作价值的动作作为策略，根据得到的策略更新状态价值，进行新一轮的计算。

$$v_k(s) \rightarrow q_k(s, a) \rightarrow \text{new greedy policy } \pi_{k+1}(s) \rightarrow \text{new value } v_{k+1}(s) \rightarrow \max_{a}q_k(s, a)$$

value iteration 的每一次迭代分为 2 步，第一步是 policy update，根据前面一次迭代的结果 $v_k$ 寻找策略

$$\pi_{k+1} = \arg \max_{\pi}(r_\pi + \gamma P_\pi v_k)$$

第二步是 value update，根据得到的策略计算新的 $v_{k+1}$

$$v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}}v_k$$

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/13/LQKyItqxma53FDA.png"  width="100%" />
</div>    

需要注意的是，中间结果 $v_k$ 并不是真正的状态价值，因为其不一定满足 $v_k = r_{\pi_k} + \gamma P_{\pi_k}v_k$ 或 $v_k = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}}v_k$，这是由于我们是通过 $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}}v_k$ 计算得到的，这不是贝尔曼方程，因此求解的结果也不是状态函数。
## Policy iteration
Policy iteration 的每一次迭代也分为 2 步，第一步是 policy evaluation，通过求解状态价值的方式评估策略

$$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k}v_{\pi_k}$$

这里 $\pi_k$ 是上一次迭代的结果，这里我们求解的是真正的状态价值，这是与 value iteration 不同的地方。

第二步是 policy improvement，基于 $v_{\pi_k}$ 得到新的策略 $\pi_{k+1}$

$$\pi_{k+1} = \arg \max_{\pi} (r_\pi + \gamma P_\pi v_{\pi_k})$$

这一步与 value iteration 的 policy update 相似。

可以证明如果从同一个初始值开始，Policy iteration 会比 value iteration 收敛得更快。原因应该在于，Policy iteration 每一步都要求解真正的状态价值，而 value iteration 只计算一次结果就进行下一步的迭代。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/16/Dx1mRrYWongJcue.png"  width="100%" />
</div>   

## Truncated policy iteration

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/16/TBSVjGFJz1MshYQ.png"  width="70%" />
</div>   

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/16/kQgnzEiAZHb2BUq.png"  width="100%" />
</div>   

可以看到 Policy iteration 与 value iteration 是非常相似，不同之处就在于“状态价值”的计算，value iteration 仅进行一步计算，而 policy iteration 需要无限次数的迭代以得到真正的状态价值。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/16/2SXKFan7LDgwfeP.png"  width="70%" />
</div>   

因此可以在两者之间做一个 trade-off，进行有限次数的迭代求解，这就是 Truncated policy iteration

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/16/LRO4DBZkrnCj8Xc.png"  width="100%" />
</div>   

# Monte Carlo Methods

上一章的方法是基于 system model 来得到最优策略，我们希望使用 model-free 的方法来得到最优策略。方法就是利用与环境交互得到的数据来代替 system model。

## MC Basic: The simplest MC-based algorithm

首先回顾 policy iteration，第一步 policy evaluation 计算每个状态的状态价值，第二步 policy improvement 利用状态价值计算动作价值，根据动作价值决定策略。对于动作价值的计算

$$q_{\pi_k}(s,a) = \sum_rp(r\vert s, a)r + \gamma\sum_{s^\prime}p(s^\prime \vert s, a)v_{\pi_k}(s^\prime)$$

需要知道 system model $\{p(r\vert s, a), p(s^\prime \vert s, a)\}$

如果使用 model-free 的方法，可以通过定义计算

$$q_{\pi_k}(s, a) = \mathbb{E}[G_t \vert S_t = s, A_t = a]$$

对于期望的计算，可以用蒙特卡洛方法，获得 n 个 episodes，然后用平均值来近似期望。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/19/C9s6K42VecTfgd1.png"  width="100%" />
</div>   

MC Basic 算法与 policy iteration 很相似，避免了 policy evaluation 中状态价值的计算，直接计算动作价值用于 policy improvement。

## MC Exploring Starts

MC Basic 算法非常简单，但其效率不高。对于一个采样得到的 episode

$$s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \cdots$$

完整的一个 episode 仅仅是为了近似初始状态的状态-动作对 $(s_1, a_2)$ 的动作价值，称为 initial visit，这样子效率是非常低的，因为后续也访问了许多其他的状态-动作对，我们也可以为其近似动作价值

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/19/cP49nBvOfNesYlJ.png"  width="80%" />
</div>   

具体来说，可以将一个 episode 拆分为多个 subepisode，每个 subepisode 都可以为对应的状态-动作对近似动作价值。

一个状态-动作对可能在 episode 中重复出现 (如 $(s_1, a_2)$)，如果仅仅计算其第一次出现的动作价值，称为 first-visit，如果记录每个状态-动作对，称为 every-visit。"If the rest of the episode is used to estimate the action value of a state-action pair every time it is visited, such a strategy is called every-visit. If we only count the first time a state-action pair is visited in the  episode, such a strategy is called first-visit."

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/19/xiOtnHo6P4rZcj5.png"  width="100%" />
</div>   

这里采用了 every-visit 的策略，得到一个 (sub)episode 的中的一个状态价值后就更新策略。这样的方法需要保证 episode 涵盖尽可能多的状态-动作对，这样才能正确地近似动作价值。

## MC $\epsilon$ -Greedy: Learning without exploring starts

MC Exploring Starts 算法需要满足每个状态-动作对都能被访问到一定次数，可以通过随机性来做到。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/19/LOB4zF3aW2vcC7w.png"  width="60%" />
</div>   

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/19/1PCxvy3VXSJYfk2.png"  width="100%" />
</div>  

# Ch7 Temporal-Difference Methods

TD 算法表达式如下，目标是为策略 $\pi$ 估计所有状态的状态价值 $v_{\pi}(s)$

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/xPfMymlzIEa9u5q.png"  width="65%" />
</div>  

## TD learning of action values: Sarsa

给定策略 $\pi$，我们的目标是估计动作价值，可以通过以下 Sarsa 算法估计：

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))]$$

因为每一轮迭代需要 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$，所以称这个算法为 Sarsa 。Sarse 算法实际上就是求解给定策略的贝尔曼方程的随机近似算法

$$q_{\pi}(s, a) = \mathbb{E}[R + \gamma q_{\pi}(S^\prime, A^\prime) \vert s, a], \text{  for all } (s, a)$$

由于 Sarsa 估计了给定策略的动作价值，可以与 policy improve 结合以得到最优策略。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/EFJBVIbDCZ34OWs.png"  width="100%" />
</div>  

Sarsa 还有名为 Expected Sarsa 的变种，就是将公式改为

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \mathbb{E}[q_t(s_{t+1}, A)] )]$$

这样可以减少估计的方差，因为将随机变量由 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ 缩减到了 $(s_t, a_t, r_{t+1}, s_{t+1})$

## TD learning of action values: n-step Sarsa

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/aMSYAVGLKEu7I3O.png"  width="65%" />
</div>  

如图所示，Sarsa 和 MC learning 可以看作是 n-step Sarsa 的两个极端情况

## TD learning of optimal action values: Q-learning

Q-learning 算法如下

$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})}q_t(s_{t+1}, a) )]$$

Q-learning 求解的是贝尔曼最优方程，可以估计最优动作价值，直接得到最优策略

## Off-policy vs on-policy

behavior policy 是指用于生成样本的策略，target policy 是指经常更新以趋近于最优策略的策略。当 behavior policy 和 target policy 是同一个策略时，称学习过程是 on-policy 的，如果是不同的策略，则称为 off-policy。
和 on-policy/off-policy 需要区分的概念是 online/offline，online learning 是指智能体在与环境交互时更新价值与策略，offline learning 是指智能体通过预先收集好的数据来更新价值与策略，并没有与环境发生交互。
Sarsa 是 on-policy 的，而 Q-learning 是 off-policy 的。这是因为 Sarsa 是在求解策略的贝尔曼方程，Q-learning 直接求解贝尔曼最优方程，可以直接得到最优策略而不需要频繁更新策略。
off-policy 的算法既可以以 on-policy 的形式实现也可以用 off-policy 的形式实现。

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/TuADnFGRMEkSt8w.png"  width="100%" />
</div>  

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/izcOn2Yut5XoAkP.png"  width="100%" />
</div>  

## Summary

几类算法的区别在于 TD target 的选择

<div  align="center">    
	<img src="https://s2.loli.net/2025/06/23/1XdZ7V3FxTefKhH.png"  width="75%" />
</div>  

Q: While Theorems 7.1 and 7.2 require that the learning rate $\alpha_t$ converges to zero gradually, why is it often set to be a small constant in practice?

A: **The fundamental reason is that the policy to be evaluated keeps changing (or called nonstationary).** In particular, a TD learning algorithm like Sarsa aims to estimate the action values of a given policy. If the policy is fixed, using a decaying learning rate is acceptable. However, in the optimal policy learning process, the policy that  Sarsa aims to evaluate keeps changing after every iteration. We need a constant learning rate in this case; otherwise, a decaying learning rate may be too small to  effectively evaluate policies. Although a drawback of constant learning rates is that the value estimate may fluctuate eventually, the fluctuation is neglectable as long as  the constant learning rate is sufficiently small.
