
<head>
       <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
       <script type="text/x-mathjax-config">
         MathJax.Hub.Config({
           tex2jax: {
             inlineMath: [ ['$','$'], ["\\(","\\)"] ],
             processEscapes: true
           }
         });
       </script>
       <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>


# Aim
The goal of this project is to explore the effects of using object properties and their interactions witheach other on a reinforcement learning game agent. The goal of the agent is to achieve maximumlifetime rewards in the environment it is playing in. Through the course of this project, we want to augment Deep RL game playing agents with more information about objects in the game and how they interact.

# Motivation
Due to advances in deep neural networks, we are able to solve many RL problems ranging from games like Chess, Go, Atari, StarCraft to real world robotics and control tasks. Unfortunately, deep RL models still face many issues including overfitting to the large training data. Detecting object representations and modeling their interactions in unsupervised way might allow deep RL models to generalize better on unseen but statistically similar test data.

# Problem Formulation

## The RL Framework
Given an environment that outputs an observation $\bf{O_t}$ and reward $\bf{R_t}$ at every time step , take an action $\bf{A_t}$ in the environment to maximize the total discounted lifetime reward accumulated in the environment. The discounted lifetime reward is defined as: 

$$
\begin{equation} \label{eq:1}
 l = \bf{R_t} + \gamma \bf{R_{t+1}} + \gamma^2 \bf{R_{t+2}} + .....
\end{equation}
$$

where $\gamma$ is the discount factor, $\bf{R_t}$ is the immediate reward for at time $t$ and $\bf{R_{t+1}}$ is the reward at time $t+1$ and so on.

We use Deep Neural Networks to learn object properties, interactions and predict optimal actions in the environment given a set of previous observations $\bf{O_{t-k}} ... \bf{O_{t}}$. So, at every time-step, our algorithm outputs the best action to take to maximize equation (1). Figure 1 describes our problem setup in a schematic way.

## Environment

Since we are designing a reinforcement learning agent, our data is derived from the environment. For the purpose of this project, we designed a new game environment for our game playing agentto play in. This environment hosts a simple game where balls collide with each other and the wall. There are five actions in the environment: no-op, left, right, up and down. No-op does no actionfor the time step and the other actions provide impulses to the ball controlled by network in thedirection of the action. To enable the network to identify itself, we differentiate the ball controlled by the network andthe other balls in the environment. The ball controlled by the network appears as a triangle whereasthe other agents appear as circles.  A screen grab of an environment with five agents is shown in the figure. 

We created 2 tasks in this environment to train out agents: 
1. In task 1, we encourage collisions with balls and discourage collisions with walls: collision with another ball earns a reward of +1 while a collision with the wall earns a reward of +1.
2. In task 2, we discourage collisions with balls and encourage collisions with walls: collision with another ball earns a reward of -1 while a collision with the wall earns a reward of +1.

# Past Works

## Q-Learning
Q-Learning is an off-policy reinforcement learning algorithm to find optimal q-values for state-action pairs. For a state $s$ and action $a$, q-value for the pair while following a policy $\pi$ is defined as the expected reward we achieve if we take action $a$ and follow the same policy $\pi$.

$$
\begin{equation} \label{eq:2}
 Q_\pi(s_t, a_t) = \sum_{i=t}^{i=\infty} r_i * \gamma^i
\end{equation}
$$

Optimal q-value is the best possible q-value we can achieve while following an optimal policy $\pi^\*$

$$
\begin{equation} \label{eq:3}
 Q^*(s_t, a_t) = \max_{\pi} Q_\pi(s_t, a_t)
\end{equation}
$$

Q-Learning updates the q-values of state-action pairs while following an exploratory policy which is why it is called off-policy learning algorithm. Precisely, the update is,

$$
\begin{equation} \label{eq:4}
Q(s_t, a_t) = (1 - \alpha) * Q(s_t, a_t) + \alpha (r_t + \gamma * \max_a Q(s_{t+1}, a)) 
\end{equation}
$$

where $\alpha$ is the learning rate.

## DQN
DQN is a neural network architecture that uses Q-Learning to find the optimal policy. The network takes in certain number of images and outputs the q-values for all the actions. To simulate Q-Learning update, we use two copies of neural networks that are synced periodically. Only one copy is trained and the weights of the other copy is kept frozen. In addition, we also maintain a replay memory $R$ that stores transitions, $(s_t, a_t, r_t, s_{t+1})$ tuples from recent episodes. This along with maintaining two copies of network, makes the updates more stable.

During training, we sample a mini-batch of transitions $(s, a, r, s^{'})$ uniformly from the replay memory and define a loss,

$$
\begin{equation} \label{eq:5}
L(\theta_i) = E_{(s, a, r, s^{'}) \sim U(R) } \lbrack (r + \gamma * \max{_{a^{'}}} Q(s^{'}, a^{'}; \theta^-_i) - Q(s, a; \theta) \rbrack^{2}
\end{equation}
$$

where $\theta_i$ and $\theta_i^-$ are parameters of training network and frozen network respectively.

## DRQN
DRQN is the recurrent version of DQN where we use a recurrent neural network instead of a feed forward network to compute the q-values. This helps the network by maintaining an internal state to keep track of the game. During game-play, the state is propagated until the end of the episode after which it is zero initialized at the start of the next episode. While training the network, we unroll the RNN for some fixed time steps which is smaller than the episode length and train the q-values for this sequence.

## RNEM
Relational Neural Expectation Maximization(RNEM) is based on Neural Expectation maximization(NEM), a neural network architecture that learns a separate distributed representation for each object described in terms of the same features through an iterative process of perceptual grouping and representation learning. In addition to NEM algorithm, RNEM also models interactions between objects efficiently.

The goal of NEM is to group pixels in the input that belong to the same object (perceptual grouping) and capture this information efficiently in a distributed representation $$\theta_{k}$$ for each object. At a high-level, the idea is that if we were to have access to the family of distributions $$P(x$$\|$$\theta_{k})$$ (a statistical model of images given object representations $$\theta_{k}$$) then we can formalize our objective as inference in a mixture of these distributions. By using Expectation Maximization to compute a Maximum Likelihood Estimate (MLE) of the parameters of this mixture $$(\theta_{1}, . . . , \theta_{K})$$, we obtain a grouping (clustering) of the pixels to each object (component) and their corresponding representation. 

NEM models each image $\boldsymbol{x} \in \mathbb{R}^{D}$ as a spatial mixture of $K$ components parameterized by vectors $\theta_{1}, . . . , \theta_{K} \in \mathbb{R}^{M}$. A neural network $f_{\phi}$ is used to transform these representations $\theta_{k}$ into parameters $$\psi_{i,k} = f_{\phi}(\theta_{k})_{i}$$ for separate pixel-wise distributions. A set of binary latent variables $$Z \in [0, 1]_{D\times K}$$ encodes the unknown true pixel assignments, such that $$z_{i,k} = 1$$ iff pixel $i$ was generated by component $k$.

The full likelihood for $$x$$ given $$\theta = (\theta_{1}, . . . , \theta_{K})$$ is given by:

$$
\begin{equation} \label{eq:6}
P(\boldsymbol{x} | \boldsymbol{\theta}) = \prod_{i=1}^{D} \sum_{k=1}^{K} P\left(z_{i, k}=1\right) P\left(x_{i} | \psi_{i, k}, z_{i, k}=1\right)
\end{equation}
$$

Marginalization over $z$ complicates this process, thus RNEM uses generalized EM to maximize the following lower bound instead:

$$
\begin{equation} \label{eq:7}
\mathcal{Q}\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text { old }}\right) = \sum_{\mathbf{z}} P\left(\mathbf{z} | \boldsymbol{x}, \boldsymbol{\psi}^{\text { old }}\right) \log P(\boldsymbol{x}, \mathbf{z} | \boldsymbol{\psi})
\end{equation}
$$

The unrolled computational graph of the generalized EM steps is differentiable, which provides a means to train $f_{\phi}$ to implement a statistical model of images given object representations. Using back-propagation through time, $f_{\phi}$ is trained to minimize the following loss:

$$
\begin{equation} \label{eq:8}
L(\boldsymbol{x})=-\sum_{i=1}^{D} \sum_{k=1}^{K} \underbrace{\gamma_{i, k} \log P\left(x_{i}, z_{i, k} | \psi_{i, k}\right)}_{\text { intra-cluster loss }} -\underbrace{\left(1-\gamma_{i, k}\right) D_{K L}\left[P\left(x_{i}\right) \| P\left(x_{i} | \psi_{i, k}, z_{i, k}\right)\right]}_{\text { inter-cluster loss }}
\end{equation}
$$

where $$\gamma_{i, k} =P $$ \( $$z_{i, k}=1$$ \| $$ x_{i}, \psi_{i}^{\text { old }}$$\) is calculated during E-step of the generalized EM algorithm.

Additionally, RNEM proposes a parametrized interaction function $$\Upsilon^{R-NEM}$$ that updates $$\theta_{k}$$ based on the pairwise effects of the objects $$i \neq k$$ on $$k$$:

$$
\boldsymbol{\theta}_{k}^{(t)}=\operatorname{RNN}\left(\tilde{\boldsymbol{x}}^{(t)}, \Upsilon_{k}^{\mathrm{R}-\mathrm{NEM}}\left(\boldsymbol{\theta}^{(t-1)}\right)\right)\\
\Upsilon_{k}^{\mathrm{R}-\mathrm{NEM}}(\boldsymbol{\theta})=\left[\hat{\boldsymbol{\theta}}_{k} ; \boldsymbol{E}_{k}\right] \text { with } \hat{\boldsymbol{\theta}}_{k}=\operatorname{MLP}^{e n c}\left(\boldsymbol{\theta}_{k}\right), \boldsymbol{E}_{k}=\sum_{i \neq k} \alpha_{k, i} \cdot \boldsymbol{e}_{k, i}\\
\alpha_{k, i}=\operatorname{MLP}^{a t t}\left(\xi_{k, i}\right), e_{k, i}=\operatorname{MLP}^{e f f}\left(\xi_{k, i}\right), \xi_{k, i}=\operatorname{MLP}^{e m b}\left(\left[\hat{\theta}_{k} ; \hat{\theta}_{i}\right]\right)\\
\text{where [·;·] is the concatenation operator and MLP(·) corresponds to a multi-layer perceptron.}
$$

# Approach

# Result

| Algorithm | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
| --- | --- | --- |
| DQN | <img left="400px" src="media/video-DQN-k-2.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-3.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-4.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-5.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-6.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-7.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-8.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-9.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DQN-k-10.gif" align="left" height="48" width="48" > | 
| DRQN | <img left="400px" src="media/video-DRQN-k-2.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-3.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-4.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-5.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-6.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-7.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-8.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-9.gif" align="left" height="48" width="48" > | <img left="400px" src="media/video-DRQN-k-10.gif" align="left" height="48" width="48" > | 



