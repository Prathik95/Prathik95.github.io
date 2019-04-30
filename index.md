
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

\begin{equation} \label{eq:1}
 l = \bf{R_t} + \gamma \bf{R_{t+1}} + \gamma^2 \bf{R_{t+2}} + .....
\end{equation}

where $\gamma$ is the discount factor, $\bf{R_t}$ is the immediate reward for at time $t$ and $\bf{R_{t+1}}$ is the reward at time $t+1$ and so on.

We use Deep Neural Networks to learn object properties, interactions and predict optimal actions in the environment given a set of previous observations $\bf{O_{t-k}} ... \bf{O_{t}}$. So, at every time-step, our algorithm outputs the best action to take to maximize equation (1). Figure 1 describes our problem setup in a schematic way.

## Environment

Since we are designing a reinforcement learning agent, our data is derived from the environment. For the purpose of this project, we designed a new game environment for our game playing agentto play in. This environment hosts a simple game where balls collide with each other and the wall. There are five actions in the environment: no-op, left, right, up and down. No-op does no actionfor the time step and the other actions provide impulses to the ball controlled by network in thedirection of the action. To enable the network to identify itself, we differentiate the ball controlled by the network andthe other balls in the environment. The ball controlled by the network appears as a triangle whereasthe other agents appear as circles.  A screen grab of an environment with five agents is shown in the figure. 

We created 2 tasks in this environment to train out agents: 
1. In task 1, we encourage collisions with balls and discourage collisions with walls: collision with another ball earns a reward of +1 while a collision with the wall earns a reward of +1.
2. In task 2, we discourage collisions with balls and encourage collisions with walls: collision with another ball earns a reward of -1 while a collision with the wall earns a reward of +1.

# Past Works and Baselines

# Approach - What is ROORL?

# Result
![Video](dqn_video.gif)

