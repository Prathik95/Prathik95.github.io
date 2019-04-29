## Aim
The goal of this project is to explore the effects of using object properties and their interactions witheach other on a reinforcement learning game agent.  The goal of the agent is to achieve maximumlifetime rewards in the environment it is playing in.  Through the course of this project, we want toaugment Deep RL game playing agents with more information about objects in the game and howthey interact.

## Motivation
Due to advances in deep neural networks,  we are able to solve many RL problems ranging fromgames  like  Chess,  Go,  Atari,  StarCraft  to  real  world  robotics  and  control  tasks.   Unfortunately,deep RL models still face many issues including over-fitting to the large training data.  Detectingobject representations and modeling their interactions in unsupervised way might allow deep RLmodels to generalize better on unseen but statistically similar test data.

## Problem Formulation
Given an environment that outputs an observation $\bf{O_t}$ and reward
$\bf{R_t}$ at every time step $t$, take an action $\bf{A_t}$ in the
environment to maximize the total discounted lifetime reward accumulated
in the environment. The discounted lifetime reward is defined as:
$$\label{eq:1}
 l = \bf{R_t} + \gamma \bf{R_{t+1}} + \gamma^2 \bf{R_{t+2}} + .....$$
where $\gamma$ is the discount factor $\bf{R_t}$ is the immediate reward
for at time $t$ and $\bf{R_{t+1}}$ is the reward at time $t+1$ and so
on.

## Past Works and Baselines

## Approach - What is ROORL?

## Result
![Video](dqn_video.gif)

