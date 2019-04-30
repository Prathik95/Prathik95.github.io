<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/config/Accessible-full.js" type="text/javascript"></script>
{% $$Q_\pi(s_t, a_t) = \sum_{i=t}^{i=\infty} r_i * \gamma^i$$%}
## Aim
The goal of this project is to explore the effects of using object properties and their interactions witheach other on a reinforcement learning game agent. The goal of the agent is to achieve maximumlifetime rewards in the environment it is playing in. Through the course of this project, we want to augment Deep RL game playing agents with more information about objects in the game and how they interact.

## Motivation
Due to advances in deep neural networks, we are able to solve many RL problems ranging from games like Chess, Go, Atari, StarCraft to real world robotics and control tasks. Unfortunately, deep RL models still face many issues including overfitting to the large training data. Detecting object representations and modeling their interactions in unsupervised way might allow deep RL models to generalize better on unseen but statistically similar test data.

## Problem Formulation
Given an environment that outputs an observation ![observation](https://latex.codecogs.com/gif.latex?%5Cbf%7BO_t%7D) and reward \(\bf{R_t}\) at every time step \(t\), take an action \(\bf{A_t}\) in the environment to maximize the total discounted lifetime reward accumulated in the environment. The discounted lifetime reward is defined as: \[\label{eq:1} l = \bf{R_t} + \gamma \bf{R_{t+1}} + \gamma^2 \bf{R_{t+2}} + .....\] where \(\gamma\) is the discount factor \(\bf{R_t}\) is the immediate reward for at time \(t\) and \(\bf{R_{t+1}}\) is the reward at time \(t+1\) and so on.

We use Deep Neural Networks to learn object properties, interactions and predict optimal actions in the environment given a set of previous observations \(\bf{O_{t-k}} ... \bf{O_{t}}\). So, at every time-step, our algorithm outputs the best action to take to maximize equation ([\[eq:1\]](#eq:1)). Figure ([1](#fig:1)) describes our problem setup in a schematic way.

## Past Works and Baselines

## Approach - What is ROORL?

## Result
![Video](dqn_video.gif)

