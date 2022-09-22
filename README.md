# PLAYER KNOWN’S BATTLEGROUNDS

This is a course project of Algorithm Design and Analysis cooperate with [Zhehuan Chen](https://acmlczh.github.io/). (May 2021 - June 2021)

Generally speaking, we designed the enviroment and task that contain two agant with competition.
We want to see what interesting strategies they will learn by implementing TD3 reinforcement learning for both agent.

## Environment build

* The two players lie on a limited 2D playground (square with the side length of 10 units) and each player is a circle (with the radius of 0.5 units).
* Player 1 is able to shoot bullets, while Player 2 trying to avoid them.
* The bullets are with fixed velocities and we only allow at most 8 bullets on the playground.
  (with means that player1 can shoot the next bullet only when one of the bullets goes out of the playground)
  
* Player 1 wins if one of the bullets hits Player 2 with in 100 timestep. Player2 wins otherwise.
* To simplify the setting, Player 1 will shoot whenever it can (less than 8 bullets on the playground, 3 timestep after it shoots the last bullets.). So it only has to control the shooting direction.
* Other details could be found in our code and gifs.

## Algorithms

* reinforcement learning
* adversarial learning
* behavior cloning

## Rewards

* if the players fall out of the playground, they will receive negative rewards and then fix their position at the nearest point on the playground to continue the game.
* if the distance between a bullet and Player 2 is less than 2 unit, Player 2 will receive a negative reward while Player 1 will receive a positive one (depends on the distance).
* every timestep that Player 2 being alive, it will receive a positive reward.
* the winner get positive reward, the losser get negative reward.

## Training

### Step 1 behavior cloning

We fail to directly train these two agent adversarially, and it would be a waste of resources. So we wrote an expert for Player 1 and apply behavior clone to start off the training.
In detail, we first BC our expert with actor in TD3, then we fix the actor and train the critic solely network until coverage. And this is the result we get after BC the expert of Player 1.

### Step 2 Train Player 2 solely

After 250 epoch training, Player 2 was able to win 90 percents of games against the expert we build.

![result_BC1](/pic/imit1.gif)
![result_player2](/pic/imit2.gif)

### Step 3 adversarial training

In this step, we got suprising results for both agents.

![result_train1](/pic/train1.gif)
![result_train2](/pic/train2.gif)
![result_train3](/pic/train3.gif)
![result_train4](/pic/train4.gif)
![result_train5](/pic/train5.gif)


