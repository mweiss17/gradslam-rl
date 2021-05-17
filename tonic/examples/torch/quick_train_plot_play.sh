#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

## Quick distributed training.
#python3 -m tonic.train \
#    --header "import tonic.torch" \
#    --agent "tonic.torch.agents.PPO(replay=tonic.replays.Segment(size=10, batch_size=2000, batch_iterations=5))" \
#    --environment "tonic.environments.ControlSuite('quadruped-walk')" \
#    --trainer "tonic.Trainer(epoch_steps=100, steps=500000, save_steps=50000)" \
#    --parallel 10 \
#    --sequential 100 \
#    --name "PPO-torch-demo-quadruped-walk" \
#    --seed 1
#

python3 -m tonic.train \
    --header "import tonic.torch" \
    --agent "tonic.torch.agents.PPO(replay=tonic.replays.Segment(size=10, batch_size=2000, batch_iterations=10))" \
    --environment "tonic.environments.ControlSuite('humanoid_CMU-run')" \
    --trainer "tonic.Trainer(epoch_steps=100, steps=50000000, save_steps=50000)" \
    --parallel 10 \
    --sequential 100 \
    --name "PPO-torch-demo-humanoid_CMU-run" \
    --seed 1

#  python3 -m tonic.train \
#    --header "import tonic.torch" \
#    --agent "tonic.torch.agents.PPO(replay=tonic.replays.Segment(size=10, batch_size=2000, batch_iterations=30))" \
#    --environment "tonic.environments.Gym('LunarLanderContinuous-v2')" \
#    --trainer "tonic.Trainer(epoch_steps=100, steps=500000, save_steps=500000)" \
#    --parallel 10 \
#    --sequential 100 \
#    --name "PPO-torch-demo" \
#    --seed 0
#

# Plot and reload.
#python3 -m tonic.plot --path quadruped-walk --baselines all &
python3 -m tonic.play --path humanoid_CMU-run/PPO-torch-demo-humanoid_CMU-run/1 --save-name humanoid_CMU.mp4 &

#python3 -m tonic.play --path quadruped-walk/PPO-torch-demo-quadruped-walk/0 --save-name movie.mp4 &
wait
