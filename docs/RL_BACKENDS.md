# RL Training Backends

## Current: SKRL PPO (MLP)

- Task: `OmniReset-FFWSG2-PegMLPGraspLift-v0`
- Config: `configs/skrl_ppo_mlp_grasp_lift_96k.yaml`
- Multi-GPU: `scripts/tmux_train_grasp_lift_ddp.sh` (torchrun DDP, 2 GPUs, 30k envs)
- Observations: joint pos/vel + peg/hole poses in robot frame (vector, no pixels)

## Future

- SAC / TD3 for sample efficiency
- Vision-based policies (camera observations)
- Curriculum learning for grasp → insert sequence
