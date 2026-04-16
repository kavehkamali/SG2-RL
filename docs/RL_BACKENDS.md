# RL Training Backends

## Current: SKRL PPO (MLP)

- Tasks:
  - Stage1: `FFWSG2-PegGraspLift-v0`
  - Stage2: `FFWSG2-PegInsert-v0`
- Configs:
  - `configs/skrl_ppo_mlp_stage1_grasp_lift.yaml`
  - `configs/skrl_ppo_mlp_stage2_insert.yaml`
- Multi-GPU: `scripts/tmux_train_ppo_32768_ddp.sh` (torchrun DDP, 2 GPUs, 32,768 envs)
- Observations: joint pos/vel + peg/hole poses in robot frame (vector, no pixels)

## Future

- SAC / TD3 for sample efficiency
- Vision-based policies (camera observations)
- Curriculum learning for grasp → insert sequence
