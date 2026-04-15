# Deploy to tai

SG2-RL is self-contained. Clone and set up:

```bash
cd ~/projects/API
git clone git@github.com:kavehkamali/SG2-RL.git
cd SG2-RL
~/.local/bin/uv venv .venv --python 3.10
~/.local/bin/uv pip install -e '.[dev]'
~/.local/bin/uv pip install 'isaacsim[all]==4.5.0' --extra-index-url https://pypi.nvidia.com
~/.local/bin/uv pip install 'isaaclab==2.0.0' --extra-index-url https://pypi.nvidia.com
```

## Sync from laptop

```bash
rsync -avz --exclude .venv --exclude artifacts --exclude __pycache__ \
  ~/projects/API/SG2-RL/ tai-32:~/projects/API/SG2-RL/
```

Or use git push/pull.

## Run

```bash
./scripts/run_on_tai.sh smoke_random_motion.py --headless --steps 32
./scripts/run_on_tai.sh record_path_apf_follow_gripper.py --headless --video_length 2160
```
