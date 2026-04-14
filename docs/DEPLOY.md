# Deploying SG2-RL to `10.225.68.32`

This repository was created locally under `Projects/API/SG2-RL` because automated SSH to `10.225.68.32` from the agent environment is not available (VPN / host key / key permissions).

## One-time: create directory on the server

From a shell that **can** reach the machine (your laptop on VPN, or `home-linux` with agent forwarding):

```bash
ssh kaveh@10.225.68.32 "mkdir -p ~/projects/API/SG2-RL"
```

## Sync the repo

With **`ssh 32`** working (see your Mac `~/.ssh/config`):

```bash
rsync -avz --delete \
  /Users/kavehkamali/Projects/API/SG2-RL/ \
  32:~/projects/API/SG2-RL/
```

Or clone: push `SG2-RL` to a git remote from this machine, then `git clone` on `10.225.68.32` into `~/projects/API/SG2-RL`.

## Scene source

The **scene** (table, pin, hole) is **not duplicated** as USD inside this repo. Training and scripts rely on the existing **UWLab** task configs (e.g. `OmniReset-FFWSG2-PegPartialAssemblySmoke-v0`) and assets shipped with `uwlab_tasks` / `uwlab_assets`. Keep your existing `UWLab` checkout on the server and `PYTHONPATH` / Isaac Lab install layout unchanged.

## SSH from `home-linux` to `10.225.68.32`

If you use a jump host, ensure your key is accepted on the inner host, for example:

```bash
ssh -A home-linux
ssh kaveh@10.225.68.32
```

Add `IdentityFile` / `ProxyJump` in `~/.ssh/config` as appropriate.
