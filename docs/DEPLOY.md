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

The **scene** (table, pin, hole) is **not duplicated** as USD inside this repo. Training and scripts rely on the existing **UWLab** task configs (e.g. `OmniReset-FFWSG2-PegPartialAssemblySmoke-v0`) and props on disk. Peg and hole USD must live under a local root with `Props/Custom/Peg/` and `Props/Custom/PegHole/` (see `scripts/run_on_tai.sh`: defaults to `~/uwlab_sync` on **tai** when present — **Hugging Face is not required**). Set `UWLAB_CLOUD_ASSETS_DIR` or `SG2_CLOUD_ASSETS_DIR` if your mirror path differs. Keep your `UWLab` checkout and Isaac Lab layout as today.

## SSH from `home-linux` to `10.225.68.32`

If you use a jump host, ensure your key is accepted on the inner host, for example:

```bash
ssh -A home-linux
ssh kaveh@10.225.68.32
```

Add `IdentityFile` / `ProxyJump` in `~/.ssh/config` as appropriate.
