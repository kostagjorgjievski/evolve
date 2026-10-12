# GPU runner (Docker)

Runs the self-hosted GitHub Actions runner for GPU jobs inside a container on the lab node.

## 1. Check that Docker can see the GPUs

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

If Docker Hub is unreachable (common on Chinese campus networks), use whichever of these sp-1 can pull, and pass the same name as `BASE_IMAGE` in step 2:

- `nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04` (NVIDIA's own registry)
- `docker.m.daocloud.io/nvidia/cuda:12.4.1-base-ubuntu22.04` (a Docker Hub mirror)

This should list every GPU. If it fails with "could not select device driver", the NVIDIA Container Toolkit isn't installed on the host and the cluster admin needs to add it.

## 2. Build the image

```bash
cd ~/projects/evolve && git pull
docker build -t evolve-gpu-runner scripts/gpu-runner
# behind the Great Firewall, for example:
docker build -t evolve-gpu-runner \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
  scripts/gpu-runner
```

## 3. Start it (once)

Get a registration token from GitHub: repo Settings → Actions → Runners → New self-hosted runner. Copy only the token value from the `./config.sh ... --token XXXX` line. Then:

```bash
docker run -d --name evolve-gpu-runner --restart unless-stopped \
  --gpus all \
  -e RUNNER_TOKEN=<token> \
  -v evolve-runner:/home/runner/actions-runner \
  evolve-gpu-runner
docker logs -f evolve-gpu-runner   # wait for "Listening for Jobs", then Ctrl-C
```

The registration is stored in the `evolve-runner` volume, so restarts and reboots don't need a new token. `--restart unless-stopped` brings it back after a reboot.

## Day to day

- Stop: `docker stop evolve-gpu-runner`. Start again: `docker start evolve-gpu-runner`.
- Logs: `docker logs --tail 50 evolve-gpu-runner`.
- Re-register from scratch: `docker rm -f evolve-gpu-runner && docker volume rm evolve-runner`, then step 3 with a new token.

## How free GPUs are detected inside the container

`gpu-pick.sh` uses `nvidia-smi` memory use and utilization, which reflect every process on the host, including other users'. A container can't see the PIDs of host processes, so the memory and utilization thresholds do the work. A GPU anyone is using shows memory well above the 1 GiB threshold. Per-GPU locks prevent two EvolveTS jobs from sharing a card.
