# GPU runner (Docker)

Runs the self-hosted GitHub Actions runner for GPU jobs inside a container on the lab node.

> **No Docker access?** (the DGX Spark `spark`): use `native.sh` instead. It runs the
> runner as your own user with labels `gpu,spark` and writes the job environment
> to the runner's `.env`:
> `RUNNER_TOKEN=<token> bash scripts/gpu-runner/native.sh setup && bash ~/actions-runner/native.sh start`.
> On aarch64, `gpu-env.sh` installs torch from PyTorch's CUDA 13 index
> (`EVOLVETS_TORCH_SPEC`, default `torch==2.9.1`) and everything else
> hash-checked from `uv.lock`. Forecasts built there are not bit-identical to
> sp-1's, so compare runs only against caches built on the same node.

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
  -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v evolve-runner:/home/runner/actions-runner \
  -v evolve-data:/data \
  evolve-gpu-runner
docker logs -f evolve-gpu-runner   # wait for "Listening for Jobs", then Ctrl-C
```

`evolve-data` holds model weights, uv's package cache and built forecast caches, so they survive rebuilds. The two mirror variables are passed on to every job (drop them outside China).

### Updating the image later

The registration lives in the `evolve-runner` volume, so no new token is needed:

```bash
cd ~/projects/evolve && git pull
docker build -t evolve-gpu-runner <same --build-arg flags as in step 2> scripts/gpu-runner
docker rm -f evolve-gpu-runner
docker run ...   # the step 3 command, without -e RUNNER_TOKEN
```

The registration is stored in the `evolve-runner` volume, so restarts and reboots don't need a new token. `--restart unless-stopped` brings it back after a reboot.

## Day to day

- Stop: `docker stop evolve-gpu-runner`. Start again: `docker start evolve-gpu-runner`.
- Logs: `docker logs --tail 50 evolve-gpu-runner`.
- Re-register from scratch: `docker rm -f evolve-gpu-runner && docker volume rm evolve-runner`, then step 3 with a new token.

## How free GPUs are detected inside the container

`gpu-pick.sh` uses `nvidia-smi` memory use and utilization, which reflect every process on the host, including other users'. A container can't see the PIDs of host processes, so the memory and utilization thresholds do the work. A GPU anyone is using shows memory well above the 1 GiB threshold. Per-GPU locks prevent two EvolveTS jobs from sharing a card.

## If jobs can't download the code

GPU workflows fetch the repo with a quick `git fetch` and fall back to a tarball from `api.github.com` (which redirects to `codeload.github.com`). If both time out, check from inside the container:

```bash
docker exec evolve-gpu-runner curl -sI -m 15 https://github.com | head -1
docker exec evolve-gpu-runner curl -sI -m 15 https://codeload.github.com | head -1
```

If these fail but the same `curl` works on sp-1 itself, recreate the container with the host's network by adding `--network host` to the `docker run` above (the registration in the `evolve-runner` volume is kept). If the lab has an HTTP proxy, add `-e https_proxy=http://<proxy>:<port> -e no_proxy=localhost,127.0.0.1` instead; the runner passes it on to every job.

## The repository is public: keep the runner locked down

A self-hosted runner on a public repo can be targeted by pull requests from forks. Rules:

- Settings -> Actions -> General -> Fork pull request workflows: **Require approval for all outside collaborators**. Never approve a fork PR's run unless you have read every changed file, including `.github/`.
- No workflow that runs on `[self-hosted, gpu]` may be triggered by `pull_request`, `pull_request_target`, `issues` or `issue_comment`. Only `workflow_dispatch` (owner-only) is allowed.
- Keep the runner stopped when it is not building: `docker stop evolve-gpu-runner`, and `docker start evolve-gpu-runner` before a GPU job.
