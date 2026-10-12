# Self-hosted GitHub Actions runner with GPU access, for the shared lab node.
# Job code runs inside this container as an unprivileged user, so it cannot
# see the host's home directories. GPUs come in through --gpus.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq tar gzip xz-utils python3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash runner
WORKDIR /home/runner/actions-runner

# Latest official runner release at build time (actions/runner on GitHub).
RUN version="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name | sed 's/^v//')" \
    && curl -fsSL -o runner.tgz "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-x64-${version}.tar.gz" \
    && tar xzf runner.tgz && rm runner.tgz \
    && ./bin/installdependencies.sh \
    && chown -R runner:runner /home/runner

COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
USER runner
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/runner/.local/bin:${PATH}"

ENTRYPOINT ["bash", "/home/runner/entrypoint.sh"]
