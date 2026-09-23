# Serving image for ftplatform. Needs an NVIDIA GPU at `docker run` time
# (--gpus all) -- vLLM has no meaningful CPU-only mode for this workload.
#
# This image installs the package; it does not decide what to run. There is
# no single sane default command (profile/customer/model are deployment-
# specific), so CMD just shows the CLI's own help -- override it explicitly,
# e.g.:
#   docker run --gpus all -p 8000:8000 -v $(pwd)/customers.db:/app/customers.db \
#     ftplatform ftplatform serve --profile saas_support --customer-id acme \
#     --model outputs/saas_support/lora_adapter
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Dependencies before source, so an app-only change doesn't reinstall vLLM.
COPY pyproject.toml README.md ./
COPY ftspec ./ftspec
COPY ftplatform ./ftplatform
COPY profiles ./profiles
RUN pip install --no-cache-dir -e ".[serve,billing]"

COPY configs ./configs

EXPOSE 8000
CMD ["ftplatform", "--help"]
