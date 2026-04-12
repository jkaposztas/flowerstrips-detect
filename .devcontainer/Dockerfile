
FROM nvcr.io/nvidia/pytorch:24.10-py3

ARG DEBIAN_FRONTEND=noninteractive

# System packages (adjust as needed)
RUN apt-get update && \
    apt-get install -y sudo screen s3cmd gdal-bin libgdal-dev openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Python tooling
RUN pip install --upgrade pip

# (Optional) Older setuptools to avoid GDAL build issues (keep only if you actually hit the error)
# RUN pip install setuptools==57.5.0

# Copy and install Python requirements from workspace root (build context must include this file)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Avoid overriding PyTorch versions included in the base image unless necessary
# If you must override, ensure CUDA/ABI compatibility and uncomment:
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Create non-root user (match host user for good file permissions)
# Adjust USERNAME, USER_UID, and USER_GID as needed
ARG USERNAME=jkaposztas
ARG USER_UID=1009
ARG USER_GID=1009

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
