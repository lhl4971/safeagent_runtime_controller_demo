FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_NO_MODIFY_PATH=1
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
ENV VIBESHELL_WORKSPACE=/home/ubuntu/workspace

# Restore standard Ubuntu utilities and documentation
RUN yes | unminimize

# Install base system packages, SSH, Python, and common utilities
RUN apt update && apt install -y \
    sudo curl wget git vim nano unzip zip htop tree procps lsof \
    net-tools iproute2 iputils-ping dnsutils telnet rsync \
    openssh-server ca-certificates bash-completion \
    python3 python3-pip python3-venv \
 && apt clean && rm -rf /var/lib/apt/lists/*

# Configure passwords and sudo access for the existing ubuntu user
RUN echo 'root:U3FsDR5sJTi' | chpasswd && \
    echo 'ubuntu:ys8k1RzDvAz' | chpasswd && \
    usermod -aG sudo ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu && \
    chmod 440 /etc/sudoers.d/ubuntu

# Configure SSH: disable root login, enable password authentication
RUN mkdir -p /var/run/sshd && \
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?UsePAM.*/UsePAM yes/' /etc/ssh/sshd_config

# Install uv using the official installer
RUN su - ubuntu -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Create application, workspace, and input directories
RUN mkdir -p /home/ubuntu/mcp /home/ubuntu/workspace /input /opt/bootstrap && \
    touch /home/ubuntu/mcp/vibe_shell.log && \
    chown -R ubuntu:ubuntu /home/ubuntu/mcp /home/ubuntu/workspace /input /opt/bootstrap

# Create a virtual environment with uv and install fastmcp into it
RUN su - ubuntu -c '/home/ubuntu/.local/bin/uv venv /home/ubuntu/mcp/.venv' && \
    su - ubuntu -c '/home/ubuntu/.local/bin/uv pip install --python /home/ubuntu/mcp/.venv/bin/python fastmcp'

# Copy a default application file into the image as a fallback
COPY --chown=ubuntu:ubuntu mcp/vibe_shell.py /opt/bootstrap/vibe_shell.py

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ubuntu
WORKDIR /home/ubuntu/mcp

EXPOSE 22 8000

CMD ["/entrypoint.sh"]