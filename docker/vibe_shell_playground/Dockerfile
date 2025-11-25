FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 使用 unminimize 让系统恢复 man / locale / help 等内容
RUN yes | unminimize

# 安装常用开发工具与 SSH 服务
RUN apt update && apt install -y \
    sudo vim nano curl wget git unzip zip htop net-tools iproute2 iputils-ping \
    lsb-release ca-certificates software-properties-common build-essential cmake gdb make \
    pkg-config python3 python3-pip python3-venv clang g++ \
    openssh-server rsync netcat-traditional dnsutils telnet tree lsof procps sysstat iotop bash-completion plocate \
 && updatedb \
 && apt clean && rm -rf /var/lib/apt/lists/*

# 创建 SSH 目录并允许密码登录
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 确保 ubuntu 用户存在并能 sudo 免密
RUN usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && chmod 440 /etc/sudoers.d/ubuntu

# SSH 自启动命令（防止容器退出）
CMD ["/usr/sbin/sshd", "-D"]

# 暴露常用端口
EXPOSE 22 8000