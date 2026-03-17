FROM mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        git \
        pkg-config \
        python3 \
        python3-pip \
        ripgrep \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g npm@latest @devcontainers/cli

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup target add wasm32-unknown-unknown \
    && cargo install wasm-pack

RUN mkdir -p /usr/local/cargo /usr/local/rustup /workspaces \
    && chown -R node:node /usr/local/cargo /usr/local/rustup /workspaces

USER node

EXPOSE 5173 4173
