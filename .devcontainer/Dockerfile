# The codespace's base image, with one fix.
# The Python dev-container image ships an apt source for Yarn (a JavaScript tool we don't use) whose
# signing key no longer verifies. Any `apt-get update` then fails, which breaks the Docker-in-Docker
# feature and the whole codespace build. Removing that source before the features install fixes it.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn*
