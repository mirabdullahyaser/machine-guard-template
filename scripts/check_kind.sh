#!/usr/bin/env bash
# Day 3 setup check for Docker and Kubernetes-in-Docker (kind). Instructor prep: run it in a
# test codespace before Day 3, not in the client's session.
# Run from the repo root:  bash scripts/check_kind.sh
# It creates a throwaway cluster, checks the node is Ready, then deletes it (also on failure).
set -euo pipefail

CLUSTER=smoke-test
PREVIOUS_CONTEXT=$(kubectl config current-context 2>/dev/null || true)

cleanup() {
  echo "== Cleaning up =="
  kind delete cluster --name "$CLUSTER" || true
  # kind switches kubectl to the throwaway cluster; switch back to any cluster that was in use.
  if [ -n "$PREVIOUS_CONTEXT" ] && [ "$PREVIOUS_CONTEXT" != "kind-$CLUSTER" ]; then
    kubectl config use-context "$PREVIOUS_CONTEXT" >/dev/null 2>&1 || true
  fi
}

echo "== Docker =="
docker run --rm hello-world >/dev/null && echo "Docker OK"

echo "== kind and kubectl versions =="
kind version
kubectl version --client

echo "== Creating cluster '$CLUSTER' (takes 1-3 minutes) =="
trap cleanup EXIT
kind create cluster --name "$CLUSTER" --wait 120s

echo "== Nodes =="
kubectl get nodes --context "kind-$CLUSTER"
kubectl wait --for=condition=Ready node --all --timeout=60s --context "kind-$CLUSTER"

echo "kind works. Day 3 Kubernetes lab is good to go."
