#!/bin/bash
# setup-hub.sh — one-time Ensue hub initialization for autoresearch swarm.
# Usage: ENSUE_API_KEY=<key> ./scripts/setup-hub.sh [experiment.go]
set -euo pipefail

API="${ENSUE_API_URL:-https://api.ensue-network.ai/}"
KEY="${ENSUE_API_KEY:?Set ENSUE_API_KEY}"
ORG="travis_cline"
SEED="${1:-experiment.go}"

rpc() {
  local tool="$1" args="$2"
  curl -sf -X POST "$API" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args},\"id\":1}"
}

share() {
  rpc "share" "{\"command\":$1}"
}

step=0
progress() { step=$((step+1)); echo "[$step] $1"; }

# 1. Create auto-approve invite link.
progress "Creating auto-approve invite link"
INV=$(rpc "create_invite" '{"auto_approve":true}')
TOKEN=$(echo "$INV" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('result',{}).get('token',''))" 2>/dev/null || echo "")

# 2. Create participants group.
progress "Creating 'participants' group"
share '{"command":"create_group","group_name":"participants"}'

# 3. Set external auto-join group.
progress "Setting external auto-join to 'participants'"
share '{"command":"set_external_group","group_name":"participants"}'

# 4. Grant permissions for both workload namespaces.
for wl in infer train; do
  for ns in "claims/" "results/" "hypotheses/" "insights/" "best/" "leaderboard"; do
    perms="read create"
    if [ "$ns" = "best/" ] || [ "$ns" = "leaderboard" ]; then
      perms="read create update"
    fi
    for perm in $perms; do
      progress "Granting $perm on $wl/$ns to participants"
      share "{\"command\":\"grant\",\"target\":{\"type\":\"group\",\"group_name\":\"participants\"},\"action\":\"$perm\",\"key_pattern\":\"@${ORG}/${wl}/${ns}\"}"
    done
  done
done

# 5. Make public keys.
for wl in infer train; do
  for pat in "leaderboard" "best/" "results/"; do
    progress "Making $wl/$pat public"
    share "{\"command\":\"make_public\",\"key_pattern\":\"@${ORG}/${wl}/${pat}\"}"
  done
done

# 6. Seed keys.
if [ -f "$SEED" ]; then
  SEED_B64=$(base64 < "$SEED")
  progress "Seeding best/experiment_go"
  rpc "create_memory" "{\"items\":[{\"key_name\":\"@${ORG}/train/best/experiment_go\",\"description\":\"[autoresearch] Current best experiment.go source code\",\"value\":\"$SEED_B64\",\"base64\":true}]}" > /dev/null

  META_B64=$(echo -n "{\"val_bpb\":null,\"val_loss\":null,\"steps\":null,\"train_secs\":null,\"status\":\"baseline\",\"description\":\"seed\",\"seeded_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | base64)
  progress "Seeding best/metadata"
  rpc "create_memory" "{\"items\":[{\"key_name\":\"@${ORG}/train/best/metadata\",\"description\":\"[autoresearch] Metadata for current best experiment.go\",\"value\":\"$META_B64\",\"base64\":true}]}" > /dev/null

  LB_B64=$(echo -n "{\"entries\":[],\"updated_at\":null}" | base64)
  progress "Seeding leaderboard"
  rpc "create_memory" "{\"items\":[{\"key_name\":\"@${ORG}/train/leaderboard\",\"description\":\"[autoresearch] Leaderboard for ANE training benchmarks\",\"value\":\"$LB_B64\",\"base64\":true}]}" > /dev/null
else
  echo "WARN: seed file '$SEED' not found, skipping seed keys"
fi

# 7. Summary.
echo ""
echo "--- Hub initialized ---"
echo "  Org:    $ORG"
echo "  API:    $API"
echo "  Token:  $TOKEN"
if [ -n "$TOKEN" ]; then
  echo "  Invite: https://ensue-network.ai/join?token=$TOKEN"
fi
