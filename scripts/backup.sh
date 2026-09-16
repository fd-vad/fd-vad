#!/bin/bash
# Crash-resilience backup. Usage: bash scripts/backup.sh "optional commit message"
#  - code + results + docs + STATE -> GitHub (heavy artifacts gitignored)
#  - all checkpoints (latest.pt + config + metrics) -> sensei-fs
#  - all data manifests + reports -> sensei-fs (audio is regenerable via prepare_smartturn.py)
#  - code tarball snapshot -> sensei-fs
set -uo pipefail
REPO=/home/colligo/semanticVAD
SENSEI=/sensei-fs/users/puneetm/semanticVAD_backup
CKPTS=/mnt/localssd/svad/checkpoints
DATA=/mnt/localssd/svad/data/smartturn_en
SECRETS=/home/colligo/secrets.txt
SP=/tmp/claude-1000/-home-colligo/b7c882bc-e3c5-4f41-8a9b-dbe0c13fce7d/scratchpad
cd "$REPO"

# 1) git commit + push
MSG="${1:-}"
if [ -n "$MSG" ] && [ -n "$(git status --porcelain)" ]; then
  git add -A && git commit -q -m "$MSG" && echo "[backup] committed: $MSG"
fi
GH_PAT=$(grep 'github cli:' "$SECRETS" | sed 's/.*github cli: *//')
cat > "$SP/askpass.sh" <<EOF
#!/bin/bash
case "\$1" in *Username*) echo "puneetm_adobe";; *Password*) echo "$GH_PAT";; esac
EOF
chmod +x "$SP/askpass.sh"
GIT_ASKPASS="$SP/askpass.sh" GIT_TERMINAL_PROMPT=0 git push origin main 2>&1 | grep -vi "$GH_PAT" | tail -2

# 2) checkpoints -> sensei (latest.pt + small files only; skip .tmp)
mkdir -p "$SENSEI/checkpoints"
for d in "$CKPTS"/*/; do
  name=$(basename "$d")
  [ -f "$d/latest.pt" ] || continue
  mkdir -p "$SENSEI/checkpoints/$name"
  cp -u "$d"/latest.pt "$d"/config.yaml "$d"/*.json "$SENSEI/checkpoints/$name/" 2>/dev/null
done
echo "[backup] checkpoints synced: $(ls "$SENSEI/checkpoints" | tr '\n' ' ')"

# 3) data manifests + reports -> sensei (light; audio regenerable)
mkdir -p "$SENSEI/data"
for d in "$DATA"/*/; do
  name=$(basename "$d")
  mkdir -p "$SENSEI/data/$name"
  cp -u "$d"/manifest.jsonl "$d"/report.json "$d"/rejects.json "$SENSEI/data/$name/" 2>/dev/null
done
echo "[backup] data manifests synced"

# 4) code tarball snapshot
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$SENSEI/code_snapshots"
tar czf "$SENSEI/code_snapshots/code_$TS.tar.gz" \
  --exclude='.git' --exclude='*_local' --exclude='__pycache__' \
  --exclude='*.egg-info' --exclude='full_duplex_audio_generation/*/audio' \
  -C /home/colligo semanticVAD 2>/dev/null
ln -sfn "code_$TS.tar.gz" "$SENSEI/code_snapshots/latest.tar.gz"
echo "[backup] code snapshot: code_$TS.tar.gz | done $(date)"
