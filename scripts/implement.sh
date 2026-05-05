#!/usr/bin/env bash
set -euo pipefail

PRD="/workspace/Ford_Dock 360/scripts/prd.json"
REPO="/workspace/Ford_Dock 360"

while true; do
  STORY_ID=$(python3 -c "
import json
with open('$PRD') as f:
    prd = json.load(f)
pending = [s for s in prd['userStories'] if not s['passes']]
print(pending[0]['id'] if pending else 'DONE')
")

  if [ "$STORY_ID" = "DONE" ]; then
    echo "Todas as stories implementadas!"
    break
  fi

  PROMPT=$(python3 -c "
import json
with open('$PRD') as f:
    prd = json.load(f)
s = next(s for s in prd['userStories'] if s['id'] == '$STORY_ID')
criteria = '\n'.join(f'- {c}' for c in s['acceptanceCriteria'])
print(f\"Implemente a story {s['id']}: {s['title']}\n\n{s['description']}\n\nCriterios de aceite:\n{criteria}\n\nApos implementar, confirme o que foi criado.\")
")

  echo "=========================================="
  echo "Implementando: $STORY_ID"
  echo "=========================================="

  echo "$PROMPT" | codex exec \
    -s danger-full-access \
    -m gpt-5.5 \
    -C "$REPO" \
    --output-last-message "/tmp/last_message.txt" \
    - \
    2>&1

  # Marca como passou
  python3 -c "
import json
with open('$PRD') as f:
    prd = json.load(f)
for s in prd['userStories']:
    if s['id'] == '$STORY_ID':
        s['passes'] = True
with open('$PRD', 'w') as f:
    json.dump(prd, f, indent=2, ensure_ascii=False)
print('Marcado como completo: $STORY_ID')
"

  sleep 2
done
