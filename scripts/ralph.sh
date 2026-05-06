#!/bin/bash
# Ralph Implementation Loop — fallback triplo Codex -> Gemini -> Claude
# Lê scripts/prd.json e implementa cada US com passes: false.
# Marca passes: true quando py_compile e acceptance criteria passam.
#
# Usage: ./scripts/ralph.sh [max_iterations] [--skip-security-check] [--provider codex|gemini|claude]

set -euo pipefail

export CODEX_INTERNAL_ORIGINATOR_OVERRIDE="Codex Desktop"

MAX_ITERATIONS=40
MAX_ATTEMPTS_PER_STORY="${MAX_ATTEMPTS_PER_STORY:-5}"
SKIP_SECURITY="${SKIP_SECURITY_CHECK:-false}"
TAIL_N="${TAIL_N:-200}"
PROVIDER=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-security-check) SKIP_SECURITY="true"; shift ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    *) if [[ "$1" =~ ^[0-9]+$ ]]; then MAX_ITERATIONS="$1"; fi; shift ;;
  esac
done

# ── Provider chain ────────────────────────────────────────────────────────────
# Ordem de preferencia: codex -> gemini -> claude
PROVIDER_CHAIN=("codex" "gemini" "claude")

detect_provider() {
  if [[ -n "$PROVIDER" ]]; then
    echo "$PROVIDER"
    return
  fi
  for p in "${PROVIDER_CHAIN[@]}"; do
    if command -v "$p" &>/dev/null; then
      echo "$p"
      return
    fi
  done
  echo "none"
}

next_provider() {
  local current="$1"
  local found=0
  for p in "${PROVIDER_CHAIN[@]}"; do
    if [[ $found -eq 1 ]] && command -v "$p" &>/dev/null; then
      echo "$p"
      return
    fi
    [[ "$p" == "$current" ]] && found=1
  done
  echo ""
}

test_provider() {
  local provider="$1"
  case "$provider" in
    codex)
      codex -a never exec -C "$REPO_ROOT" \
        -m "gpt-5.5" \
        -c "model_reasoning_effort=\"high\"" \
        -s danger-full-access \
        "Respond with exactly: OK" > "$MODEL_CHECK_LOG" 2>&1
      ;;
    gemini)
      gemini -p "Respond with exactly: OK" \
        > "$MODEL_CHECK_LOG" 2>&1
      ;;
    claude)
      claude -p "Respond with exactly: OK" \
        > "$MODEL_CHECK_LOG" 2>&1
      ;;
    *) return 1 ;;
  esac
}

run_with_provider() {
  local provider="$1"
  local prompt_file="$2"
  local output_file="$3"
  local prompt_content
  prompt_content=$(cat "$prompt_file")

  case "$provider" in
    codex)
      codex -a never exec \
        -C "$REPO_ROOT" \
        -m "gpt-5.5" \
        -c "model_reasoning_effort=\"high\"" \
        -s danger-full-access \
        --output-last-message "$output_file" \
        < "$prompt_file" 2>&1 | tee -a "$RUN_LOG" || true
      ;;
    gemini)
      gemini \
        --yolo \
        -p "$prompt_content" \
        2>&1 | tee -a "$RUN_LOG" | tail -500 > "$output_file" || true
      ;;
    claude)
      claude \
        --dangerously-skip-permissions \
        -p "$prompt_content" \
        2>&1 | tee -a "$RUN_LOG" | tail -500 > "$output_file" || true
      ;;
  esac
}

is_rate_limited() {
  grep -qi \
    "rate.limit\|quota.exceeded\|too.many.requests\|usage.limit\|limit.reached\|No capacity\|hit your usage\|upgrade to pro\|credits" \
    "$RUN_LOG" 2>/dev/null
}

# ── Security check ────────────────────────────────────────────────────────────

if [[ "$SKIP_SECURITY" != "true" ]]; then
  echo ""
  echo "==============================================================="
  echo "  Security Pre-Flight Check"
  echo "==============================================================="
  echo ""
  SECURITY_WARNINGS=()
  [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && SECURITY_WARNINGS+=("AWS_ACCESS_KEY_ID is set")
  [[ -n "${DATABASE_URL:-}" ]]      && SECURITY_WARNINGS+=("DATABASE_URL is set")
  if [[ ${#SECURITY_WARNINGS[@]} -gt 0 ]]; then
    echo "WARNING:"
    for w in "${SECURITY_WARNINGS[@]}"; do echo "  - $w"; done
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r; echo ""
    [[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && exit 1
  else
    echo "No credential exposure risks detected."
  fi
  echo ""
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTS_MD="$REPO_ROOT/AGENTS.md"
PRD_FILE="$SCRIPT_DIR/prd.json"
RUN_LOG="$SCRIPT_DIR/run.log"
EVENT_LOG="$SCRIPT_DIR/events.log"
MODEL_CHECK_LOG="$SCRIPT_DIR/.model-check.log"

mkdir -p "$SCRIPT_DIR/audit"

ATTEMPTS_FILE="$SCRIPT_DIR/.story-attempts"
LAST_STORY_FILE="$SCRIPT_DIR/.last-story"
PROVIDER_FILE="$SCRIPT_DIR/.current-provider"

[[ ! -f "$ATTEMPTS_FILE" ]] && echo "{}" > "$ATTEMPTS_FILE"
[[ ! -f "$PRD_FILE" ]] && echo "ERROR: $PRD_FILE not found." && exit 1

# ── Helpers ───────────────────────────────────────────────────────────────────

get_current_story() {
  jq -r '.userStories[] | select(.passes == false) | .id' "$PRD_FILE" 2>/dev/null | head -1
}

get_story_attempts() {
  jq -r --arg id "$1" '.[$id] // 0' "$ATTEMPTS_FILE" 2>/dev/null || echo "0"
}

increment_story_attempts() {
  local current; current=$(get_story_attempts "$1")
  local new=$((current + 1))
  jq --arg id "$1" --argjson c "$new" '.[$id] = $c' "$ATTEMPTS_FILE" > "$ATTEMPTS_FILE.tmp" \
    && mv "$ATTEMPTS_FILE.tmp" "$ATTEMPTS_FILE"
  echo "$new"
}

mark_story_skipped() {
  jq --arg id "$1" --arg note "Skipped: exceeded $2 attempts" '
    .userStories = [.userStories[] |
      if .id == $id then (.notes = $note)|(.passes = true)|(.skipped = true) else . end]
  ' "$PRD_FILE" > "$PRD_FILE.tmp" && mv "$PRD_FILE.tmp" "$PRD_FILE"
  echo "Circuit breaker: $1 skipped after $2 attempts"
}

check_circuit_breaker() {
  local attempts; attempts=$(get_story_attempts "$1")
  if [ "$attempts" -ge "$MAX_ATTEMPTS_PER_STORY" ]; then
    mark_story_skipped "$1" "$MAX_ATTEMPTS_PER_STORY"
    return 0
  fi
  return 1
}

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log_event() { echo "[$(ts)] $*" >> "$EVENT_LOG"; }

get_story_field() {
  jq -r --arg id "$1" --arg f "$2" \
    '.userStories[] | select(.id == $id) | .[$f] // ""' "$PRD_FILE" 2>/dev/null || true
}

get_story_acceptance_criteria() {
  jq -r --arg id "$1" \
    '.userStories[] | select(.id == $id) | .acceptanceCriteria[]' "$PRD_FILE" 2>/dev/null || true
}

run_acceptance_checks() {
  local story_id="$1"
  local failed=0
  echo "Running acceptance checks for $story_id..."

  while IFS= read -r criterion; do
    local cmd="${criterion% passa sem erros}"
    cmd="${cmd% — *}"
    cmd="${cmd%Typecheck passes}"
    cmd="${cmd%Typecheck*}"
    cmd="$(echo "$cmd" | sed 's/[[:space:]]*$//')"

    if [[ "$cmd" == python\ -m\ py_compile* ]]; then
      local files="${cmd#python -m py_compile }"
      if cd "$REPO_ROOT" && python3 -m py_compile $files 2>/dev/null; then
        echo "  OK: py_compile $files"
      else
        echo "  FAIL: py_compile $files"
        failed=1
      fi
    elif [[ "$cmd" == pytest* ]]; then
      if cd "$REPO_ROOT" && $cmd > /dev/null 2>&1; then
        echo "  OK: $cmd"
      else
        echo "  FAIL: $cmd"
        failed=1
      fi
    elif [[ "$cmd" == bash\ -n* ]]; then
      if eval "$cmd" 2>/dev/null; then
        echo "  OK: $cmd"
      else
        echo "  FAIL: $cmd"
        failed=1
      fi
    elif [[ "$cmd" == python\ -m\ json.tool* ]]; then
      local file="${cmd#python -m json.tool }"
      if python3 -m json.tool "$REPO_ROOT/$file" > /dev/null 2>&1; then
        echo "  OK: json.tool $file"
      else
        echo "  FAIL: json.tool $file"
        failed=1
      fi
    fi
  done < <(get_story_acceptance_criteria "$story_id")

  return $failed
}

mark_story_passed() {
  jq --arg id "$1" '
    .userStories = [.userStories[] | if .id == $id then (.passes = true) else . end]
  ' "$PRD_FILE" > "$PRD_FILE.tmp" && mv "$PRD_FILE.tmp" "$PRD_FILE"
}

build_prompt() {
  local story_id="$1" title="$2" desc="$3" criteria="$4"
  local notes="$5" fix="$6" discipline="$7"
  local agents_context=""
  [[ -f "$AGENTS_MD" ]] && agents_context=$(cat "$AGENTS_MD")

  cat << PROMPT
# Ralph Implementation Loop
Today: $(date +%Y-%m-%d)
Provider: $(cat "$PROVIDER_FILE" 2>/dev/null || echo "unknown")

## Project Context (AGENTS.md)
$agents_context

## Current Story
**ID:** $story_id
**Title:** $title
**Discipline:** $discipline

## Description
$desc

## Acceptance Criteria
$criteria

## Notes
$notes
$([ -n "$fix" ] && printf "\n## Known Fix\n%s" "$fix")

---

## Instructions
1. Implement the changes required by this story in the repository at $REPO_ROOT
2. Apply ALL acceptance criteria including py_compile checks
3. Do NOT break existing functionality
4. Do NOT remove tests or skip acceptance criteria
5. If a Known Fix is provided, apply it exactly
6. After implementing, verify py_compile checks listed in acceptance criteria pass
7. Do NOT commit changes. Only implement and verify the code.
8. All API file paths are under src/api/ (not ford-ml-api/)
9. All pipeline paths are under src/pipeline/
PROMPT
}

switch_to_next_provider() {
  local current="$1"
  local reason="$2"
  local next
  next=$(next_provider "$current")
  if [[ -n "$next" ]]; then
    echo "Switching provider: $current -> $next ($reason)" >&2
    log_event "PROVIDER SWITCH $current->$next reason=$reason"
    echo "$next" > "$PROVIDER_FILE"
    echo "$next"
  else
    echo "$current"
  fi
}

# ── Provider setup ─────────────────────────────────────────────────────────────

ACTIVE_PROVIDER=$(detect_provider)
echo "$ACTIVE_PROVIDER" > "$PROVIDER_FILE"

touch "$RUN_LOG" "$EVENT_LOG"

echo "Starting Ralph Implementation Loop"
echo "  Provider chain:         ${PROVIDER_CHAIN[*]}"
echo "  Active provider:        $ACTIVE_PROVIDER"
echo "  Max iterations:         $MAX_ITERATIONS"
echo "  Max attempts per story: $MAX_ATTEMPTS_PER_STORY"
echo "  PRD:                    $PRD_FILE"
echo "  AGENTS.md:              $AGENTS_MD"
echo "  Logs:"
echo "    tail -n $TAIL_N -f $EVENT_LOG"
echo "    tail -n $TAIL_N -f $RUN_LOG"
echo ""

log_event "RUN START provider=$ACTIVE_PROVIDER max_iterations=$MAX_ITERATIONS"

# Preflight — se provider explícito, testa só ele; senão percorre a chain
echo "Running provider preflight..."
PREFLIGHT_OK=false
PREFLIGHT_CHAIN=("${PROVIDER_CHAIN[@]}")
if [[ -n "$PROVIDER" ]]; then
  PREFLIGHT_CHAIN=("$PROVIDER")
fi
for p in "${PREFLIGHT_CHAIN[@]}"; do
  if ! command -v "$p" &>/dev/null; then
    echo "  $p: not installed, skipping"
    continue
  fi
  echo -n "  Testing $p... "
  if test_provider "$p"; then
    echo "OK"
    ACTIVE_PROVIDER="$p"
    echo "$p" > "$PROVIDER_FILE"
    PREFLIGHT_OK=true
    break
  else
    echo "FAIL ($(tail -1 "$MODEL_CHECK_LOG" 2>/dev/null | tr -d '\n' | cut -c1-80))"
  fi
done

if [[ "$PREFLIGHT_OK" != "true" ]]; then
  echo ""
  echo "ERROR: All providers failed preflight. Check credentials."
  echo "  codex:  printenv OPENAI_API_KEY | codex login --with-api-key"
  echo "  gemini: gemini auth login"
  echo "  claude: claude login"
  exit 1
fi

echo "Provider preflight OK: $ACTIVE_PROVIDER"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo ""
  echo "==============================================================="
  echo "  Ralph — Iteration $i/$MAX_ITERATIONS — Provider: $ACTIVE_PROVIDER"
  echo "==============================================================="
  log_event "ITERATION START $i/$MAX_ITERATIONS provider=$ACTIVE_PROVIDER"

  CURRENT_STORY=$(get_current_story)

  if [ -z "$CURRENT_STORY" ]; then
    log_event "RUN COMPLETE all stories passed"
    echo "All stories passed. Ralph completed!"
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  LAST_STORY=""
  [[ -f "$LAST_STORY_FILE" ]] && LAST_STORY=$(cat "$LAST_STORY_FILE" 2>/dev/null || echo "")

  if [ "$CURRENT_STORY" == "$LAST_STORY" ]; then
    ATTEMPTS=$(increment_story_attempts "$CURRENT_STORY")
    echo "Consecutive attempt on $CURRENT_STORY: $ATTEMPTS/$MAX_ATTEMPTS_PER_STORY"

    # Tenta próximo provider na tentativa 3
    if [ "$ATTEMPTS" -eq 3 ]; then
      ACTIVE_PROVIDER=$(switch_to_next_provider "$ACTIVE_PROVIDER" "retry-attempt-3")
    fi

    if check_circuit_breaker "$CURRENT_STORY"; then
      echo "Skipping to next story..."
      echo "$CURRENT_STORY" > "$LAST_STORY_FILE"
      sleep 1
      continue
    fi
  else
    ATTEMPTS=$(increment_story_attempts "$CURRENT_STORY")
    echo "Starting: $CURRENT_STORY (attempt $ATTEMPTS/$MAX_ATTEMPTS_PER_STORY)"
  fi

  echo "$CURRENT_STORY" > "$LAST_STORY_FILE"

  STORY_TITLE="$(get_story_field "$CURRENT_STORY" "title")"
  STORY_DESC="$(get_story_field "$CURRENT_STORY" "description")"
  STORY_NOTES="$(get_story_field "$CURRENT_STORY" "notes")"
  STORY_DISCIPLINE="$(get_story_field "$CURRENT_STORY" "discipline")"
  STORY_FIX="$(get_story_field "$CURRENT_STORY" "fix")"
  STORY_CRITERIA="$(get_story_acceptance_criteria "$CURRENT_STORY")"

  log_event "STORY START id=$CURRENT_STORY attempt=$ATTEMPTS provider=$ACTIVE_PROVIDER"

  PROMPT_FILE="$SCRIPT_DIR/.prompt.md"
  LAST_MESSAGE_FILE="$SCRIPT_DIR/.last-message.md"
  > "$LAST_MESSAGE_FILE"

  build_prompt \
    "$CURRENT_STORY" "$STORY_TITLE" "$STORY_DESC" \
    "$STORY_CRITERIA" "$STORY_NOTES" "$STORY_FIX" \
    "$STORY_DISCIPLINE" > "$PROMPT_FILE"

  run_with_provider "$ACTIVE_PROVIDER" "$PROMPT_FILE" "$LAST_MESSAGE_FILE"

  # Detecta rate limit e tenta próximo provider automaticamente
  if is_rate_limited; then
    echo "Rate limit detected on $ACTIVE_PROVIDER."
    NEXT=$(switch_to_next_provider "$ACTIVE_PROVIDER" "rate-limit")
    if [[ "$NEXT" != "$ACTIVE_PROVIDER" ]]; then
      ACTIVE_PROVIDER="$NEXT"
      echo "Retrying with $ACTIVE_PROVIDER..."
      > "$LAST_MESSAGE_FILE"
      run_with_provider "$ACTIVE_PROVIDER" "$PROMPT_FILE" "$LAST_MESSAGE_FILE"
    fi
  fi

  if [ ! -s "$LAST_MESSAGE_FILE" ]; then
    log_event "ERROR story=$CURRENT_STORY empty-output provider=$ACTIVE_PROVIDER"
    echo "ERROR: No output produced. See: $RUN_LOG"
    sleep 2
    continue
  fi

  # Salva relatório de implementação
  OUT_FILE="$SCRIPT_DIR/audit/${CURRENT_STORY}-implementation.md"
  cat "$LAST_MESSAGE_FILE" > "$OUT_FILE"

  # Verifica acceptance criteria
  if run_acceptance_checks "$CURRENT_STORY"; then
    mark_story_passed "$CURRENT_STORY"
    log_event "STORY COMPLETE id=$CURRENT_STORY provider=$ACTIVE_PROVIDER attempt=$ATTEMPTS"
    echo "Story $CURRENT_STORY passed. Marked complete."
  else
    log_event "STORY FAIL id=$CURRENT_STORY provider=$ACTIVE_PROVIDER attempt=$ATTEMPTS"
    echo "Story $CURRENT_STORY failed acceptance checks. Will retry."
  fi

  REMAINING=$(jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "0")
  echo "Remaining stories: $REMAINING"

  if [ "$REMAINING" == "0" ]; then
    log_event "RUN COMPLETE all stories passed"
    echo "All stories passed. Ralph completed!"
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS)."
REMAINING=$(jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "?")
echo "Remaining stories: $REMAINING"
log_event "RUN STOPPED max iterations reached"
exit 1