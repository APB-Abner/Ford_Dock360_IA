#!/bin/bash
# Ralph Implementation Loop (OpenAI Codex) - Autonomous implementation loop.
# Reads scripts/prd.json and implements each US with passes: false.
# Marks passes: true when py_compile and acceptance criteria pass.
#
# Usage: ./scripts/ralph.sh [max_iterations] [--skip-security-check] [--no-search]
#
# Writes all artifacts under scripts/audit/ (logs and implementation reports).

set -euo pipefail

export CODEX_INTERNAL_ORIGINATOR_OVERRIDE="Codex Desktop"

MAX_ITERATIONS=40
MAX_ATTEMPTS_PER_STORY="${MAX_ATTEMPTS_PER_STORY:-5}"
SKIP_SECURITY="${SKIP_SECURITY_CHECK:-false}"
ENABLE_SEARCH="true"
TAIL_N="${TAIL_N:-200}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-security-check)
      SKIP_SECURITY="true"
      shift
      ;;
    --search)
      ENABLE_SEARCH="true"
      shift
      ;;
    --no-search)
      ENABLE_SEARCH="false"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

if [[ "$SKIP_SECURITY" != "true" ]]; then
  echo ""
  echo "==============================================================="
  echo "  Security Pre-Flight Check"
  echo "==============================================================="
  echo ""

  SECURITY_WARNINGS=()

  if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    SECURITY_WARNINGS+=("AWS_ACCESS_KEY_ID is set - production credentials may be exposed")
  fi

  if [[ -n "${DATABASE_URL:-}" ]]; then
    SECURITY_WARNINGS+=("DATABASE_URL is set - database credentials may be exposed")
  fi

  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    SECURITY_WARNINGS+=("OPENAI_API_KEY is set - will be used for Codex calls")
  fi

  if [[ ${#SECURITY_WARNINGS[@]} -gt 0 ]]; then
    echo "WARNING: Potential credential exposure detected:"
    echo ""
    for warning in "${SECURITY_WARNINGS[@]}"; do
      echo "  - $warning"
    done
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborted. Use --skip-security-check to bypass."
      exit 1
    fi
  else
    echo "No credential exposure risks detected."
  fi
  echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRD_FILE="$SCRIPT_DIR/prd.json"
RUN_LOG="$SCRIPT_DIR/run.log"
EVENT_LOG="$SCRIPT_DIR/events.log"
MODEL_CHECK_LOG="$SCRIPT_DIR/.model-check.log"

mkdir -p "$SCRIPT_DIR/audit"

ATTEMPTS_FILE="$SCRIPT_DIR/.story-attempts"
LAST_STORY_FILE="$SCRIPT_DIR/.last-story"

if [ ! -f "$ATTEMPTS_FILE" ]; then
  echo "{}" > "$ATTEMPTS_FILE"
fi

if [ ! -f "$PRD_FILE" ]; then
  echo "ERROR: $PRD_FILE not found. Copy prd.json to scripts/prd.json before running."
  exit 1
fi

get_current_story() {
  jq -r '.userStories[] | select(.passes == false) | .id' "$PRD_FILE" 2>/dev/null | head -1
}

get_story_attempts() {
  local story_id="$1"
  jq -r --arg id "$story_id" '.[$id] // 0' "$ATTEMPTS_FILE" 2>/dev/null || echo "0"
}

increment_story_attempts() {
  local story_id="$1"
  local current
  current=$(get_story_attempts "$story_id")
  local new_count=$((current + 1))
  jq --arg id "$story_id" --argjson count "$new_count" '.[$id] = $count' "$ATTEMPTS_FILE" > "$ATTEMPTS_FILE.tmp" \
    && mv "$ATTEMPTS_FILE.tmp" "$ATTEMPTS_FILE"
  echo "$new_count"
}

mark_story_skipped() {
  local story_id="$1"
  local max_attempts="$2"
  local note="Skipped: exceeded $max_attempts attempts without passing"
  jq --arg id "$story_id" --arg note "$note" '
    .userStories = [
      .userStories[]
      | if .id == $id then
          (.notes = $note) | (.passes = true) | (.skipped = true)
        else
          .
        end
    ]
  ' "$PRD_FILE" > "$PRD_FILE.tmp" && mv "$PRD_FILE.tmp" "$PRD_FILE"
  echo "Circuit breaker: Marked $story_id as skipped after $max_attempts attempts"
}

check_circuit_breaker() {
  local story_id="$1"
  local attempts
  attempts=$(get_story_attempts "$story_id")
  if [ "$attempts" -ge "$MAX_ATTEMPTS_PER_STORY" ]; then
    echo "Circuit breaker: $story_id reached max attempts ($attempts/$MAX_ATTEMPTS_PER_STORY)"
    mark_story_skipped "$story_id" "$MAX_ATTEMPTS_PER_STORY"
    return 0
  fi
  return 1
}

ts() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log_event() {
  echo "[$(ts)] $*" >> "$EVENT_LOG"
}

get_story_field() {
  local story_id="$1"
  local field="$2"
  jq -r --arg id "$story_id" --arg f "$field" \
    '.userStories[] | select(.id == $id) | .[$f] // ""' "$PRD_FILE" 2>/dev/null || true
}

get_story_acceptance_criteria() {
  local story_id="$1"
  jq -r --arg id "$story_id" \
    '.userStories[] | select(.id == $id) | .acceptanceCriteria[]' "$PRD_FILE" 2>/dev/null || true
}

get_story_fix() {
  local story_id="$1"
  jq -r --arg id "$story_id" \
    '.userStories[] | select(.id == $id) | .fix // ""' "$PRD_FILE" 2>/dev/null || true
}

run_acceptance_checks() {
  local story_id="$1"
  local failed=0

  echo "Running acceptance checks for $story_id..."

  while IFS= read -r criterion; do
    if [[ "$criterion" == python\ -m\ py_compile* ]]; then
      clean="${criterion% passa sem erros}"
      clean="${clean% Typecheck passes}"
      files="${clean#python -m py_compile }"
      if python3 -m py_compile $files 2>/dev/null; then
        echo "  OK: $criterion"
      else
        echo "  FAIL: $criterion"
        failed=1
      fi
    elif [[ "$criterion" == pytest* ]]; then
      if cd "$REPO_ROOT" && $criterion > /dev/null 2>&1; then
        echo "  OK: $criterion"
      else
        echo "  FAIL: $criterion"
        failed=1
      fi
    elif [[ "$criterion" == bash\ -n* ]]; then
      if eval "$criterion" 2>/dev/null; then
        echo "  OK: $criterion"
      else
        echo "  FAIL: $criterion"
        failed=1
      fi
    elif [[ "$criterion" == python\ -m\ json.tool* ]]; then
      file="${criterion#python -m json.tool }"
      if python3 -m json.tool "$REPO_ROOT/$file" > /dev/null 2>&1; then
        echo "  OK: $criterion"
      else
        echo "  FAIL: $criterion"
        failed=1
      fi
    fi
  done < <(get_story_acceptance_criteria "$story_id")

  return $failed
}

mark_story_passed() {
  local story_id="$1"
  jq --arg id "$story_id" '
    .userStories = [
      .userStories[]
      | if .id == $id then (.passes = true) else . end
    ]
  ' "$PRD_FILE" > "$PRD_FILE.tmp" && mv "$PRD_FILE.tmp" "$PRD_FILE"
}

REQUESTED_MODEL="gpt-5.5"
REASONING_EFFORT="high"

if [[ -n "${CODEX_MODEL:-}" && "${CODEX_MODEL}" != "$REQUESTED_MODEL" ]]; then
  echo "ERROR: Loop pinned to CODEX_MODEL=$REQUESTED_MODEL. Unset CODEX_MODEL to continue."
  exit 1
fi

touch "$RUN_LOG" "$EVENT_LOG"

echo "Starting Ralph Implementation Loop (OpenAI Codex)"
echo "  Max iterations:           $MAX_ITERATIONS"
echo "  Max attempts per story:   $MAX_ATTEMPTS_PER_STORY"
echo "  Model:                    $REQUESTED_MODEL (reasoning_effort=$REASONING_EFFORT)"
echo "  PRD:                      $PRD_FILE"
echo "  Logs:"
echo "    tail -n $TAIL_N -f $EVENT_LOG"
echo "    tail -n $TAIL_N -f $RUN_LOG"
echo ""

log_event "RUN START max_iterations=$MAX_ITERATIONS max_attempts=$MAX_ATTEMPTS_PER_STORY model=$REQUESTED_MODEL"

# Preflight: verify model access
MODEL_CHECK_CMD=(
  codex -a never exec
  -C "$REPO_ROOT"
  -m "$REQUESTED_MODEL"
  -c "model_reasoning_effort=\"$REASONING_EFFORT\""
  -s danger-full-access
  "Respond with exactly: OK"
)

if ! "${MODEL_CHECK_CMD[@]}" > "$MODEL_CHECK_LOG" 2>&1; then
  echo "ERROR: Model preflight failed for '$REQUESTED_MODEL'. See: $MODEL_CHECK_LOG"
  echo "Re-auth: printenv OPENAI_API_KEY | codex login --with-api-key"
  exit 1
fi

echo "Model preflight OK."
echo ""

CODEX_ARGS=(-a never)
if [[ "$ENABLE_SEARCH" == "true" ]]; then
  CODEX_ARGS+=(--search)
fi
CODEX_ARGS+=(
  exec
  -C "$REPO_ROOT"
  -m "$REQUESTED_MODEL"
  -c "model_reasoning_effort=\"$REASONING_EFFORT\""
  -s danger-full-access
)

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Implementation — Iteration $i of $MAX_ITERATIONS"
  echo "==============================================================="

  log_event "ITERATION START $i/$MAX_ITERATIONS"

  CURRENT_STORY=$(get_current_story)

  if [ -z "$CURRENT_STORY" ]; then
    log_event "RUN COMPLETE all stories passed"
    echo ""
    echo "All stories are marked passes: true."
    echo "Ralph implementation completed!"
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  LAST_STORY=""
  if [ -f "$LAST_STORY_FILE" ]; then
    LAST_STORY=$(cat "$LAST_STORY_FILE" 2>/dev/null || echo "")
  fi

  if [ "$CURRENT_STORY" == "$LAST_STORY" ]; then
    ATTEMPTS=$(increment_story_attempts "$CURRENT_STORY")
    echo "Consecutive attempt on $CURRENT_STORY: $ATTEMPTS/$MAX_ATTEMPTS_PER_STORY"
    if check_circuit_breaker "$CURRENT_STORY"; then
      echo "Skipping to next story..."
      echo "$CURRENT_STORY" > "$LAST_STORY_FILE"
      sleep 1
      continue
    fi
  else
    ATTEMPTS=$(increment_story_attempts "$CURRENT_STORY")
    echo "Starting story: $CURRENT_STORY (attempt $ATTEMPTS/$MAX_ATTEMPTS_PER_STORY)"
  fi

  echo "$CURRENT_STORY" > "$LAST_STORY_FILE"

  STORY_TITLE="$(get_story_field "$CURRENT_STORY" "title")"
  STORY_DESC="$(get_story_field "$CURRENT_STORY" "description")"
  STORY_NOTES="$(get_story_field "$CURRENT_STORY" "notes")"
  STORY_DISCIPLINE="$(get_story_field "$CURRENT_STORY" "discipline")"
  STORY_FIX="$(get_story_fix "$CURRENT_STORY")"
  STORY_CRITERIA="$(get_story_acceptance_criteria "$CURRENT_STORY")"

  log_event "STORY START id=$CURRENT_STORY attempt=$ATTEMPTS title=$(printf '%s' "$STORY_TITLE" | tr '\n' ' ')"

  PROMPT_FILE="$SCRIPT_DIR/.prompt.md"
  LAST_MESSAGE_FILE="$SCRIPT_DIR/.last-message.md"

  {
    printf -- "# Ralph Implementation Loop\n\n"
    printf -- "Today's date: %s\n\n" "$(date +%Y-%m-%d)"
    printf -- "## Current Story\n\n"
    printf -- "**ID:** %s\n" "$CURRENT_STORY"
    printf -- "**Title:** %s\n" "$STORY_TITLE"
    printf -- "**Discipline:** %s\n\n" "$STORY_DISCIPLINE"
    printf -- "## Description\n\n%s\n\n" "$STORY_DESC"
    printf -- "## Acceptance Criteria\n\n%s\n\n" "$STORY_CRITERIA"
    printf -- "## Notes\n\n%s\n\n" "$STORY_NOTES"
    if [ -n "$STORY_FIX" ]; then
      printf -- "## Known Fix\n\n%s\n\n" "$STORY_FIX"
    fi
    printf -- "---\n\n"
    printf -- "## Instructions\n\n"
    printf -- "1. Implement the changes required by this story in the repository.\n"
    printf -- "2. Apply ALL acceptance criteria, including py_compile checks.\n"
    printf -- "3. Do NOT break existing functionality.\n"
    printf -- "4. Do NOT remove tests or skip acceptance criteria.\n"
    printf -- "5. If a Known Fix is provided above, apply it exactly.\n"
    printf -- "6. After implementing, run the py_compile checks listed in acceptance criteria.\n"
    printf -- "7. Commit the changes with message: 'fix(%s): %s'\n\n" "$CURRENT_STORY" "$STORY_TITLE"
    printf -- "Repository root: %s\n" "$REPO_ROOT"
  } > "$PROMPT_FILE"

  codex "${CODEX_ARGS[@]}" --output-last-message "$LAST_MESSAGE_FILE" < "$PROMPT_FILE" 2>&1 | tee -a "$RUN_LOG" || true

  if [ ! -s "$LAST_MESSAGE_FILE" ]; then
    log_event "ERROR story=$CURRENT_STORY codex-empty-last-message"
    echo "ERROR: Codex did not produce output. See: $RUN_LOG"
    sleep 2
    continue
  fi

  # Save implementation report
  OUT_FILE="$SCRIPT_DIR/audit/${CURRENT_STORY}-implementation.md"
  cat "$LAST_MESSAGE_FILE" > "$OUT_FILE"

  # Run acceptance checks
  if run_acceptance_checks "$CURRENT_STORY"; then
    mark_story_passed "$CURRENT_STORY"
    log_event "STORY COMPLETE id=$CURRENT_STORY attempt=$ATTEMPTS"
    echo "Story $CURRENT_STORY passed acceptance checks. Marked as complete."
  else
    log_event "STORY FAIL id=$CURRENT_STORY attempt=$ATTEMPTS acceptance-checks-failed"
    echo "Story $CURRENT_STORY failed acceptance checks. Will retry."
  fi

  REMAINING=$(jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE" 2>/dev/null || echo "0")
  echo "Remaining stories: $REMAINING"

  if [ "$REMAINING" == "0" ]; then
    log_event "RUN COMPLETE all stories passed"
    echo ""
    echo "All stories are marked passes: true."
    echo "Ralph implementation completed!"
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Remaining: $(jq -r '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE") stories"
echo "Tail log: tail -f $RUN_LOG"
log_event "RUN STOPPED reached max iterations"
exit 1
