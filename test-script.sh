#!/bin/bash
#
# Timing sweep for the serial NN.
#
# Flags note: unoptimized and optimized are built with the SAME compiler flags
# (see CMakeLists add_nn_target -> NN_COMPILE_OPTIONS). The difference between
# them is purely in the code, not the flags.
#
# Timing method: with no hyperfine / /usr/bin/time available, we run each
# config REPEATS times and report the MINIMUM wall-clock. Min is the best
# estimator -- noise only ever makes a run slower, so the fastest run is
# closest to the true compute time. The FIRST run streams the program's
# output live so you can see it working; extra runs just re-time (dots).
#
# Batch sizes divide evenly so nothing is dropped/miscounted:
#   - training batches all divide 60000
#   - eval batch divides 10000 (evaluate() assumes an exact fit)
#
# Config is env-overridable, e.g.:  REPEATS=1 EPOCHS=2 ./test-script.sh

set -u
cd "$(dirname "$0")" || exit 1

EPOCHS=${EPOCHS:-5}
LR=${LR:-0.05}
HIDDEN=${HIDDEN:-128}
EVAL_BS=${EVAL_BS:-100}                          # 10000 / 100 = 100 (exact)
REPEATS=${REPEATS:-3}                            # runs per config; min reported
read -r -a TRAIN_BATCHES <<< "${TRAIN_BATCHES:-16 32 50 100 200 400}"   # divisors of 60000

# ---- colors (auto-disable if not a terminal or NO_COLOR set) ----------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    BLUE=$'\033[34m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=; DIM=; RED=; GREEN=; YELLOW=; BLUE=; CYAN=; RESET=
fi

WIDTH=60

rule () { printf "%s%s%s\n" "$DIM" "$(printf '─%.0s' $(seq 1 $WIDTH))" "$RESET"; }

box () {  # big centered banner
    local text="$1" len pad rpad
    len=${#text}; pad=$(( (WIDTH - len) / 2 )); rpad=$(( WIDTH - len - pad ))
    printf "%s╔%s╗%s\n" "$CYAN$BOLD" "$(printf '═%.0s' $(seq 1 $WIDTH))" "$RESET"
    printf "%s║%*s%s%*s║%s\n" "$CYAN$BOLD" "$pad" "" "$text" "$rpad" "" "$RESET"
    printf "%s╚%s╝%s\n" "$CYAN$BOLD" "$(printf '═%.0s' $(seq 1 $WIDTH))" "$RESET"
}

section () { echo; printf "  %s%s%s\n" "$BOLD$YELLOW" "$1" "$RESET"; rule; }

# time_config <command...> -> sets RESULT_TIME (min secs) and RESULT_ACC.
# First run streams live (indented, dim); remaining runs are silent re-timings.
RESULT_TIME=""; RESULT_ACC=""
time_config () {
    local tmp best=""
    tmp=$(mktemp)
    for ((r = 1; r <= REPEATS; r++)); do
        local start end elapsed
        start=$(date +%s.%N)
        if [ "$r" -eq 1 ]; then
            stdbuf -oL "$@" 2>/dev/null | tee "$tmp" | while IFS= read -r line; do
                printf "      %s%s%s\n" "$DIM" "$line" "$RESET"
            done
            [ "$REPEATS" -gt 1 ] && printf "      %sre-timing %d×: %s" "$DIM" "$((REPEATS - 1))" "$RESET"
        else
            "$@" > "$tmp" 2>/dev/null
            printf "%s·%s" "$DIM" "$RESET"
        fi
        end=$(date +%s.%N)
        elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')
        if [ -z "$best" ] || awk -v e="$elapsed" -v b="$best" 'BEGIN { exit !(e < b) }'; then
            best=$elapsed
        fi
    done
    [ "$REPEATS" -gt 1 ] && echo
    RESULT_TIME=$best
    RESULT_ACC=$(grep -oiE "test accuracy: [0-9.]+" "$tmp" | tail -1 | grep -oE "[0-9.]+$")
    rm -f "$tmp"
}

# ---- rebuild so we never time a stale binary --------------------------------
if [ -f build/Makefile ]; then
    printf "%sBuilding...%s " "$DIM" "$RESET"
    make -s -C build >/dev/null || { echo "${RED}build failed${RESET}"; exit 1; }
    printf "%sok%s\n" "$GREEN" "$RESET"
fi

box "NEURAL NETWORK - SERIAL TIMING SWEEP"
printf " %sepochs%s=%s  %slr%s=%s  %shidden%s=%s  %seval_batch%s=%s  %srepeats%s=%s %s(min)%s\n" \
    "$CYAN" "$RESET" "$EPOCHS" "$CYAN" "$RESET" "$LR" "$CYAN" "$RESET" "$HIDDEN" \
    "$CYAN" "$RESET" "$EVAL_BS" "$CYAN" "$RESET" "$REPEATS" "$DIM" "$RESET"
printf " %ssame compiler flags on both builds -- difference is code, not flags%s\n" "$DIM" "$RESET"

# ---- baseline: unoptimized (no batching; epochs/lr/hidden only) --------------
section "UNOPTIMIZED  (serial baseline, no batching)"
printf " %s▶%s %sserial%s  %s./unoptimized-serial-nn %s %s %s%s\n" \
    "$RED" "$RESET" "$BOLD" "$RESET" "$DIM" "$EPOCHS" "$LR" "$HIDDEN" "$RESET"
time_config ./unoptimized-serial-nn "$EPOCHS" "$LR" "$HIDDEN"
BASE_TIME=$RESULT_TIME
printf "   %s→ min %ss%s   %sacc %s%s   %s(baseline)%s\n" \
    "$BOLD$RED" "$RESULT_TIME" "$RESET" "$CYAN" "$RESULT_ACC" "$RESET" "$DIM" "$RESET"

# ---- optimized sweep over training batch size -------------------------------
section "OPTIMIZED  (batched + __restrict, eval_bs=$EVAL_BS)"
for bs in "${TRAIN_BATCHES[@]}"; do
    printf " %s▶%s %sbs=%-3s%s %s./optimized-serial-nn %s %s %s %s %s%s\n" \
        "$GREEN" "$RESET" "$BOLD" "$bs" "$RESET" \
        "$DIM" "$EPOCHS" "$LR" "$HIDDEN" "$bs" "$EVAL_BS" "$RESET"
    time_config ./optimized-serial-nn "$EPOCHS" "$LR" "$HIDDEN" "$bs" "$EVAL_BS"
    speedup=$(awk -v base="$BASE_TIME" -v t="$RESULT_TIME" 'BEGIN { printf "%.2f", base / t }')
    printf "   %s→ min %ss%s   %sacc %s%s   %s%s× vs baseline%s\n" \
        "$BOLD$GREEN" "$RESULT_TIME" "$RESET" "$CYAN" "$RESULT_ACC" "$RESET" \
        "$BOLD$BLUE" "$speedup" "$RESET"
    echo
done
