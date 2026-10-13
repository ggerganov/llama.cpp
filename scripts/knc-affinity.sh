#!/usr/bin/env bash
# Run on KNC. Reserve the physical core containing CPU 0.
set -euo pipefail
[[ $(uname -m) == k1om ]] || { echo 'KNC hardware required' >&2; exit 2; }
siblings=${1:?Usage: knc-affinity.sh 1..4 command [arguments...]}
shift
[[ $siblings =~ ^[1-4]$ && $# -gt 0 ]] || exit 2

expand() {
    local part first last cpu
    local -a parts
    IFS=, read -r -a parts <<< "$1"
    for part in "${parts[@]}"; do
        [[ $part =~ ^[0-9]+(-[0-9]+)?$ ]] || return 2
        first=${part%-*}; last=${part#*-}
        (( first <= last && last < 244 )) || return 2
        for ((cpu=first; cpu<=last; ++cpu)); do printf '%s\n' "$cpu"; done
    done
}

declare -A allowed
while read -r cpu; do allowed[$cpu]=1; done < <(expand "$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)")
mapfile -t groups < <(cat /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list | sort -u -n)
case ${#groups[@]} in 57|60|61) ;; *) echo 'Unexpected KNC topology' >&2; exit 2;; esac
declare -a cpus=() members=()
for ((s=0; s<siblings; ++s)); do
    for group in "${groups[@]}"; do
        mapfile -t members < <(expand "$group")
        ((${#members[@]} == 4)) || { echo 'Expected four siblings per core' >&2; exit 2; }
        [[ ${members[0]} != 0 ]] || continue
        cpu=${members[$s]}
        [[ ${allowed[$cpu]:-0} == 1 ]] || { echo "CPU $cpu excluded by current affinity" >&2; exit 2; }
        cpus+=("$cpu")
    done
done
unset OMP_PLACES KMP_AFFINITY OMP_THREAD_LIMIT
export GOMP_CPU_AFFINITY="${cpus[*]}"
export OMP_NUM_THREADS=${#cpus[@]} OMP_DYNAMIC=FALSE OMP_NESTED=FALSE OMP_MAX_ACTIVE_LEVELS=1 OMP_PROC_BIND=TRUE
printf 'KNC workers=%s CPUs=%s\n' "$OMP_NUM_THREADS" "$GOMP_CPU_AFFINITY" >&2
exec "$@"
