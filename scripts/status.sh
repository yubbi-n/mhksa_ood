#!/bin/bash
# One-screen status of a train.py / evaluate.py run, one line per seed.
#   bash scripts/status.sh train_in1k_m20 [refresh seconds, 0 = once]      (train logs: <prefix>_seed<S>.out)
P=${1:?usage: bash scripts/status.sh <train log prefix> [refresh seconds]}
N=${2:-30}
while true; do
  clear
  printf '%s | disk avail %s | server mem avail %sG\n\n' "$(date '+%m-%d %H:%M')" \
    "$(df -h /data2 | awk 'NR==2{print $4}')" "$(free -g | awk 'NR==2{print $7}')"
  printf '%-5s %-10s %-9s %-7s %s\n' seed 'done H' elapsed mem state
  for f in ${P}_seed*.out; do
    [ -e "$f" ] || continue
    S=${f##*_seed}; S=${S%.out}
    last=$(tail -n 1 "$f")
    H=$(grep -oE 'H= ?[0-9]+' "$f" | tail -n 1 | tr -d 'H= ')
    T=$(grep -oE '\([0-9]+s\)$' "$f" | tail -n 1 | tr -d '()s')
    T=${T:+$((T / 3600))h$(( T % 3600 / 60 ))m}
    pid=$(pgrep -u "$USER" -f "train.py.*run.seeds=\[$S\]" | head -n 1)
    mem=${pid:+$(ps -o rss= -p "$pid" | awk '{printf "%.1fG", $1/1048576}')}
    case "$last" in
      checkpoints*) st=finished ;;
      *) st=${pid:+running}; st=${st:-stopped} ;;
    esac
    printf '%-5s %-10s %-9s %-7s %s\n' "$S" "${H:--}/10" "${T:--}" "${mem:--}" "$st"
  done
  ev=$(pgrep -u "$USER" -af "evaluate.py" | grep -oE 'run.seeds=\[[0-9,]+\]' | head -n 1)
  [ -n "$ev" ] && printf '\nevaluate running: %s\n' "$ev"
  printf '\ncheckpoints:\n'; du -sh checkpoints/* 2>/dev/null | sed 's/^/  /'
  [ "$N" = 0 ] && break
  sleep "$N"
done
