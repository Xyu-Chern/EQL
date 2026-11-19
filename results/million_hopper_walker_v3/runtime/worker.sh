set -eu
umask 077
root=/gscratch/stf/xyuchern/eql-million-hopper-walker-v3
source "$root/activate-v3.sh"
target=/tmp/eql-xyuchern-cb5f7e7-v1/EQL-million-v3
mkdir -p "$target"
cd "$target"
bash "$root/stage-code.sh"
python -u "$root/probe.py"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
tasks="Hopper-v3 Walker2d-v3"
case "$1" in
 a40) methods="EQL_bagging Adaptive11 SAC100 EQL SAC10";;
 rtx) methods="SAC10 Adaptive11 EQL SAC100 EQL_bagging"; tasks="Walker2d-v3 Hopper-v3";;
 l40) methods="SAC100 Adaptive11 EQL_bagging EQL SAC10";;
 ti) methods="EQL Adaptive11 SAC10 EQL_bagging SAC100"; tasks="Walker2d-v3 Hopper-v3";;
 *) exit 2;;
esac
for method in $methods; do
 for task in $tasks; do
    echo "START $task $method $(date -Iseconds)"
    python -u run_million_native.py "$task" "$method" --output "$root/results"
 done
done
