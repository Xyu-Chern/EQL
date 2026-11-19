set -eu
root=/gscratch/stf/xyuchern/eql-million-hopper-walker-v3
target=/tmp/eql-xyuchern-cb5f7e7-v1/EQL-million-v3
for group in fixed adaptive; do
 mkdir -p "$target/$group"
 tar -xzf "$root/code-base.tar.gz" -C "$target/$group"
 cp "$root"/src/*.py "$target/$group/"
done
cp "$root/src-adaptive/run_adaptive_tasks.py" "$target/adaptive/"
cp "$root/dispatch.py" "$target/dispatch.py.tmp"
mv "$target/dispatch.py.tmp" "$target/run_million_native.py"
