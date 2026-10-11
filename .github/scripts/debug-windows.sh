#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Runs a program under the Windows debugger with full page heap, so a
# heap overrun stops at the write that makes it rather than at a later
# free, and prints the call stack where it stopped. Exits with failure,
# as it runs only to explain one.
#
#     debug-windows.sh SYMBOL_DIR PROGRAM [ARGUMENT...]
set -uo pipefail

symbols=$1
program=$2
shift 2
debuggers="/c/Program Files (x86)/Windows Kits/10/Debuggers/x64"
if [ ! -x "$debuggers/cdb.exe" ]; then
    echo "No Windows debugger at $debuggers"
    exit 1
fi
image=$(basename "$program")
echo "Debugging $program $*"

"$debuggers/gflags.exe" //p //enable "$image" //full
commands="$RUNNER_TEMP/cdb-commands.txt"
cat > "$commands" <<'EOF'
.lines
sxe -c ".echo *** Access violation; .ecxr; kn 60; q" av
sxe -c ".echo *** Heap corruption; .ecxr; kn 60; q" c0000374
sxe -c ".echo *** Break; kn 60; q" bpe
g
EOF
timeout 1800 "$debuggers/cdb.exe" -G -y "$(cygpath -w "$symbols")" \
    -cf "$(cygpath -w "$commands")" "$(cygpath -w "$program")" "$@" \
    < /dev/null
"$debuggers/gflags.exe" //p //disable "$image"
exit 1
