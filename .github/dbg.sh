#!/usr/bin/env bash
# usage: dbg.sh <title> <lines> <command...>
# runs the command and copies the tail of its output into an annotation
title=$1; lines=$2; shift 2
set +e
"$@" > "out_$$.txt" 2>&1
code=$?
tail -n "$lines" "out_$$.txt"
level=notice; [ $code -ne 0 ] && level=error
msg=$(tail -n "$lines" "out_$$.txt" | cut -c1-300 | sed -e 's/%/%25/g' -e 's/\r//g' | awk '{printf "%s%%0A", $0}')
echo "::$level title=$title::$msg"
exit $code
