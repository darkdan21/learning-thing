#!/bin/bash

outfile="mark.npy"

if [ -e "$outfile" ]
then
	echo "$outfile aready exists!" >&2
	exit 1
fi

if [ "$#" -eq 0 ] || [ "$1" == '-h' ]
then
	echo "Usage: $0 file [file ...]" >&2
	exit 1
fi

[ -d "venv" ] && source venv/bin/activate

python3 scripts/build-marking-file.py -l debug --prefix --out "$outfile" $*

