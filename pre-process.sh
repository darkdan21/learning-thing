#!/bin/bash

# Die on error
set -e

usage() {
	if [ "$1" == "" ]
	then
		stat="0"
	else
		stat="$1"
	fi
	cat >&2 <<EOF
Usage: $0 [-h] -d directory_to_proccess [-s save_directory]

  -h: Display this message and exit
  -d: Directory containing the images to pre-process
  -s: Directory to save processed images to (defaults to new temporary
  	  directory) - will be created if it does not exist.
EOF
	exit $stat
}

while getopts "hd:s:" arg
do
	case $arg in
		h)
			usage 0
			;;
		d)
			dir_to_process="$OPTARG"
			;;
		s)
			save_dir="$OPTARG"
			;;
		*)
			usage 1
			;;
	esac
done

if [ -z "$dir_to_process" ]
then
	echo "-d is a required argument" >&2
	usage 1
fi

if [ ! -d "$dir_to_process" ]
then
	echo "Directory '$dir_to_process' does not exist" >&2
	usage 1
fi

if [ -z "$save_dir" ]
then
	save_dir="$( mktemp -d )"
	echo "Created output directory $save_dir"
fi

if [ ! -d "$save_dir" ]
then
	mkdir "$save_dir"
fi

echo "Processing images from $dir_to_process to $save_dir"
[ -d "venv" ] && source venv/bin/activate
python3 scripts/pre-process.py "$dir_to_process" --outdir="$save_dir"
