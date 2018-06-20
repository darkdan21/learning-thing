#!/bin/bash

usage() {
	if [ "$1" == "" ]
	then
		stat="0"
	else
		stat="$1"
	fi
	cat >&2 <<EOF
Usage: $0 [-h] -d directory_to_mnist [-e epochs]

  -h: Display this message and exit
  -d: Directory containing the mnist dataset
  -e: epochs to train (defaults to 1)
EOF
	exit $stat
}

epochs=1

while getopts "hd:e:" arg
do
	case $arg in
		h)
			usage 0
			;;
		d)
			dir_to_mnist="$OPTARG"
			;;
		e)
			epochs="$OPTARG"
			;;
		*)
			usage 1
			;;
	esac
done

if [ -z "$dir_to_mnist" ]
then
	echo "-d is required argument" >&2
	usage 1
fi

if [ ! -d "$dir_to_mnist" ]
then
	echo "Directory '$dir_to_mnist' does not exist" >&2
	usage 1
fi

checkpoint_dir="$( mktemp -d )"
save_dir="save"

[ -d "venv" ] && source venv/bin/activate

python3 scripts/demo-network-train.py "$dir_to_mnist" "$checkpoint_dir" --epochs "$epochs" --savedir="$save_dir"

# Clear up checkpoint directory
rm -rf "$checkpoint_dir"

