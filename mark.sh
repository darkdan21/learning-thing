#!/bin/bash

usage() {
	if [ "$1" == "" ]
	then
		stat="0"
	else
		stat="$1"
	fi
	cat >&2 <<EOF
Usage: $0 [-h] -d directory_to_nn

  -h: Display this message and exit
  -d: Directory containing the save neural network
EOF
	exit $stat
}

epochs=1

while getopts "hd:" arg
do
	case $arg in
		h)
			usage 0
			;;
		d)
			dir_to_nn="$OPTARG"
			;;
		*)
			usage 1
			;;
	esac
done

if [ -z "$dir_to_nn" ]
then
	echo "-d is required argument" >&2
	usage 1
fi

if [ ! -d "$dir_to_nn" ]
then
	echo "Directory '$dir_to_nn' does not exist" >&2
	usage 1
fi

saved_model_cli run --dir $dir_to_nn --tag_set serve --signature_def serving_default --inputs image=mark.npy

