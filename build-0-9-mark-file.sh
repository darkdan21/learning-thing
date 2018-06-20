#!/bin/bash

usage() {
	if [ "$1" == "" ]
	then
		stat="0"
	else
		stat="$1"
	fi
	cat >&2 <<EOF
Usage: $0 [-h] -d directory_to_process

  -h: Display this message and exit
  -d: Directory containing the pre-processed images (must be named 0.png,
  	  1.png, ..., 9.png)
EOF
	exit $stat
}

while getopts "hd:" arg
do
	case $arg in
		h)
			usage 0
			;;
		d)
			dir_to_process="$OPTARG"
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
	echo "$dir_to_process is not a directory!" >&2
	exit 1
fi

sh build-mark-file.sh $( for i in $( seq 0 9 ); do echo $dir_to_process/$i.png; done )
