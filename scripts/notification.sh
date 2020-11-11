#!/bin/bash

while true; do
	str=$(condor_q|tail -n3|head -n1|cut -d" " -f1)
	if [ $str != zzhang ]; then
		mail -s "job's done" zzhang@l3s.de </dev/null;
		break
	fi
	sleep 2
	condor_q
done

