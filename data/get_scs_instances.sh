#!/usr/bin/env bash

BASEURL="https://miplib.zib.de/WebData/instances"

while read instance_name
do
	wget "${BASEURL}/${instance_name}.mps.gz" && gunzip "${instance_name}.mps.gz"
done < scs_instance_list
