#!/bin/bash
host="https://kind-plant-0ef821803.azurestaticapps.net"
urlpathsfile="urlpath.txt"
urlpaths=$(cat $urlpathsfile)
for urlpath in $urlpaths
do
    fullurl=$host$urlpath
    echo "$fullurl"
    benchmark=$(cassowary run -u $fullurl -n 5 -c 1 -t 300)
    time=$(echo "$benchmark" | grep 'Server Processing')
    echo $time
    fail=$(echo "$benchmark" | grep 'Failed Req')
    echo $fail
done
