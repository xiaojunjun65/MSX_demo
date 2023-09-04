#!/bin/bash
cnmon
export PATH=/usr/local/neuware/bin:/usr/bin${PATH}
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH 
python /work/script/model-serving.py
echo "exit"
while true
do
    sleep 5
done
