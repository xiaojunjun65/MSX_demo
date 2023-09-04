#!/bin/bash
cnmon
export PATH=/usr/local/neuware/bin:/usr/bin${PATH}
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
# bash timeout.sh >/dev/null 2>&1 &
echo ${RUN_ONCE}

# sshd
mkdir -p /run/sshd
/usr/sbin/sshd

# tensorborad
tensorboard  --logdir out1  &

# mkdir -p /root/.config || true
# mkdir -p $HOME/.config || true
# cp -r /root/.config/Ultralytics $HOME/.config/

# echo "============================================================="
# echo $HOME
# cp /root/pretrain/*.pt $HOME/

# start trainning
python /work/train/train.py &

# pause
# do serve tensorboard forever
while true
do
    sleep 5
done

