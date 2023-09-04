#!/bin/bash
cnmon
export PATH=/usr/local/neuware/bin:/usr/bin${PATH}
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH

echo ${RUN_ONCE}

# sshd
mkdir -p /run/sshd
/usr/sbin/sshd

# tensorborad
tensorboard --bind_all --logdir out  &

mkdir -p /root/.config || true
mkdir -p $HOME/.config || true
cp -r /root/.config/Ultralytics $HOME/.config/

if [ "_${RUN_ONCE}_" == "__" ] ;
then
echo "============================================================="
echo $HOME
# cp /root/pretrain/*.pt $HOME/

# start trainning
python /work/script/train-default-guojun.py &

# pause
# do serve tensorboard forever
while true
do
    sleep 5
done

else
# else
python /work/script/train-default-guojun.py

fi

