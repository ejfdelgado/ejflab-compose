#!/bin/sh

if [ "$2" = "gpu" ]; then
    echo "Using GPU..."
    export CUDA_HOME=/usr/local/cuda/
    export CUDA_PATH=/usr/local/cuda/
    export TF_ENABLE_ONEDNN_OPTS=0
    #export PATH=$PATH:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib
    #echo "Cuda Compiler?"
    #nvcc --version
    echo "Drivers?"
    nvidia-smi
    #echo "Licence"
    #cat /usr/local/cuda/gds/README
    export DEVICE=cuda
    # nvidia-smi
else
  echo "No use GPU"
  export DEVICE=cpu
fi

cd /tmp/imageia/processor-pyclient

/usr/sbin/sshd -D &
recPID2=$!
echo $recPID2

python3 -u reloadable.py $1