### 1. create conda env ###
# INSTALL Miniconda first (https://docs.conda.io/en/latest/miniconda.html) #
# Then run the following commands to set up a conda environment #
env_name='ProLMEnv' # change to your preferred name
conda create -n ${env_name} python=3.9 -y
conda activate ${env_name}

#! please make sure the env is activated correctly
# codes below is to make sure this
if grep -q "${env_name}" <<< $(which python); then
echo "env correctly activated"
else
echo "env fail to be activated" && exit
fi


### 2. install 3rd party packages ###
conda install -c conda-forge ninja python-lmdb future matplotlib numpy pandas scikit-learn scipy seaborn filelock tensorboardx boto3 requests -y

### 3. install Pytorch ###
# pick the version you want to install #
torch_version='torch1.12.1-cu116'
if [ torch_version == 'torch1.12.1-cu113' ]
then
pip3 install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
elif [ torch_version == 'torch1.12.1-cu116' ]
then
pip3 install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
else
echo "Invalid pytorch version" && exit
fi


##-----------------------------------------------------------
##|    CUDA version            |  max supported GCC version |
##| 11.4.1+, 11.5, 11.6, 11.7  |	10.*                    |
##| 11.1, 11.2, 11.3, 11.4.0   |	9.*                     |
##| 11	                       |    8.*                     |
##-----------------------------------------------------------

### 4. install apex (support for multi-node and mixed percision training) ###
# first load cuDNN and GCC packages
# presets are given here assuming on HPRC-Grace cluster
# query the right version for cuDNN and GCC if on other clusters

if [ torch_version == 'torch1.12.1-cu113' ]
then
module load cuDNN/8.2.1.32-CUDA-11.3.1 GCC/9.3.0
elif [ torch_version == 'torch1.12.1-cu116' ]
then
module load cuDNN/8.4.1.50-CUDA-11.6.0 GCC/10.3.0
fi
apex_install_dir=$SCRATCH ## change to your preferred dir
cd $apex_install_dir
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
