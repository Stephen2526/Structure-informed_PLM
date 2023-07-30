'''
1. split all training data into training and validation
2. combine all mutant seq data into one npy file
'''
import numpy as np
import os
from sklearn.model_selection import train_test_split

#root_dir = '/home/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/inputs'
testDt_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/BLAT_ECOLX_mutant'

## load data
#inputs_dir_str = root_dir+'/BLAT_ECOLX_encoded_npy'
inputs_dir_str = testDt_dir+'/BLAT_ECOLX_mutant_inputs_npy'

inputs_dir = os.fsencode(inputs_dir_str)

all_data = []
for file in os.listdir(inputs_dir):
    filename = os.fsdecode(file)
    tmpOne = np.transpose(np.load(inputs_dir_str+'/'+filename)) # 45 * L to L * 45
    all_data.append(tmpOne)


## convert to array
all_data = np.array(all_data)
print('all_data shape: ', all_data.shape)

## training/validation split
dt_all = all_data.astype('float32')
#dt_train, dt_val = train_test_split(dt_all, test_size=0.1, random_state=42)

## save to files
#save_dir = root_dir+'/BLAT_ECOLX_Tr_Val_npy'

#np.save(save_dir+'/training.npy', dt_train)
#np.save(save_dir+'/validation.npy', dt_val)
np.save(testDt_dir+'/BLAT_ECOLX_mutant.npy', dt_all)
