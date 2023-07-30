import numpy as np
import re

unmatch_fams = np.loadtxt('unmatch_fams', dtype='str', delimiter=', ')
#print(unmatch_fams[:10,:])
famAcc_arr = unmatch_fams[:,0]
aligned_num_arr = np.array([re.split(" vs ", unmatch_fams[i,1]) for i in range(unmatch_fams.shape[0])])
counted_num = aligned_num_arr[:,0].astype(float)
actual_num = aligned_num_arr[:,1].astype(float)

#print(famAcc_arr[:10])
#print(counted_num[:10])
#print(actual_num[:10])

num_diff = counted_num - actual_num
uniq_diff, uniq_count = np.unique(num_diff, return_counts=True)
print('unique diff:{}, uniq_count:{}'.format(uniq_diff, uniq_count))
print('total mismatch fam num:{}/{}'.format(np.sum(uniq_count),famAcc_arr.shape[0]))

np.savetxt('unmatch_fams_2', np.stack((famAcc_arr,counted_num, actual_num,
  num_diff), axis=-1), fmt='%s')
