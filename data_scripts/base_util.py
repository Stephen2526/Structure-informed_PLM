import lmdb, time
import numpy as np
import pickle as pkl

def read_lmdb(file_path: str = None,
              data_idx: int = 0):
    input_env = lmdb.open(f'{file_path}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
    with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        num_count = int(input_env.stat()['entries'])
        # for key, value in txn.cursor():
        #     item = pkl.loads(value)
        #     break
        # for i in range(num_examples):
        #     try:
        #         item = pkl.loads(txn.get(str(i).encode()))
        #     except:
        #         print(i)
            #print(item['mutants'],item['mut_relative_idxs'],item['fitness'],item['wt_seq'][int(item['mut_relative_idxs'][0])],item['mut_seq'][int(item['mut_relative_idxs'][0])])
        item = pkl.loads(txn.get(str(data_idx).encode()))
    print(item.keys())
    #print(item)
    print(item['seq_primary'])
    print(f'total number: {num_examples}/{num_count}')
    #print(f'{data_idx}th item: {item}')
    # map_size = (1024 * 100) * (2 ** 20) # 100G
    # env = lmdb.open(f"{file_path}_zeros.lmdb", map_size=map_size)
    # with env.begin(write=True) as txn:
    #   for i, entry in enumerate(data_list):
    #     txn.put(str(i).encode(), pkl.dumps(entry))
    #   txn.put(b'num_examples', pkl.dumps(i+1))
    # env.close()

def trim_seq_center(max_len,seq_len,mut_relative_idxs):
        """trim seq which is longer than max_len, and keep variant position at center if possible
        """
        mut_idx_min, mut_idx_max = min(mut_relative_idxs), max(mut_relative_idxs)
        if mut_idx_min < mut_idx_max: # multi-site mut
            middle_idx = (mut_idx_min + mut_idx_max) // 2
            half_size = (max_len - 1) // 2
            tmp_seg_start = middle_idx - half_size
            tmp_seg_end = middle_idx + (max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        elif mut_idx_min == mut_idx_max: #single-site mut
            half_size = (max_len - 1) // 2
            tmp_seg_start = mut_idx_min - half_size
            tmp_seg_end = mut_idx_min + (max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        else:
            Exception('Invalid mutation indices')
        return seg_start, seg_end, mut_relative_idxs

def bin_a_value(v):
        if v == np.nan:
            return -1
        bin_cutoffs = np.linspace(2,22,num=63)
        assign_bin = np.sum(v > bin_cutoffs, axis=-1)
        return assign_bin

def speed_test():
    dmap = np.random.randint(50,size=(1000,1000))
    dmap[0,0] = np.nan
    bins = np.linspace(2,22,num=63)
    start_time = time.time()
    dmap_cls_1=np.sum([dmap>cut for cut in bins],axis=0)
    print("---M1: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dmap_cls_2=np.vectorize(bin_a_value)(dmap)
    print("---M2: %s seconds ---" % (time.time() - start_time))
    print("same out? ",np.array_equal(dmap_cls_1,dmap_cls_2))

def discretize_distogram_fast(distance_map: np.ndarray,
                            first_cutoff: float,
                            last_cutoff: float,
                            num_bins: int,
                            ignore_index: int=-1):
    """discretize distance value into bins. 
    """
    nan_indices = np.nonzero(np.isnan(distance_map))
    bin_cutoffs = np.linspace(first_cutoff,last_cutoff,num=num_bins-1)
    assign_bin = np.sum([distance_map > cutof for cutof in bin_cutoffs],axis=0)
    assign_bin[nan_indices] = ignore_index
    print(bin_cutoffs)
    return assign_bin

if __name__ == '__main__':
    proj_dir = '/scratch/user/sunyuanfei/Projects'
    set_file = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames'
    # set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
    # for set_i in range(len(set_list)):
    #     set_name = set_list[set_i]
    #     print(f'set: {set_name}')
    #     read_lmdb(file_path=f'{proj_dir}/data_process/structure/shin2021/structure_features/{set_name}',data_idx=0)
    
    read_lmdb(file_path=f'{proj_dir}/ProteinMutEffBenchmark/data_process/mutagenesis/wt_seq_structure/set_structure_data/AMIE_PSEAE_Whitehead_AFDB',data_idx=0)
    
    #print(trim_seq_center(256,240,[150,160]))
    #speed_test()
    
    # dmap = np.random.rand(10,10)
    # dmap[([0,0,0],[0,3,5])] = np.nan
    # assign_bin = discretize_distogram_fast(dmap,0,2,5)
    # print(assign_bin.shape)
    # print(dmap[0])
    # print(assign_bin[0])