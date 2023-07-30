'''
* Load raw feature outputs;
* One-hot encode(except PSSM);
* For gap and unused part of sequence (in MSA), use value 0 for all encodings
* Save to npy: one example(row): 1) seq-20 (1-20), 2)ss-3 (21-23), 3)sa-2 (24-25), 4)pssm-20 (26-45)

encoded with 0
X for any amino acid
B for N or D
Z for Q or E
O for creating a free-insertion module (FIM) 
'''
import sys
sys.path.insert(0, '/home/sunyuanfei/Projects/ProteinMutEffBenchmark/scripts')
from helper_tools import translate_string_to_one_hot, pssm_profile
from helper_tools import recoverGap, recoverWTseq, recoverAlignedPssm, recoverWtPssm
import numpy as np
import re
import os

geneNm = 'BLAT_ECOLX'
proj_dir = '/home/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process'
seq_dir = proj_dir+'/sequence/folds_Evmutation'
SsSa_dir = proj_dir+'/SsSa/outputs_Evmutation'
pssm_dir = proj_dir+'/pssm/pssm_Evmutation'
save_dir = proj_dir+'/inputs/'+geneNm+'_encoded_npy'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


seq_list = list('ARNDCQEGHILKMFPSTWYV')
ss_list = list('CHE')
sa_list = list('e-')

pssm_db = 'env_nr'
## reference of all following index: 0-start wt seq
wtSeq_len = 286
msaIdxFrom = 29-1 # 0-index 
msaIdxTo = 283-1 # inclusive
insertion_in_aligned = [x+24-1 for x in [32, 214]] # [55, 269]

sets = 40
# loop through all sets
gloCount = 1
for i in range(sets):
    ssFH = open(SsSa_dir+'/BLAT_ECOLX_NoGap_Evmutation_set_'+str(i+1)+'.out.ss', 'r')
    saFH = open(SsSa_dir+'/BLAT_ECOLX_NoGap_Evmutation_set_'+str(i+1)+'.out.acc','r')
    
    #seq
    with open(seq_dir+'/BLAT_ECOLX_NoGap_Evmutation.fasta_set_'+str(i+1), 'r') as seqFl:
        count = 1
        for line in seqFl:
            if line[0] == '>':
                ## acquire gap positions
                commGp = re.split(r'/', line[:-1])
                tmpGapPos = commGp[-1]
                print('seq: ', commGp[0], commGp[1])

                raw_gapPosList = re.split(r'_', tmpGapPos)
                if len(raw_gapPosList[1]) > 0:
                    gapPosListStr = raw_gapPosList[1:]
                else:
                    gapPosListStr = []
                gapPosList = [int(n) for n in gapPosListStr]

            elif line[0] != '>':
                ## assemble feature matrix if only the pssm of this sequence exists
                pssmDir = pssm_dir+'/pssm_set_'+str(i+1)+'/seq_'+str(count)+'_ascii_pssm.'+pssm_db+'.pssm.3'
                exists = os.path.isfile(pssmDir)
                if exists:
                    ## recover gaps and unused part
                    proSeq = line[:-1]
                    pro_alignedSeq = recoverGap(proSeq, gapPosList)
                    pro_fullSeq = recoverWTseq(pro_alignedSeq, wtSeq_len, msaIdxFrom, msaIdxTo, insertion_in_aligned)
                    tmpEncode = translate_string_to_one_hot(pro_fullSeq,seq_list)
                    
                    next(ssFH)
                    ssSeq = ssFH.readline()[:-1]
                    ss_alignedSeq = recoverGap(ssSeq, gapPosList)
                    ss_fullSeq = recoverWTseq(ss_alignedSeq, wtSeq_len, msaIdxFrom, msaIdxTo, insertion_in_aligned)
                    tmpEncode = np.concatenate((tmpEncode, translate_string_to_one_hot(ss_fullSeq, ss_list)), axis=0)
                    
                    next(saFH)
                    saSeq = saFH.readline()[:-1]
                    sa_alignedSeq = recoverGap(saSeq, gapPosList)
                    sa_fullSeq = recoverWTseq(sa_alignedSeq, wtSeq_len, msaIdxFrom, msaIdxTo, insertion_in_aligned) 
                    tmpEncode = np.concatenate((tmpEncode, translate_string_to_one_hot(sa_fullSeq, sa_list)), axis=0)
                    
                    partial_pssm = pssm_profile(pssmDir)
                    aligned_pssm = recoverAlignedPssm(partial_pssm, gapPosList)
                    full_pssm = recoverWtPssm(aligned_pssm, wtSeq_len, msaIdxFrom, msaIdxTo, insertion_in_aligned)
                    tmpEncode = np.concatenate((tmpEncode, full_pssm), axis=0)
                    
                    print('matrix shape: ', tmpEncode.shape)
                    #input('pause...')

                    np.save(save_dir+'/input_'+str(i+1)+'_'+str(count)+'_'+str(gloCount)+'.npy', tmpEncode)
                    count += 1
                    gloCount += 1
