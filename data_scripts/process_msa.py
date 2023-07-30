'''
processing of raw msa file with .a2m format
.A2M format:
	The alignment information is encoded using uppercase and lowercase characters, and the special gap character "-". Uppercase characters and "-" represent alignment columns, and there must be exactly the same number of alignment columns in each sequence. Lowercase characters (and spaces or ".") represent insertion positions between alignment columns or at the ends of the sequence. The spaces or periods in the multiple alignments are only for human readability, and may be omitted. (https://compbio.soe.ucsc.edu/a2m-desc.html)

* remove lower-case letter, space, and '.' from sequences (the remaining Upper-case and gap'-' are aligned columns)
* remove gap for SCRATCH input
* record deleted residue positions of target sequence (0-index)
* save to *.fasta

Input: .a2m msa file
Output: two fasta files with position index of deleted residues
		one - Lowercase letter, spaces, '.' removed
		two - Gap removed based on 'one'
'''


def seqPrune(seq):
	outp_seq = '' # aligned seq
	insert_pos = [] # position reference: raw sequence
	
	for c in range(len(seq)):
		if seq[c].islower() or seq[c] == '.' or seq[c] == ' ':
			insert_pos.append(str(c))
		else:
			outp_seq += seq[c]
	return outp_seq, '_'.join(insert_pos)

def gapPrune(seq):
	outp_seq_noGap = '' # aligned seq with no gap
	gap_pos = [] # position reference: aligned MSA(BLAT_ECOLX_Evmutation.fasta)
	for c in range(len(seq)):
		if seq[c] == '-':
			gap_pos.append(str(c))
		else:
			outp_seq_noGap += seq[c]
			
	return outp_seq_noGap, '_'.join(gap_pos)

proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
humanMSA_dir = proj_dir+'/data_process/human_protein/alignments'
humanMSAList_dir = proj_dir+'/data_process/human_protein/alignments_list'

''' loop over human protein MSA list '''
with open(humanMSAList_dir, 'r') as fl:
  for li in fl:
    l = li[:-1]
    print('file: ', l)
    rawDt_dir = humanMSA_dir+'/'+l
    outp_fl = proj_dir+'/data_process/human_protein/aligned_seqs/'+l
    outp_noGap_fl =  proj_dir+'/data_process/human_protein/aligned_seqs_noGap/'+l
    
    out_f = open(outp_fl,'w')
    out_f_noGap = open(outp_noGap_fl,'w')
    
    tmp_seq = ''
    with open(rawDt_dir, 'r') as dt:
      for line in dt:
        '''
        if line[0:11]=='>BLAT_ECOLX':
          tmp_header = line[:-1]
          tmp_seq = ''
          for i in range(0,4):
            tmp_seq += dt.readline()[:-1]
          #print(len(tmp_seq))
          new_seq, prune_pos = seqPrune(tmp_seq)
          new_seq_noGap, gap_pos = gapPrune(new_seq)

          out_f.write(tmp_header+'/length_'+str(len(new_seq))+'/insertPos2raw_'+prune_pos+'\n')
          out_f.write(new_seq+'\n')

          out_f_noGap.write(tmp_header+'/length_'+str(len(new_seq_noGap))+'/gapPos2aligned_'+gap_pos+'\n')
          out_f_noGap.write(new_seq_noGap+'\n')
        '''
        if line[0]=='>':
          tmp_header = line[:-1]
          
          ## process raw sequence
          if len(tmp_seq) > 0:
            new_seq, prune_pos = seqPrune(tmp_seq)
            new_seq_noGap, gap_pos = gapPrune(new_seq)

            out_f.write(tmp_header+'/length_'+str(len(new_seq))+'/insertPos2raw_'+prune_pos+'\n')
            out_f.write(new_seq+'\n')

            out_f_noGap.write(tmp_header+'/length_'+str(len(new_seq_noGap))+'/gapPos2aligned_'+gap_pos+'\n')
            out_f_noGap.write(new_seq_noGap+'\n')

          tmp_seq = ''
        
        else: # ensemble sequence
          tmp_seq += line[:-1]

    out_f.close()
    out_f_noGap.close()
