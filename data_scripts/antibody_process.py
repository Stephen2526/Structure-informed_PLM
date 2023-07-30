import numpy as np
import os, re
import json
import lmdb
import pickle as pkl
from Bio import SeqIO

def generateSeqData(working_dir):
  """ prepare H-L sequence pair data
      
      * prune pairs of same pdb entity

  """
  ## load summary file
  #sumry_list = np.loadtxt('{}/summary_seq_pair.tsv'.format(working_dir),dtype='str',delimiter='\t',skiprows=1)
  sumry_list = np.loadtxt('{}/summary_seq_pair_test.tsv'.format(working_dir),dtype='str',delimiter='\t',skiprows=0)
  ## target variable
  data2save = []
  cdr_abnm_cases_H = []
  cdr_abnm_cases_L = []
  uniq_letters = []
  ## repeating entity prune
  tar_pdbId = None
  save_flag = True
  tar_seq_H = []
  tar_seq_L = []
  repeat_entity = []
  for l in range(sumry_list.shape[0]):
    pdbId, chainH, chainL, subcH, subcL  = sumry_list[l,0], sumry_list[l,1], sumry_list[l,2], sumry_list[l,3], sumry_list[l,4]
    print('>{}; {}; {}'.format(pdbId,chainH,chainL))
    if tar_pdbId != pdbId:
      ## reset
      save_flag = True
      tar_pdbId = pdbId
      tar_seq_H = []
      tar_seq_L = []
    ## extract V-region from pdb
    os.system("grep 'ATOM' {}/imgt/{}.pdb | grep -vwE 'HOH' | grep '[A-Z][A-Z][A-Z] {} \|[A-Z][A-Z][A-Z] {}[0-9]' | cut -c 18-26 | tr -s ' ' | uniq > tmpH".format(working_dir,pdbId,chainH,chainH))
    os.system("grep 'ATOM' {}/imgt/{}.pdb | grep -vwE 'HOH' | grep '[A-Z][A-Z][A-Z] {} \|[A-Z][A-Z][A-Z] {}[0-9]' | cut -c 18-26 | tr -s ' ' | uniq > tmpL".format(working_dir,pdbId,chainL,chainL))
    
    resi3_H_list = []
    cdr1_start_end_H = [] #[[start_idx,start_resi],[end_idx,end_resi]]
    cdr2_start_end_H = []
    cdr3_start_end_H = []
    with open('tmpH') as pdbH:
      first_line = pdbH.readline()
      start_res, chain_nm, start_idx = re.split(r' ',first_line)
      resi3_H_list.append(resi2single[start_res])
      #print('>>{},{},{}'.format(start_res,chain_nm,start_idx))
      if resi2single[start_res] not in uniq_letters:
        uniq_letters.append(resi2single[start_res])
      for line in pdbH:
        res_i, chain_nm, idx_i = re.split(r' ',line)
        res_sgl = resi2single[res_i]
        #print('>>{},{},{}'.format(res_sgl,chain_nm,idx_i))
        if res_sgl not in uniq_letters:
          uniq_letters.append(res_sgl)
        try:
          num_idx = int(re.findall(r'(\d+)',idx_i)[0])
          if num_idx > int(start_idx) and num_idx <= 128:
            resi3_H_list.append(res_sgl)
            if num_idx == 27:
              cdr1_start_end_H.append([len(resi3_H_list)-1,res_sgl])
            elif num_idx == 38:
              cdr1_start_end_H.append([len(resi3_H_list)-1,res_sgl])
            elif num_idx == 56:
              cdr2_start_end_H.append([len(resi3_H_list)-1,res_sgl])
            elif num_idx == 65:
              cdr2_start_end_H.append([len(resi3_H_list)-1,res_sgl])
            elif num_idx == 105:
              cdr3_start_end_H.append([len(resi3_H_list)-1,res_sgl])
            elif num_idx == 117:
              cdr3_start_end_H.append([len(resi3_H_list)-1,res_sgl])
          else:
            break
        except:
          Exception('abnormal')
    
    #print('>> H,{}:{}'.format(len(resi3_H_list),resi3_H_list))
    print('>> cdrH 1,2,3: ',cdr1_start_end_H, cdr2_start_end_H, cdr3_start_end_H)    
    
    ## check cdr abnormal
    if len(cdr1_start_end_H) != 2 or len(cdr2_start_end_H) != 2 or len(cdr3_start_end_H) != 2:
      print("cdr H abnormal")
      cdr_abnm_cases_H.append([pdbId,len(cdr1_start_end_H),len(cdr2_start_end_H),len(cdr3_start_end_H)])

    resi3_L_list = []
    cdr1_start_end_L = []
    cdr2_start_end_L = []
    cdr3_start_end_L = []
    with open('tmpL') as pdbL:
      first_line = pdbL.readline()
      start_res, chain_nm, start_idx = re.split(r' ',first_line)
      resi3_L_list.append(resi2single[start_res])
      if resi2single[start_res] not in uniq_letters:
        uniq_letters.append(resi2single[start_res])
      for line in pdbL:
        res_i, chain_nm, idx_i = re.split(r' ',line)
        res_sgl = resi2single[res_i]
        if res_sgl not in uniq_letters:
          uniq_letters.append(res_sgl)
        try:
          num_idx = int(re.findall(r'(\d+)',idx_i)[0])
          if num_idx > int(start_idx) and num_idx <= 128:
            resi3_L_list.append(res_sgl)
            if num_idx == 27:
              cdr1_start_end_L.append([len(resi3_L_list)-1,res_sgl])
            elif num_idx == 38:
              cdr1_start_end_L.append([len(resi3_L_list)-1,res_sgl])
            elif num_idx == 56:
              cdr2_start_end_L.append([len(resi3_L_list)-1,res_sgl])
            elif num_idx == 65:
              cdr2_start_end_L.append([len(resi3_L_list)-1,res_sgl])
            elif num_idx == 105:
              cdr3_start_end_L.append([len(resi3_L_list)-1,res_sgl])
            elif num_idx == 117:
              cdr3_start_end_L.append([len(resi3_L_list)-1,res_sgl])
          else:
            break
        except:
          Exception('abnormal')
    #print('>> L,{}:{}'.format(len(resi3_L_list),resi3_L_list))
    print('>> cdrL 1,2,3: ',cdr1_start_end_L, cdr2_start_end_L, cdr3_start_end_L)
    os.system('rm tmpH tmpL')

    if len(cdr1_start_end_L) != 2 or len(cdr2_start_end_L) != 2 or len(cdr3_start_end_L) != 2:
      print('cdr L abnormal')
      cdr_abnm_cases_L.append([pdbId,len(cdr1_start_end_L),len(cdr2_start_end_L),len(cdr3_start_end_L)])

    seqH = ''.join(resi3_H_list)
    seqL = ''.join(resi3_L_list)
    
    ## check repeating pdb entity
    for e in range(len(tar_seq_H)):
      if seqH == tar_seq_H[e] and seqL == tar_seq_L[e]:
        save_flag = False
        break
      else:
        save_flag = True

    if save_flag:
      ## record seq
      tar_seq_H.append(seqH)
      tar_seq_L.append(seqL)
    
      data2save.append({"pdbId": pdbId,
                        "chainH": chainH,
                        "chainL": chainL,
                        "subclassH": subcH,
                        "subclassL": subcL,
                        "seqH" : seqH,
                        "seqL" : seqL,
                        "cdr1H": cdr1_start_end_H,
                        "cdr2H": cdr2_start_end_H,
                        "cdr3H": cdr3_start_end_H,
                        "cdr1L": cdr1_start_end_L,
                        "cdr2L": cdr2_start_end_L,
                        "cdr3L": cdr3_start_end_L
                     })
    else:
      repeat_entity.append([pdbId,chainH,chainL])
    #print('**>tar_pdbId: {}'.format(tar_pdbId))
    #print('**>',tar_seq_H)
    #print('**>',tar_seq_L)

  print('total saved num:{}'.format(len(data2save)))
  for dt in data2save:
    print(dt["pdbId"],dt["chainH"],dt["chainL"])
  '''
  # save data
  np.savetxt('{}/processedData/cdr_abnm_H.csv'.format(working_dir),cdr_abnm_cases_H,fmt='%s',delimiter=',')
  np.savetxt('{}/processedData/cdr_abnm_L.csv'.format(working_dir),cdr_abnm_cases_L,fmt='%s',delimiter=',')
  np.savetxt('{}/processedData/uniq_letters'.format(working_dir),uniq_letters,fmt='%s')
  np.savetxt('{}/processedData/repeat_entity.csv'.format(working_dir),repeat_entity,fmt='%s',delimiter=',')
  
  with open('{}/processedData/HL_pair.json'.format(working_dir),'w') as fl:
    json.dump(data2save,fl)

  map_size = (1024 * 15) * (2 ** 20) # 15G
  wrtEnv = lmdb.open('{}/processedData/HL_pair.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data2save):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  '''
  return None

""" pipeline 2 for processing H-L chain sequence

    * download fasta sequence file for each pdb
    * extract chains belong to the same entity
    * use ANARCI to do index renumbering for each sequence
    * for each pdb, keep unique H-L seq pairs and extract variable region
"""

def fastaData(working_dir):
  fastaData_json = {}
  with open('{}/summary_pdbIds'.format(working_dir)) as pdbFl:
    for line in pdbFl:
      pdbId = line.replace('\n','')
      print('>pdbId: {}'.format(pdbId))
      if pdbId in obselete_pdbs:
        continue
      fastaData_json[pdbId] = [] ## pdb entity
      # load pdb fasta
      entity_count = 0
      with open('{}/pdb_fastas/{}'.format(working_dir,pdbId)) as fastaFl:
        for fasta_l in fastaFl:
          #print('!!!:{}'.format(fasta_l))
          if fasta_l[0] == '>':
            comment_line_groups = re.split(r'\|',fasta_l[1:-1])
            #print('*>comm:{}'.format(comment_line_groups))
            entity_nm = comment_line_groups[0]
            #print('*>entity_nm:{}'.format(entity_nm))
            chain_list_str = re.findall(r'Chains?(.+)',comment_line_groups[1])
            #print('*>chain_list_str:{}'.format(chain_list_str))
            chain_list = re.split(r',',chain_list_str[0])
            #print('*>chain_list:{}'.format(chain_list))
            chain_nms = []
            for chain_info in chain_list:
              #print('*>>chain_info:{}'.format(chain_info))
              chain_id = re.findall(r'\[auth\s(\w+)\]',chain_info)
              if len(chain_id) == 0:
                chain_id = re.findall(r'\s(\w+)',chain_info)
                assert len(chain_id) > 0
              chain_nms.append(chain_id[0])
            print('*>chain_nms: {}'.format(chain_nms))
            fastaData_json[pdbId].append({'entity_nm':entity_nm,
                                          'chain_nms':chain_nms})
          else:
            seq = fasta_l.replace('\n','')
            fastaData_json[pdbId][entity_count]['seq'] = seq
            #print(fastaData_json[pdbId])
            entity_count += 1
  # save json
  print('pdb num: {}'.format(len(fastaData_json.keys())))
  with open('{}/processedData/pdb_entity_chainNMs.json'.format(working_dir),'w') as jfl:
    json.dump(fastaData_json,jfl)
  
  return None

def reNumAndCut(entity_H, entity_L, H_seq,L_seq):
  # run ANARCI
  os.system('ANARCI -i {} -o anarci_outputs/{} -s imgt -r ig'.format(H_seq,entity_H))
  os.system('ANARCI -i {} -o anarci_outputs/{} -s imgt -r ig'.format(L_seq,entity_L))
  # extract variable region
  os.system("grep '^H' anarci_outputs/{} | tr -s ' ' > tmpH".format(entity_H))
  os.system("grep '^L' anarci_outputs/{} | tr -s ' ' > tmpL".format(entity_L))
  
  # target var
  seq_VH_list = []
  seq_VL_list = []
  cdr1_start_end_H = []
  cdr2_start_end_H = []
  cdr3_start_end_H = []
  cdr1_start_end_L = []
  cdr2_start_end_L = []
  cdr3_start_end_L = []
  with open('tmpH') as pdbH:
    for line in pdbH:
      line_split = re.split(r' ',line[:-1])
      if len(line_split)  == 3:
        idx_str, resi_i = line_split[1], line_split[2]
      elif len(line_split) == 4:
        idx_str, idx_exta_str, resi_i = line_split[1], line_split[2], line_split[3]
      else:
        Exception('anarchi line split error')
      num_idx = int(idx_str)
      if resi_i != '-':
        seq_VH_list.append(resi_i)
        if num_idx == 27:
          cdr1_start_end_H.append(len(seq_VH_list)-1)
        elif num_idx == 38:
          cdr1_start_end_H.append(len(seq_VH_list)-1)
        elif num_idx == 56:
          cdr2_start_end_H.append(len(seq_VH_list)-1)
        elif num_idx == 65:
          cdr2_start_end_H.append(len(seq_VH_list)-1)
        elif num_idx == 105:
          cdr3_start_end_H.append(len(seq_VH_list)-1)
        elif num_idx == 117:
          cdr3_start_end_H.append(len(seq_VH_list)-1)
        else:
          pass
  with open('tmpL') as pdbL:
    for line in pdbL:
      line_split = re.split(r' ',line[:-1])
      if len(line_split)  == 3:
        idx_str, resi_i = line_split[1], line_split[2]
      elif len(line_split) == 4:
        idx_str, idx_exta_str, resi_i = line_split[1], line_split[2], line_split[3]
      else:
        Exception('anarchi line split error')
      num_idx = int(idx_str)
      if resi_i != '-':
        seq_VL_list.append(resi_i)
        if num_idx == 27:
          cdr1_start_end_L.append(len(seq_VL_list)-1)
        elif num_idx == 38:
          cdr1_start_end_L.append(len(seq_VL_list)-1)
        elif num_idx == 56:
          cdr2_start_end_L.append(len(seq_VL_list)-1)
        elif num_idx == 65:
          cdr2_start_end_L.append(len(seq_VL_list)-1)
        elif num_idx == 105:
          cdr3_start_end_L.append(len(seq_VL_list)-1)
        elif num_idx == 117:
          cdr3_start_end_L.append(len(seq_VL_list)-1)
        else:
          pass
  os.system('rm tmpL tmpH')

  seq_VH = ''.join(seq_VH_list)
  seq_VL = ''.join(seq_VL_list)
  cdrHIdx = [cdr1_start_end_H,cdr2_start_end_H,cdr3_start_end_H]
  cdrLIdx = [cdr1_start_end_L,cdr2_start_end_L,cdr3_start_end_L]
  try:
    cdrHSeq = [seq_VH[cdr1_start_end_H[0]:cdr1_start_end_H[1]+1],
               seq_VH[cdr2_start_end_H[0]:cdr2_start_end_H[1]+1],
               seq_VH[cdr3_start_end_H[0]:cdr3_start_end_H[1]+1]]

    cdrLSeq = [seq_VL[cdr1_start_end_L[0]:cdr1_start_end_L[1]+1],
               seq_VL[cdr2_start_end_L[0]:cdr2_start_end_L[1]+1],
               seq_VL[cdr3_start_end_L[0]:cdr3_start_end_L[1]+1]]
  except:
    Exception('{} cdr abnormal'.format(pdbId))
  return seq_VH,seq_VL,cdrHIdx,cdrLIdx,cdrHSeq,cdrLSeq 

def uniqSeqReNumber(working_dir):
  # pdb entity chain_list
  with open('{}/processedData/pdb_entity_chainNMs.json'.format(working_dir)) as jfl:
    pdb_entity_chainNMs = json.load(jfl)
  # pdbId, chainH, chainL
  #sumry_list = np.loadtxt('{}/summary_seq_pair.tsv'.format(working_dir),dtype='str',delimiter='\t',skiprows=1)
  sumry_list = np.loadtxt('{}/summary_seq_pair_test.tsv'.format(working_dir),dtype='str',delimiter='\t',skiprows=0)
  
  ## target variable
  pdb_entityPair = []
  data2save = []
  uniq_letters = []
  seqwithX = []
  seqProb = []
  ## repeating entity prune
  tar_pdbId = None
  entity_pair_in_hand = [] # already included pdb entities
  for l in range(sumry_list.shape[0]):
    try:
      pdbId, chainH, chainL, subcH, subcL, lctype  = sumry_list[l,0], sumry_list[l,1], sumry_list[l,2], sumry_list[l,3], sumry_list[l,4], sumry_list[l,5]
      if pdbId in obselete_pdbs:
        continue
      if pdbId in triLetter_pdbs:
        chainH = ''.join([chainH]*3)
        chainL = ''.join([chainL]*3)
      print('>{}; {}; {}'.format(pdbId,chainH,chainL))
      if tar_pdbId != pdbId:
        ## reset
        tar_pdbId = pdbId
        entityH_in_hand = []
        entityL_in_hand = []

      # check repeating entity
      entity_list = pdb_entity_chainNMs[pdbId]
      entity_H, entity_L = None, None
      for entity_dict in entity_list:
        if chainH in entity_dict['chain_nms']:
          entity_H = entity_dict['entity_nm']
          entity_H_seq = entity_dict['seq']
        if chainL in entity_dict['chain_nms']:
          entity_L = entity_dict['entity_nm']
          entity_L_seq = entity_dict['seq']
      assert entity_H is not None
      assert entity_L is not None
      
     
      if [entity_H,entity_L] not in entity_pair_in_hand:
        ## handle X in seq
        if 'X' in entity_H_seq or 'X' in entity_L_seq:
          seqwithX.append([pdbId,entity_H,entity_L])
          entity_H_seq = entity_H_seq.replace('X','')
          entity_L_seq = entity_L_seq.replace('X','')

        entity_pair_in_hand.append([entity_H,entity_L])
        pdb_entityPair.append([pdbId,entity_H,entity_L,subcH,subcL,lctype])
        seq_VH,seq_VL,cdrHIdx,cdrLIdx,cdrHSeq,cdrLSeq = reNumAndCut(entity_H,entity_L,entity_H_seq,entity_L_seq)
        uniq_letters = list(set(list(seq_VH) + list(seq_VL) + uniq_letters))
        data2save.append({"pdbId": pdbId,
                          "entityH": entity_H,
                          "entityL": entity_L,
                          "subclassH": subcH,
                          "subclassL": subcL,
                          "seqVH" : seq_VH,
                          "seqVL" : seq_VL,
                          "cdr1HIdx": cdrHIdx[0],
                          "cdr1HSeq": cdrHSeq[0],
                          "cdr2HIdx": cdrHIdx[1],
                          "cdr2HSeq": cdrHSeq[1],
                          "cdr3HIdx": cdrHIdx[2],
                          "cdr3HSeq": cdrHSeq[2],
                          "cdr1LIdx": cdrLIdx[0],
                          "cdr1LSeq": cdrLSeq[0],
                          "cdr2LIdx": cdrLIdx[1],
                          "cdr2LSeq": cdrLSeq[1],
                          "cdr3LIdx": cdrLIdx[2],
                          "cdr3LSeq": cdrLSeq[2]
                       })
    except:
      seqProb.append([pdbId,entity_H,entity_L])
  print('total saved num:{}'.format(len(data2save)))
  
  # save data
  np.savetxt('{}/processedData/uniq_letters'.format(working_dir),uniq_letters,fmt='%s')
  np.savetxt('{}/processedData/pdb_entityPair.tsv'.format(working_dir),pdb_entityPair,fmt='%s',delimiter='\t')
  np.savetxt('{}/processedData/seq_withX.tsv'.format(working_dir),seqwithX,fmt='%s',delimiter='\t')
  np.savetxt('{}/processedData/seq_prob.tsv'.format(working_dir),seqProb,fmt='%s',delimiter='\t')
  with open('{}/processedData/HL_pair.json'.format(working_dir),'w') as fl:
    json.dump(data2save,fl)

  map_size = (1024 * 15) * (2 ** 20) # 15G
  wrtEnv = lmdb.open('{}/processedData/HL_pair.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data2save):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  
  return None

## For IMGT_3D
def findHLPair(working_dir):
  ## load pdb id list
  pdbId_list = np.loadtxt('{}/target_rec_list.tsv'.format(working_dir),dtype='str',delimiter='\t')
  pair_collect = []
  for l in range(pdbId_list.shape[0]):
    pdbId = pdbId_list[l,0]
    recp_desp_list = re.split(';',pdbId_list[l,1])
    print('>>{}'.format(pdbId))
    with open('imgt_pdbs/IMGT-{}.pdb'.format(pdbId.upper())) as fl:
      pdb_str = fl.read()
    for recp_desp in recp_desp_list:
      if 'FAB' in recp_desp.upper() or 'FV' in recp_desp.upper() or 'IG' in recp_desp.upper():
        ## find pair
        ptn = 'REMARK\s410\sIMGT\sreceptor\sdescription\nREMARK\s410\s'+recp_desp+'\s+\nREMARK\s410.+\nREMARK\s410.+\nREMARK\s410\sChain\sID\nREMARK\s410\s\w{4}_(.),\w{4}_(.)\s+\n'
        match_grp = re.findall(r'{}'.format(ptn), pdb_str)
        #print('>{}'.format(match_grp))
        if len(match_grp) > 0:
          for pair in match_grp:
            print('>{}'.format(pair))
            chain1, chain2 = pair[0], pair[1]
            ptn_fir = 'REMARK\s410\sChain\sID\s+{}_{}\s\({}{}\)\nREMARK\s410\sIMGT\schain\sdescription\s+(.+)\n'.format(pdbId,pair[0],pdbId.upper(),pair[0])
            ptn_sec = 'REMARK\s410\sChain\sID\s+{}_{}\s\({}{}\)\nREMARK\s410\sIMGT\schain\sdescription\s+(.+)\n'.format(pdbId,pair[1],pdbId.upper(),pair[1])
            match_chain1_grp = re.findall(r'{}'.format(ptn_fir),pdb_str)
            match_chain2_grp = re.findall(r'{}'.format(ptn_sec),pdb_str)
            print('>>{},{}'.format(match_chain1_grp,match_chain2_grp))
            if len(match_chain1_grp) > 0 and len(match_chain2_grp) > 0:
              # tell H / L
              if match_chain1_grp[0] in heavy_chain_desps and match_chain2_grp[0] in light_chain_desps:
                pair_collect.append([pdbId, chain1, chain2, recp_desp, match_chain1_grp[0], match_chain2_grp[0]])
              elif match_chain1_grp[0] in light_chain_desps and match_chain2_grp[0] in heavy_chain_desps:
                pair_collect.append([pdbId, chain2, chain1, recp_desp, match_chain2_grp[0], match_chain1_grp[0]])
              else:
                Exception('Sth Wrong 1')
            else:
              Exception('Sth Wrong 2')
      else:
        Exception('Sth Wrong 3')
  # save
  np.savetxt('{}/summary.tsv'.format(working_dir),pair_collect,fmt='%s',delimiter='\t',header='pdbId\tHChain\tLChain\trecp_desp\tH_desp\tL_desp')

## exp_data
def seq_extractor_batch2():
  ## uniq clonotype id
  clotype_list = np.loadtxt('exp_batch2/clonotype_list',dtype='str')
  data_list = []
  seqOutlier_list = []
  for ct in clotype_list:
    ## grep lines
    os.system('grep {} exp_batch2/20201222_antibody_amino_acid_sequences.csv > tmp_grep'.format(ct))
    with open('tmp_grep') as fl:
      grep_str = fl.read()
    os.system('rm tmp_grep')
    ## find heavy chain seq
    heavy_matches = re.findall(r'.+{},IGHV.+'.format(ct),grep_str)
    light_matches = re.findall(r'.+{},IG[L|K]V.+'.format(ct),grep_str)
    ## validate
    if len(heavy_matches) == 0 or len(light_matches) == 0:
      pass
    else:
      for heavy_line in heavy_matches:
        for light_line in light_matches:
          splitH = re.split(r',',heavy_line)
          splitL = re.split(r',',light_line)
          geneV_H = splitH[3]
          geneV_L = splitL[3]
          seqH = splitH[7]
          seqL = splitL[7]
          seqH_match = re.search(r'[\*|X]',seqH)
          seqL_match = re.search(r'[\*|X]',seqL)
          if (seqH_match is not None) or (seqL_match is not None):
            seqOutlier_list.append([heavy_line,light_line])
          else:
            seq_dict = {'clonotype': ct,
                        'geneV_heavy': geneV_H,
                        'geneV_light': geneV_L,
                        'seq_heavy': seqH,
                        'seq_light': seqL
                       }
            data_list.append(seq_dict)
  print('total {} pairs'.format(len(data_list)))
  ## save
  np.savetxt('exp_batch2/seqOutliers.csv',seqOutlier_list,fmt='%s',delimiter=',')
  with open('exp_batch2/exp_batch2_whole_seq_pair.json','w') as fl:
    json.dump(data_list,fl)

## exp_data
def seq_extractor_batch1():
  ## load barcode id
  barcode_list = np.loadtxt('exp_batch1/barcode_list',dtype='str')
  data_list = [] 
  seqOutlier_list = []
  for bc in barcode_list:
    ## find lines
    os.system('grep {} exp_batch1/20200814_Antibody_clonotype_master_list.tsv > tmp_grep'.format(bc))
    with open('tmp_grep') as fl:
      grep_str = fl.read()
    os.system('rm tmp_grep')
    ## split heavy, light
    heavy_matches = re.findall(r'.+{}.+IGHV.+'.format(bc),grep_str)
    light_matches = re.findall(r'.+{}.+IG[L|K]V.+'.format(bc),grep_str)
    ## validate
    if len(heavy_matches) == 0 or len(light_matches) == 0:
      pass
    else:
      for heavy_line in heavy_matches:
        for light_line in light_matches:
          splitH = re.split(r'\t',heavy_line)
          splitL = re.split(r'\t',light_line)
          geneV_H = splitH[10]
          geneV_L = splitL[10]
          seqIdH = splitH[0]
          seqIdL = splitL[0]
          config_idH = splitH[5]
          config_idL = splitL[5]
          ## get seq from fasta file
          seq_dictH = {rec.id : str(rec.seq) for rec in SeqIO.parse('exp_batch1/all_amino_acids/{}.fasta'.format(seqIdH), "fasta")}
          seq_dictL = {rec.id : str(rec.seq) for rec in SeqIO.parse('exp_batch1/all_amino_acids/{}.fasta'.format(seqIdL), "fasta")}
          seqH = seq_dictH.get(config_idH)
          seqL = seq_dictL.get(config_idL)
          assert (seqH is not None) and (seqL is not None)

          seqH_match = re.search(r'[\*|X]',seqH)
          seqL_match = re.search(r'[\*|X]',seqL)
          if (seqH_match is not None) or (seqL_match is not None):
            seqOutlier_list.append([heavy_line,light_line])
          else:
            seq_dict = {'barcode': bc,
                        'geneV_heavy': geneV_H,
                        'geneV_light': geneV_L,
                        'seqId_heavy': seqIdH,
                        'seqId_light': seqIdL,
                        'seq_heavy': seqH,
                        'seq_light': seqL
                       }
            data_list.append(seq_dict)
  print('total {} pairs'.format(len(data_list)))
  ## save
  np.savetxt('exp_batch1/seqOutliers.csv',seqOutlier_list,fmt='%s',delimiter=',')
  with open('exp_batch1/exp_batch1_whole_seq_pair.json','w') as fl:
    json.dump(data_list,fl)


# exp_data
def get_Vregion_cdr(batch_nm: str = '1'):
  cases_pass = ['0702-clonotype143']
  uniq_subClassPair = {}
  uniq_subClassH = {}
  uniq_subClassL = {}
  ## load H-L seq
  with open('exp_batch{}/exp_batch{}_whole_seq_pair.json'.format(batch_nm,batch_nm)) as fl:
    whole_seq_pair = json.load(fl)
  data2save = [] 
  uniq_letters = []
  for one_pair in whole_seq_pair:
    seqH = one_pair['seq_heavy']
    seqL = one_pair['seq_light']
    if batch_nm == '1':
      seqIdH = 'batch{}_H_{}'.format(batch_nm,one_pair['barcode'])
      seqIdL = 'batch{}_L_{}'.format(batch_nm,one_pair['barcode'])
      if one_pair['barcode'] in cases_pass:
        continue
    elif batch_nm == '2':
      seqIdH = 'batch{}_H_{}'.format(batch_nm,one_pair['clonotype'])
      seqIdL = 'batch{}_L_{}'.format(batch_nm,one_pair['clonotype'])
      if one_pair['clonotype'] in cases_pass:
        continue
    print('>seqIdH: {}; seqIdL: {}'.format(seqIdH,seqIdL))
    subClassH = re.split(r'[-,/]',one_pair['geneV_heavy'])[0]
    subClassL = re.split(r'[-,/]',one_pair['geneV_light'])[0]
    subClassPair = '{}-{}'.format(subClassH,subClassL)
    print('>->subClassPair: {}'.format(subClassPair))

    if subClassPair not in uniq_subClassPair.keys():
      uniq_subClassPair[subClassPair] = 1
    else:
      uniq_subClassPair[subClassPair] += 1
    
    if subClassH not in uniq_subClassH.keys():
      uniq_subClassH[subClassH] = 1
    else:
      uniq_subClassH[subClassH] += 1
    
    if subClassL not in uniq_subClassL.keys():
      uniq_subClassL[subClassL] = 1
    else:
      uniq_subClassL[subClassL] += 1
    
    seq_VH,seq_VL,cdrHIdx,cdrLIdx,cdrHSeq,cdrLSeq = reNumAndCut(seqIdH, seqIdL, seqH, seqL)
    uniq_letters = list(set(list(seq_VH) + list(seq_VL) + uniq_letters))
    data2save.append({"entityH": seqIdH,
                      "entityL": seqIdL,
                      "subclassH": subClassH,
                      "subclassL": subClassL,
                      "seqVH" : seq_VH,
                      "seqVL" : seq_VL,
                      "cdr1HIdx": cdrHIdx[0],
                      "cdr1HSeq": cdrHSeq[0],
                      "cdr2HIdx": cdrHIdx[1],
                      "cdr2HSeq": cdrHSeq[1],
                      "cdr3HIdx": cdrHIdx[2],
                      "cdr3HSeq": cdrHSeq[2],
                      "cdr1LIdx": cdrLIdx[0],
                      "cdr1LSeq": cdrLSeq[0],
                      "cdr2LIdx": cdrLIdx[1],
                      "cdr2LSeq": cdrLSeq[1],
                      "cdr3LIdx": cdrLIdx[2],
                      "cdr3LSeq": cdrLSeq[2]
                       })
  # save data
  np.savetxt('exp_batch{}/uniq_letters'.format(batch_nm),uniq_letters,fmt='%s')
  
  with open('exp_batch{}/uniq_subClassPair_count.json'.format(batch_nm),'w') as fl:
    json.dump(uniq_subClassPair,fl)
  with open('exp_batch{}/uniq_subClassH_count.json'.format(batch_nm),'w') as fl:
    json.dump(uniq_subClassH,fl)
  with open('exp_batch{}/uniq_subClassL_count.json'.format(batch_nm),'w') as fl:
    json.dump(uniq_subClassL,fl)
  
  with open('exp_batch{}/HL_pair_all.json'.format(batch_nm),'w') as fl:
    json.dump(data2save,fl)

  map_size = (1024 * 15) * (2 ** 20) # 15G
  wrtEnv = lmdb.open('exp_batch{}/HL_pair_all.lmdb'.format(batch_nm),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data2save):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()

  return None

## exp_data
def batch1_idx_barcode_bindingInhibit_map():
  '''
  exp batch1 antibody index - barcode - binding_inhinition value mapping
  '''
  bind_inhibit_avail_pairs = ['1-2','7-8','11-12','13-14','15-16','17-18','19-20','25-26','31-32','39-40','43-44','51-52','86-87','88-89','92-93','94-95','96-97','100-101','104-105','106-107','108-109','110-111','114-115','116-117','118-119']
  exp_binding_inhibit_values = [99.9,33.33,-19.97,-11.36,2.9,21.16,100,86.34,100,14.77,-0.07,62.2,85.8,59.09,100,92.78,99.52,87.97,96.63,100,99.52,99.76,91.58,93.5,-9.73]
  idx_barcode_inhibit_map = []
  for pair_i in range(len(bind_inhibit_avail_pairs)):
    idx_pair = bind_inhibit_avail_pairs[pair_i]
    bi_value = exp_binding_inhibit_values[pair_i]
    ## grep barcode
    idx1, idx2 = re.split(r'-',idx_pair)
    os.system('grep {} exp_data/exp_batch1/20200814_Antibody_clonotype_master_list.tsv | cut -f5 > tmp_bc'.format(idx1.zfill(4)))
    with open('tmp_bc') as fl:
      bc1=fl.read().replace('\n','')
    os.system('grep {} exp_data/exp_batch1/20200814_Antibody_clonotype_master_list.tsv | cut -f5 > tmp_bc'.format(idx2.zfill(4)))
    with open('tmp_bc') as fl:
      bc2=fl.read().replace('\n','')
    os.system('rm tmp_bc')
    assert bc1 == bc2
    idx_barcode_inhibit_map.append([idx_pair,bc1,bi_value])

  ## save
  np.savetxt('exp_data/exp_batch1/idx_barcode_bindInhibit.csv',idx_barcode_inhibit_map,fmt='%s',delimiter=',')


## IMGT - light chain desp
light_chain_desps = ['L-KAPPA','L-LAMBDA','V-KAPPA','V-LAMBDA']
heavy_chain_desps = ['H-ALPHA','H-ALPHA-1','H-ALPHA-2','H-DELTA','H-EPSILON','H-GAMMA','H-GAMMA-1','H-GAMMA-2','H-GAMMA-2-A','H-GAMMA-2-B','H-GAMMA-2-C','H-GAMMA-3','H-GAMMA-4','H-MU','VH-CH1','VH']


triLetter_pdbs = ['6srv','6srx','6ss0','6ss2','6ss4','6ss5','6ss6','6t9d','6t9e','6tkb','6tkc','6tkd','6tke','6tkf','6tul','6yxl','6yxm','6zh9','6zlr','6zjg']

obselete_pdbs = ['1om3','1op3','1op5','1zls','1zlu','1zlv','1zlw','2dtg','3l5y','3loh','3qos','3qot','3sm5','3w14','3wxv','3wxw','4k4m','4nx3','4q5z','4x4y','5f45','5kmv','5usi','5uwe','6erx','6mf7']

resi2single = {"GLY":"G","ALA":"A","SER":"S","ASER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
    "PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","AHIS":"H","LYS":"K","ALYS":"K","ARG":"R","UNK":"X",\
 "SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}

uniqSeqReNumber('/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/antibody/sabdab')

## IMGT_3D
#findHLPair('/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/antibody/IMGT_3D')

## exp data
#seq_extractor_batch2()
#seq_extractor_batch1()
#get_Vregion_cdr(batch_nm='1')
#get_Vregion_cdr(batch_nm='2')
#batch1_idx_barcode_bindingInhibit_map()
