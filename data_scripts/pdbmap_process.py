import Bio.PDB
import numpy as np
import re,os,requests,sys,random
import json
from json import JSONEncoder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import lmdb
import pickle as pkl
import seaborn as sns
import pandas as pd
from typing import Sequence, List
from sklearn.linear_model import LogisticRegression
import joblib


def uniAcc_range(pdbmap_path):
  """
  For each uniprot item
  * select non-overlapping ranges
  * among overlapping ones, select longest one
  Output:
  * uniprot, start-end_1, start-end_2, ...
  """
  # define vars
  unip_dict = {} # uni: [range1,range2,...]
  pdbmap = np.loadtxt(pdbmap_path+'/pdbmap_sets/pfam_unip_range', dtype='str',delimiter='\t')
  for ln in range(pdbmap.shape[0]):
    pfamAcc = pdbmap[ln,0][:-1]
    uniAcc = pdbmap[ln,1][:-1]
    ranStr = pdbmap[ln,2][:-1]
    print('{},{},{}'.format(pfamAcc,uniAcc,ranStr))
    idxList = re.split(r'-',ranStr)
    start_idx, end_idx = int(idxList[0]), int(idxList[1])
    iden = '{}_{}'.format(pfamAcc,uniAcc)
    if iden not in unip_dict.keys():
      unip_dict[iden] = [[start_idx,end_idx]]
    else:
      # compare range
      for pair in unip_dict[iden]:
        [st,en] = pair
        if st > end_idx or en < start_idx:
          unip_dict[iden].append([start_idx,end_idx])
        elif st >= start_idx and en <= end_idx:
          unip_dict[iden].remove(pair)
          unip_dict[iden].append([start_idx,end_idx])
        elif st <= start_idx and en >= end_idx:
          pass
        else:
          pass

  unip_list  = []
  for key,val in unip_dict.items():
    one_list = [key]
    for l in val:
      one_list.append('-'.join(l))
    unip_list.append(one_list)

  unip_arr = np.array(unip_list)
  np.savetxt('{}/pdbmap_sets/uniAcc_range.csv'.format(pdbmap_path),unip_arr,delimiter=',',fmt='%s')

def uniAcc_range_jsonVer(pdbmap_path):
  """
  For each pfam, it
  * contains multiple uniprot ids
  * for each uniprot id, it
    * contains multiple ranges
    * for each ragne
      * select the one with largest coverage
      * overlapping happens for some uniport ids
  Output:
  * pfam_id,uniprot_id,start-end_1,start-end_2, ...
  """
  # define vars
  pfam_dict = {} # {pfam1:{unp1:[range1,range2,...],unp2:[range1,...]},pfam2:{},...}
  pdbmap = np.loadtxt(pdbmap_path+'/pdbmap_sets/pfam_unip_range', dtype='str',delimiter='\t')
  #fl_save = open('{}/pdbmap_sets/uniAcc_range.csv'.format(pdbmap_path),'w')
  for ln in range(pdbmap.shape[0]):
    pfamAcc = pdbmap[ln,0][:-1]
    uniAcc = pdbmap[ln,1][:-1]
    ranStr = pdbmap[ln,2][:-1]
    print('{},{},{}'.format(pfamAcc,uniAcc,ranStr),flush=True)
    idxList = re.split(r'-',ranStr)
    start_idx, end_idx = int(idxList[0]), int(idxList[1])
    if pfamAcc not in pfam_dict.keys():
      unp_dict = {uniAcc:[[start_idx,end_idx]]}
      pfam_dict[pfamAcc] = unp_dict
    else:
      # check unp_dicts in pfam_dict
      if uniAcc not in pfam_dict[pfamAcc].keys():
        pfam_dict[pfamAcc][uniAcc] = [[start_idx,end_idx]]
      else:
        flag_add = False
        for pair_idx in pfam_dict[pfamAcc][uniAcc]:
          st,en = int(pair_idx[0]),int(pair_idx[1])
          if st > end_idx or en < start_idx:
            flag_add = True
          elif st >= start_idx and en <= end_idx:
            pfam_dict[pfamAcc][uniAcc].remove(pair_idx)
            flag_add=True
          elif st <= start_idx and en >= end_idx:
            flag_add = False
          else:
            flag_add = True
        if flag_add:
          pfam_dict[pfamAcc][uniAcc].append([start_idx,end_idx])
    print('>>>{}'.format(pfam_dict[pfamAcc][uniAcc]),flush=True)
  pfam_list  = []
  for key_pfam,val_pfam in pfam_dict.items():
    for key_unp,val_unp in pfam_dict[key_pfam].items():
      one_list = [key_pfam,key_unp]
      range_list = []
      for rans in val_unp:
        range_list.append('{}-{}'.format(rans[0],rans[1]))
      one_list.append(' '.join(range_list))
      pfam_list.append(one_list)

  pfam_arr = np.array(pfam_list)
  np.savetxt('{}/pdbmap_sets/uniAcc_range.csv'.format(pdbmap_path),pfam_arr,delimiter=',',fmt='%s')

def queryApi_bestPdb(working_dir,fl2load):
  """
  https://www.ebi.ac.uk/pdbe/graph-api/pdbe_doc/ contains various APIs
  query best structures for a uniprot with residue ranges
   - https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/:accession/:unpStart/:unpEnd
   - sorted by coverage of the protein and, if the same, resolution
   - if query with range gives no results, then only query with uniprot accession
  *** actually, query only with uniprot accession is not correct since pfam is
  defined over a region of whole pretein which means, the range matters. Quite
  amount of uniports got no responses of best pdb. This function is sufficient
  to get job done.***
  Input:
    fl2load: name of file containing pfam,unpAcc,range_list
  Output:
    bestPdb.json: json file containing best pdb and its resolution
    noResp.csv: list of (pfam,unpAcc,range) tuples for which Api returns no response
  """
  tmp_dir = 'tmp_download'
  if not os.path.isdir(tmp_dir):
    os.mkdir('tmp_download')
  
  # load pfam-uniprot targets
  tar_list = np.loadtxt('{}/pdbmap_sets/uniAcc_range/{}'.format(working_dir,fl2load),dtype='str',delimiter=',')
  record_list = []
  noRsps_list = []

  for tar in tar_list:
    pfamAcc = tar[0]
    uniAcc = tar[1]
    range_str = tar[2]
    print('{},{},{}'.format(pfamAcc,uniAcc,range_str),flush=True)
    range_list = re.split(r' ',range_str)
    #loop through range set
    for i in range(len(range_list)):
      ran_list = re.split(r'-',range_list[i])
      print('--{}'.format(ran_list),flush=True)
      start_idx, end_idx = ran_list[0], ran_list[1] #str
      api_url = 'https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/{}/{}/{}'.format(uniAcc,start_idx,end_idx)
      response = requests.get(api_url)
      # handle none response
      if len(response.text) == 2 or response.status_code != requests.codes.ok:
        print('>>>no response',flush=True)
        noRsps_list.append([uniAcc,start_idx,end_idx])
      else:
        with open('{}/{}_{}_{}'.format(tmp_dir,pfamAcc,uniAcc,range_str),'w') as fl:
          fl.write(response.text)
        #process json response
        with open('{}/{}_{}_{}'.format(tmp_dir,pfamAcc,uniAcc,range_str),'r') as res_fl:
          res_data = json.load(res_fl)
        if len(res_data[uniAcc]) < 1:
          print('>>>no items returned,trying query with only unpAcc',flush=True)
          api_url='https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures/{}'.format(uniAcc)
          response = requests.get(api_url)
          if len(response.text) == 2 or response.status_code != requests.codes.ok:
            print('>>>no response',flush=True)
            noRsps_list.append([uniAcc,start_idx,end_idx])
          else:
            with open('{}/{}_{}_{}'.format(tmp_dir,pfamAcc,uniAcc,range_str),'w') as fl:
              fl.write(response.text)
            #process json response
            with open('{}/{}_{}_{}'.format(tmp_dir,pfamAcc,uniAcc,range_str),'r') as res_fl:
              res_data = json.load(res_fl)
        else:
          pass
        best_set = res_data[uniAcc][0]
        pdb_id = best_set['pdb_id']
        print('>>>best pdb:{}'.format(pdb_id),flush=True)
        chain_id = best_set['chain_id']
        unp_start,unp_end = best_set['unp_start'], best_set['unp_end']
        resolution = best_set['resolution']
        coverage = best_set['coverage']
        record_list.append([pfamAcc,uniAcc,start_idx,end_idx,pdb_id,chain_id,unp_start,unp_end,resolution,coverage])
        os.remove('{}/{}_{}_{}'.format(tmp_dir,pfamAcc,uniAcc,range_str))
  record_list = np.array(record_list)
  #print(record_list.shape)
  # convert list to dicionary
  record_dict = {}
  for ln in range(record_list.shape[0]):
    one_record = record_list[ln,:]
    pfamAcc,uniAcc= one_record[0],one_record[1]
    if pfamAcc not in record_dict.keys():
      unp_dict = {uniAcc:[one_record[2:].tolist()]}
      record_dict[pfamAcc] = unp_dict
    else:
      if uniAcc not in record_dict[pfamAcc].keys():
        record_dict[pfamAcc][uniAcc] = [one_record[2:].tolist()]
      else:
        record_dict[pfamAcc][uniAcc].append(one_record[2:].tolist())
  #print(json.dumps(record_dict, indent = 4))
  with open('{}/pdbmap_sets/pdbmap_bestPdb/{}.json'.format(working_dir,fl2load),'w') as fl:
    json.dump(record_dict,fl)

      
  #np.savetxt('{}/pdbmap_sets/pdbmap_bestPdb.csv'.format(working_dir),record_list,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_noRsps/{}.csv'.format(working_dir,fl2load),noRsps_list,fmt='%s',delimiter=',')
  #os.rmdir('{}/tmp'.format(tmp_dir))

def bestPdb_resolution(working_dir,fl2load):
  """
  select pdb with smallest resolution as best one
  query with RSCB PDB Api(https://data.rcsb.org/#data-api)
  Input:
  * fl2load: name of file containing (pfamAcc,unpAcc,range_list)
  Output:
  * bestPdb.json: contains pdbId-chainId for each (pfamAcc,unpAcc,range) tuple 
    and its resolution, sorted with increasing resolution. For NMR pdb, use
    rama_outliers_percent. If none for both value, check published year
    pfamAcc:{
      unpAcc:{
        range:{
          'best_pdb':xxxx,
          'pdb_info':{
            pdb_id:info,
          }}}}
  * noRsp.csv: (pdbId, chainId, pfamAcc, unpAcc, range) tuple with no response from Api
  """
  tmp_dir = 'tmp_download'
  if not os.path.isdir(tmp_dir):
    os.mkdir('tmp_download')
  
  rcsbBase_url = "https://data.rcsb.org/graphql"
  # load (pfamAcc,unpAcc,range_list) tuples
  tar_list = np.loadtxt('{}/pdbmap_sets/uniAcc_range/{}'.format(working_dir,fl2load),dtype='str',delimiter=',')
  #tar_list = np.loadtxt('{}/pdbmap_sets/{}.csv'.format(working_dir,fl2load),dtype='str',delimiter=',')
  record_dict = {}
  noRsps_list = []
  obsoletePdb_list = []
  for tar in tar_list:
    pfamAcc = tar[0]
    unpAcc = tar[1]
    range_str = tar[2]
    print('{},{},{}'.format(pfamAcc,unpAcc,range_str),flush=True)
    range_list = re.split(r' ',range_str)
    #build json
    if pfamAcc not in record_dict.keys():
      unp_dict = {unpAcc:{}}
      record_dict[pfamAcc] = unp_dict
    else:
      if unpAcc not in record_dict[pfamAcc].keys():
        record_dict[pfamAcc][unpAcc] = {}
      else:
        pass 
    #loop through range list
    for i in range(len(range_list)):
      ran_list = re.split(r'-',range_list[i])
      print('--{}'.format(ran_list),flush=True)
      start_idx, end_idx = ran_list[0], ran_list[1] #str
      # extract (pdbId,chainId)
      os.system("grep '{};'$'\t''{};'$'\t''{};' {}/pdbmap | cut -d$'\t' -f1,2 \
          > {}/pdb-chain_{}_{}_{}".format(pfamAcc,unpAcc,range_list[i],working_dir,tmp_dir,pfamAcc,unpAcc,range_list[i]))
      pdb_chain_list=np.loadtxt('{}/pdb-chain_{}_{}_{}'.format(tmp_dir,pfamAcc,unpAcc,range_list[i]),dtype='str',delimiter='\t').reshape(-1,2)
      uniq_pdb_list = np.unique(pdb_chain_list[:,0])
      pdb_resolution = []
      for ele in uniq_pdb_list:
        pdb = ele[:-1]
        #print('{}'.format(pdb),flush=True)
        # query resolution
        query_resolution = '''
          {{entries(entry_ids: ["{}"]) {{
              rcsb_id
              pdbx_vrpt_summary{{
                PDB_resolution
                percent_ramachandran_outliers_full_length
              }}
              rcsb_entry_info{{
                resolution_combined
              }}
              citation{{
                year
              }}
            }}
          }}
          '''.format(pdb)
        res_rslu = requests.post(rcsbBase_url,json={'query':query_resolution})
        rslu_value, rama_outlier, year = None, None, None
        rslu_value_list, rama_outlier_list, year_list = None, None, None
        if res_rslu.status_code != 200:
          noRsps_list.append([pdb,pfamAcc,unpAcc,range_list[i]])
        else:
          res_rslu_json = res_rslu.json()
          # obsolete pdb 
          if len(res_rslu_json['data']['entries']) == 0:
            rslu_value = '-2'
            rama_outlier = '-2'
            year = '0'
          else:
            #rslu_value=res_rslu_json['data']['entries'][0]['pdbx_vrpt_summary']['PDB_resolution']
            rslu_value_list=res_rslu_json['data']['entries'][0]['rcsb_entry_info']['resolution_combined']
            rama_outlier_list = res_rslu_json['data']['entries'][0]['pdbx_vrpt_summary']
            year_list = res_rslu_json['data']['entries'][0]['citation']
            # check resolution and rama_outlier value
            if year_list is None:
              year = '0'
            else:
              if year_list[0]['year'] is None:
                year = '0'
              else:
                year = year_list[0]['year']
            if rslu_value_list is None and rama_outlier_list is None:
              rslu_value = '-1'
              rama_outlier = '-1'
            elif rslu_value_list is None and rama_outlier_list is not None:
              rslu_value = '-1'
              tmp_rama = rama_outlier_list['percent_ramachandran_outliers_full_length']
              if tmp_rama is None:
                rama_outlier = '-1'
              else:
                rama_outlier = tmp_rama
            elif rslu_value_list is not None:
              rslu_value = rslu_value_list[0]
              rama_outlier = '-1'
          pdb_resolution.append([pdb,rslu_value,rama_outlier,year])
      os.remove('{}/pdb-chain_{}_{}_{}'.format(tmp_dir,pfamAcc,unpAcc,range_list[i]))

      # select pdb with smallest resolution
      pdb_resolution = np.array(pdb_resolution).reshape(-1,4)
      print(pdb_resolution)
      if pdb_resolution.shape[0] == 1: # only one pdb
        if float(pdb_resolution[0,1]) == -2: # the only pdb is obsolete
          obsoletePdb_list.append([pfamAcc,unpAcc,range_list[i]])
          best_pdb = "obsolete"
        else:
          best_pdb = pdb_resolution[0,0]
      else:
        obso_idx = np.where(pdb_resolution[:,1].astype(float) == -2.)[0]
        if len(obso_idx) == pdb_resolution.shape[0]:
          # all pdbs are obsolete
          obsoletePdb_list.append([pfamAcc,unpAcc,range_list[i]])
          best_pdb = "obsolete"
        else:
          valid_idx = np.where(pdb_resolution[:,1].astype(float) > 0.)[0]
          # in case all pdb entries have no resolution param(like their methods are 'solution NMR')
          #print('valid:{}'.format(valid_idx))
          if len(valid_idx) == 0:
            nmr_idx = np.where(pdb_resolution[:,2].astype(float) > 0.)[0]
            #print('nmr:{}'.format(nmr_idx))
            if len(nmr_idx) == 0: # all pdbs have no resolution and rama_outlier_percent
              # pick latest pdb which is not obsolete
              nonObso_idx = np.where(pdb_resolution[:,1].astype(float) == -1.)[0]
              min_idx = np.argmax(pdb_resolution[nonObso_idx,3].astype(int))
              best_pdb=pdb_resolution[nonObso_idx[min_idx],0]
            else: # select the one with smallest percent_ramachandran_outliers
              min_idx = np.argmin(pdb_resolution[nmr_idx,2].astype(float))
              best_pdb = pdb_resolution[nmr_idx[min_idx],0]
          else:
            min_idx = np.argmin(pdb_resolution[valid_idx,1].astype(float))
            best_pdb = pdb_resolution[valid_idx[min_idx],0]
      # build json
      if range_list[i] not in record_dict[pfamAcc][unpAcc].keys():
        pdb_rslu_dict = {}
        for ele in pdb_resolution:
          if float(ele[1]) == -1:
            if float(ele[2]) == -1:
              pdb_rslu_dict[ele[0]] = int(ele[3])
            else: 
              pdb_rslu_dict[ele[0]] = "{}%".format(ele[2])
          else:
            pdb_rslu_dict[ele[0]] = float(ele[1])
        record_dict[pfamAcc][unpAcc][range_list[i]] = {"best_pdb":best_pdb,
                                                       "pdb_info":pdb_rslu_dict}
      else:
        pass
  with open('{}/pdbmap_sets/pdbmap_bestPdb/{}.json'.format(working_dir,fl2load),'w') as fl:
  #with open('{}/pdbmap_sets/pdbmap_bestPdb.json'.format(working_dir),'w') as fl:
    json.dump(record_dict,fl)
  np.savetxt('{}/pdbmap_sets/pdbmap_noRsps/{}.csv'.format(working_dir,fl2load),noRsps_list,fmt='%s',delimiter=',')
  #np.savetxt('{}/pdbmap_sets/pdbmap_noRsps.csv'.format(working_dir),noRsps_list,fmt='%s',delimiter=',')
  #np.savetxt('{}/pdbmap_sets/pdbmap_obsolete.csv'.format(working_dir),obsoletePdb_list,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_obsolete/{}.csv'.format(working_dir,fl2load),obsoletePdb_list,fmt='%s',delimiter=',')

def unpAcc2Nm(unpAcc):
  """

 
  https://www.uniprot.org/uniprot/P20345.fasta
  """
  return unpNm

def query_unpSeq(unpAcc):
  query_url = 'https://www.uniprot.org/uniprot/{}.fasta'.format(unpAcc)
  res_obj = requests.get(query_url)
  fasta_str = res_obj.text
  seq = ""
  segments = re.split(r'\n',fasta_str)
  for seg in segments[1:]:
    seq += seg
  return seq

def queryApi_pdbInfo(working_dir,pdb_id,chain_id,unpAcc):
  """
  Descriptions:
    query information from RCSB PDB Data API & RCSB PDB 1D Coordinate Server API
    * Residue index mappings between author provided and pdb sequence positions
    * Sequence alignment between seq of uniprot and pdb
    * Unmodelled regions(residues;atoms)

  Highlights:
    When requesting data for multiple objects compound identifiers should follow the format:
    * [pdb_id]_[entity_id] - for polymer, branched, or non-polymer entities (e.g. 4HHB_1)
    * [pdb_id].[asym_id] - for polymer, branched, or non-polymer entity instances (e.g. 4HHB.A)
      * chain ID corresponds to _label_asym_id in PDBx/mmCIF schema
    * [pdb_id]-[assembly_id] - for biological assemblies (e.g. 4HHB-1)
  
  Inputs:
    * pdb_id
    * chain_id(asym_id, not author defined)
    * unpAcc: uniprot accesion
  
  Outputs:
    * hasRes: True - has response; False - no response
    * auth_pdbSeq_mapping: list,author defined residue indices from start to end, e.g. ['-3','-2',.,'1','2',.'40','1000','1001',.,'1020','65','66',...]
    * unp_seq: str,uniprot seq
    * pdb_seq: str,pdb seq
    * aligned_regions: list of dictionary, each dictionary contains mapping from pdb (query_begin,query_end) to uniprot (target_begin,target_end), e.g. [{"query_begin": 5,"query_end": 232,"target_begin": 1,"target_end": 228},{"query_begin": 338,"query_end": 422,"target_begin": 335,"target_end": 419}]
    * unobserved_residues: list of dict, each dict has keys "beg_seq_id","end_seq_id" (pdb seq index)
    * unobserved_atoms: list of dict, each dict has keys "beg_seq_id","end_seq_id" (pdb seq index)
  """
  rcsbBase_url = "https://data.rcsb.org/graphql"
  rcsb1d_url = "https://1d-coordinates.rcsb.org/graphql"
  pdb_instance = '{}.{}'.format(pdb_id,chain_id)
  """
  # need a testing query before every thing since chain id may not be correct for a query
  query_test ='''
  {{polymer_entity_instances(instance_ids: ["{pdb_ins}"]) {{
    rcsb_id
    }}
  }}
  '''.format(pdb_ins=pdb_instance)
  res_test = requests.post(rcsbBase_url,json={'query':query_test})
  if res_test.status_code == 200:
    res_test_json = res_test.json()
    if len(res_test_json['data']['polymer_entity_instances']) == 0:
      chain_id = 'A'
    else:
      pass
  else:
    pass
  # update pdb instance name
  pdb_instance = '{}.{}'.format(pdb_id,chain_id)
  """

  query_idxMap = '''
  {{polymer_entity_instances(instance_ids: ["{pdb_ins}"]) {{
    rcsb_id
    rcsb_polymer_entity_instance_container_identifiers {{
      auth_asym_id
      entity_id
      auth_to_entity_poly_seq_mapping}}
    }}
  }}
  '''.format(pdb_ins=pdb_instance)
  query_align = '''
  {{alignment(from:PDB_INSTANCE,to:UNIPROT,queryId:"{}"){{
    query_sequence
    target_alignment {{
      target_id
      target_sequence
      aligned_regions {{
        query_begin
        query_end
        target_begin
        target_end}}
    }}
   }}
  }}
  '''.format(pdb_instance)
  query_unmodel = '''
  {{annotations(reference:PDB_INSTANCE,sources:[UNIPROT,PDB_ENTITY,PDB_INSTANCE],queryId:"{}",
                filters:[{{field:type
                           operation:contains
                           values:["UNOBSERVED_RESIDUE_XYZ","UNOBSERVED_ATOM_XYZ"]
                         }}])
    {{target_id
      features {{
        feature_id
        description
        name
        provenance_source
        type
        feature_positions{{
          beg_seq_id
          end_seq_id}}
      }}
    }}
  }}

  '''.format(pdb_instance)
  res_idxMap = requests.post(rcsbBase_url,json={'query':query_idxMap})
  res_align = requests.post(rcsb1d_url,json={'query':query_align})
  res_unmodel = requests.post(rcsb1d_url,json={'query':query_unmodel}) 
  auth_pdbSeq_mapping,unp_seq,pdb_seq,aligned_regions,unobserved_residues,unobserved_atoms=None,None,None,None,None,None
  # extract info from response
  if res_idxMap.status_code != 200 or res_align.status_code != 200 or res_unmodel.status_code != 200:
    return False,None,None,None,None,None,None
  else:
    res_idxMap_json,res_align_json,res_unmodel_json=res_idxMap.json(),res_align.json(),res_unmodel.json()
    auth_pdbSeq_mapping=res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_to_entity_poly_seq_mapping']
    pdb_seq=res_align_json['data']['alignment']['query_sequence']
    # one pdb seq could have more than 1 unp correspondence
    for d in res_align_json['data']['alignment']['target_alignment']:
      if d['target_id'] == unpAcc:
        unp_seq=d['target_sequence']
        aligned_regions=d['aligned_regions']
    if unp_seq is None: 
      # no such unpAcc under this pdb,
      pass
    # loop over unmodelled res and atom
    if len(res_unmodel_json['data']['annotations']) == 0:
      unobserved_atoms,unobserved_residues = None,None
    else:
      for d in res_unmodel_json['data']['annotations'][0]['features']:
        if d['type'] == 'UNOBSERVED_ATOM_XYZ':
          unobserved_atoms=d['feature_positions']
        elif d['type'] == 'UNOBSERVED_RESIDUE_XYZ':
          unobserved_residues=d['feature_positions']
        else:
          pass
    return True,auth_pdbSeq_mapping,unp_seq,pdb_seq,aligned_regions,unobserved_residues,unobserved_atoms

def pdb_unp_align(pdb_seq,unp_seq):
  return 0

class NumpyArrayEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)

def asym_mapping(best_pdb):
  """
  generate mapping between natural chain id and author defined chain id
  
  *Input:
   - best_pdb id
  *Output:
   - asym_mapping_dict: {auth_asym_id:asym_id}
  """
  asym_mapping_dict = {}
  rcsbBase_url = "https://data.rcsb.org/graphql"
  # query entity ids
  entityIds_query = '''
    {{entries(entry_ids: ["{}"]) {{
        rcsb_entry_container_identifiers {{
          polymer_entity_ids}}
      }}
    }}	
  '''.format(best_pdb)
  res_entityIds = requests.post(rcsbBase_url,json={'query':entityIds_query})
  if res_entityIds.status_code != 200:
    return None
  else:
    try:
      entityIds_list = res_entityIds.json()['data']['entries'][0]['rcsb_entry_container_identifiers']['polymer_entity_ids']
      for ent_id in entityIds_list:
        # query asym_ids, auth_asym_ids
        asymIds_query = '''
          {{polymer_entities(entity_ids:["{}_{}"])
              {{rcsb_polymer_entity_container_identifiers {{
                  asym_ids
                  auth_asym_ids}}
                entity_poly {{
                  pdbx_strand_id}}
              }}
          }}
        '''.format(best_pdb,ent_id)
        res_asymIds = requests.post(rcsbBase_url,json={'query':asymIds_query})
        if res_asymIds.status_code != 200:
          return None
        else:
          rec_asymIds_json = res_asymIds.json()
          asymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['asym_ids']
          #authAsymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] # auth_asym_ids may not have the right order of ids
          pdbx_strandId_list = re.split(r',',rec_asymIds_json['data']['polymer_entities'][0]['entity_poly']['pdbx_strand_id'])
          assert len(asymIds_list) == len(pdbx_strandId_list), "asym_ids length not same with auth_asym_ids"
          # upate mapping dict
          for asym_i in range(len(asymIds_list)):
            asym_mapping_dict[pdbx_strandId_list[asym_i]] = asymIds_list[asym_i]
      return asym_mapping_dict
    except:
      return None

def kth_diag_indices(arr,k_list):
  """
  return indices of k-th diagnals

  Input:
  * arr: array of size (N,N)
  * k: list of k-th diagonals, e.g. [-1,0,1]
  """
  rowIdxs,colIdxs = np.diag_indices_from(arr)
  out_row_idxs = []
  out_col_idxs = []
  for k in k_list:
    if k < 0:
      out_row_idxs.extend(rowIdxs[-k:])
      out_col_idxs.extend(colIdxs[:k])
    elif k > 0:
      out_row_idxs.extend(rowIdxs[:-k])
      out_col_idxs.extend(colIdxs[k:])
    else:
      out_row_idxs.extend(rowIdxs)
      out_col_idxs.extend(colIdxs)
  return out_row_idxs,out_col_idxs

def pfam_contact(working_dir, uniAcc_range_nm):
  """
  generate true binary contact map for seq with known pdb and
  infer contact map for seqs in the same pfam from alignment

  Cases happen like for one pfam:
  * only one seq has known pdb
  * more than one seqs have known pdbs
    * majority voting for each pair of positions
  * one pdb covers the whole seq range
  * one pdb only cover partial seq range
    * overlapping happens across pdbs
    * each pdb coverd range is one training example(one pfam generate multiple examples)  
  Cutoff
  * CB-CB, 8A
  Inputs:
  *  pfam msa file
  *  best pdb list for each pfam

  Outputs:
  * contactData: list, each element is a dictionary for one seq
  """

  '''
  # load pfam msa file name - pfamAcc correspondence
  pfam_flNm = np.loadtxt('{}/pfam_rp{}_seqs_files_famAcc'.format(working_dir,rpLevel),dtype='str',delimiter=' ')
  pfamAcc_list = np.array([re.split(r'.',itm)[0] for itm in pfam_flNm[:,1]])
  '''
  # load (pfamAcc,unpAcc,range_list) tuple
  pfam_unp_range = np.loadtxt('{}/pdbmap_sets/uniAcc_range/{}'.format(working_dir,uniAcc_range_nm),dtype='str',delimiter=',')
  # load best pdb list for pfam
  with open('{}/pdbmap_sets/pdbmap_bestPdb/{}.json'.format(working_dir,uniAcc_range_nm),'r') as fl:
    bestPdb_dict = json.load(fl)
  noRsp_pdb = []
  noAlign_pdb = []
  noValidReg_pdb = []
  noAsymId_pdb = []
  noAnyErr_pdb = []
  obsolete_pdb = []
  noExpAtom_pdb = []
  packed_data = []
  cutoff = 8.0
  exp_num = 0
  fail_expNum = 0
  bio_pdbList = Bio.PDB.PDBList()
  #bio_pdbParser = Bio.PDB.PDBParser()
  bio_mmcifParser = Bio.PDB.FastMMCIFParser(QUIET=True)
  for tar in pfam_unp_range:
    pfamAcc = tar[0]
    unpAcc = tar[1]
    range_str = tar[2]
    print('{},{},{}'.format(pfamAcc,unpAcc,range_str),flush=True)
    range_list = re.split(r' ',range_str)
    for ran in range_list:
      try:
        print('--:{}'.format(ran))
        ran_split = re.split(r'-',ran)
        unp_startIdx,unp_endIdx=int(ran_split[0]),int(ran_split[1])
        best_pdb = bestPdb_dict[pfamAcc][unpAcc][ran]['best_pdb']
        print('>>>{}'.format(best_pdb))
        if best_pdb == 'obsolete':
          fail_expNum += 1
          obsolete_pdb.append([pfamAcc,unpAcc,ran])
        else:
          # obtain chain Id
          #re.search(r"{};\s+([A-Z]);.+{};\s+{};\s+{}".format(best_pdb,pfamAcc,unpAcc,ran),)
          os.system("grep '{};'$'\t''{};'$'\t''{}' {}/pdbmap | grep {} | head -n 1 | cut -d$'\t' -f2 | cut -d';' -f1 > tmp_fl_{}".format(pfamAcc,unpAcc,ran,working_dir,best_pdb,uniAcc_range_nm))
          with open('tmp_fl_{}'.format(uniAcc_range_nm), 'r') as chain_fl:
            chain_id = chain_fl.read()[:-1]
          os.remove('tmp_fl_{}'.format(uniAcc_range_nm))
          print('>>>chain:{}'.format(chain_id))
          # query mapping between pdb instance asym_id and auth_asym_id
          asym_mapping_dict = asym_mapping(best_pdb)
          if asym_mapping_dict is None:
            noAsymId_pdb.append([best_pdb,chain_id,pfamAcc,unpAcc,ran])
            # make a dumb asym_mapping_dict
            asym_mapping_dict = {chain_id:chain_id}

          # query info about pdb
          res_flag,auth_pdbSeq_mapping,unp_seq,pdb_seq,aligned_regions,unobserved_residues,unobserved_atoms=queryApi_pdbInfo(working_dir,best_pdb,asym_mapping_dict[chain_id],unpAcc)
          if not res_flag: # no response
            fail_expNum += 1
            noRsp_pdb.append([best_pdb,chain_id,pfamAcc,unpAcc,ran])
            continue
          else:
            # fetch pdb file and generate pdb object
            pdb_flNm = bio_pdbList.retrieve_pdb_file(best_pdb,pdir='tmp_download/{}_pdb'.format(uniAcc_range_nm),file_format='mmCif',overwrite=True)
            #pdb_struc = bio_pdbParser.get_structure(best_pdb,'{}'.format(pdb_flNm))
            pdb_struc = bio_mmcifParser.get_structure(best_pdb,'{}'.format(pdb_flNm))
            pdb_model = pdb_struc[0]
            #os.remove(pdb_flNm)

            # build residue id dict
            resiId_dict={}
            for resi_obj in pdb_model[chain_id]:
              resiId_tuple = resi_obj.get_id()
              if resiId_tuple[2] != ' ':
                resiId_dict['{}{}'.format(resiId_tuple[1],resiId_tuple[2])] = resiId_tuple
              else:
                resiId_dict['{}'.format(resiId_tuple[1])] = resiId_tuple

            if aligned_regions is None: # this unpAcc not covered by this pdb
              fail_expNum += 1
              # get unp_seq, do seq alignment
              noAlign_pdb.append([best_pdb,chain_id,pfamAcc,unpAcc,ran])
              #unp_seq = query_unpSeq(unpAcc)
              #aligned_range = pdb_unp_align(pdb_seq,unp_seq)
              continue
            else:
              unp_pdb_seqIdx_mapping = get_unp_pdb_seqIdx_mapping(aligned_regions)
              unmodelResi_pdbIdxs,unmodelAtom_pdbIdxs = unmodel_pdb_idx(unobserved_residues,unobserved_atoms)
              valid_unpIdx_list = check_valid_pos(unp_startIdx,unp_endIdx,aligned_regions,unmodelResi_pdbIdxs,unp_pdb_seqIdx_mapping)
              if len(valid_unpIdx_list) > 0:
                # loop over multiple valid regions of the pdb
                print('>>>>>val_region:{}-{},len:{}'.format(valid_unpIdx_list[0],valid_unpIdx_list[-1],len(valid_unpIdx_list)))
                # build json
                tmp_data_dict = {} 
                tmp_data_dict['pfamAcc'] = pfamAcc
                tmp_data_dict['unpAcc'] = unpAcc
                tmp_data_dict['range'] = ran
                # uniprot seq is used as target seq
                seq_tar = ""
                unpSeq_len = len(unp_seq)
                unp_pfam_range = range(unp_startIdx, min(unp_endIdx,unpSeq_len)+1) # index should not exceed unp seq length
                unp_pfam_range_len = len(unp_pfam_range)
                for unpIdx_i in unp_pfam_range:
                  seq_tar += unp_seq[unpIdx_i-1]
                tmp_data_dict['unp_pfam_range'] = '{}-{}'.format(unp_pfam_range[0],unp_pfam_range[-1])
                tmp_data_dict['target_seq'] = seq_tar
                tmp_data_dict['targetSeq_len'] = unp_pfam_range_len
                tmp_data_dict['best_pdb'] = best_pdb
                tmp_data_dict['chain_id'] = chain_id
                tmp_data_dict['valid_unpIdxs_len'] = len(valid_unpIdx_list)
                tmp_data_dict['valid_unpIdxs'] = valid_unpIdx_list
                # generate contact-map (with self-self, self-neighbor as 1)
                contact_mat = np.zeros((unp_pfam_range_len,unp_pfam_range_len))
                diagIdx = kth_diag_indices(contact_mat,[-1,0,1])
                contact_mat[diagIdx] = 1
                # loop rows and cols
                for valReg_row in range(unp_pfam_range_len):
                  for valReg_col in range(unp_pfam_range_len):
                    unp_idx_row = unp_pfam_range[valReg_row]
                    unp_idx_col = unp_pfam_range[valReg_col]
                    if unp_idx_row in valid_unpIdx_list and unp_idx_col in valid_unpIdx_list:
                      # get pdb natural seq idx(1-idxed), then author-defined index
                      resIdx_pdbSeq_row = unp_pdb_seqIdx_mapping[unp_idx_row]
                      resIdx_pdbSeq_col = unp_pdb_seqIdx_mapping[unp_idx_col]
                      resIdx_pdbAuth_row = str(auth_pdbSeq_mapping[resIdx_pdbSeq_row-1])
                      resIdx_pdbAuth_col = str(auth_pdbSeq_mapping[resIdx_pdbSeq_col-1])
                      if 'CB' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
                        atomNm_row = 'CB'
                      elif 'CA' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
                        atomNm_row = 'CA'
                      elif 'CB1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
                        atomNm_row = 'CB1'
                      elif 'CA1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
                        atomNm_row = 'CA1'
                      else:
                        '''
                        # debug
                        print(resiId_dict[resIdx_pdbAuth_row])
                        for atm in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
                          print(atm.get_name)
                        '''
                        #raise Exception("No atoms CB,CA,CB1,CA1")
                        noExpAtom_pdb.append([best_pdb,chain_id,resiId_dict[resIdx_pdbAuth_row],pfamAcc,unpAcc,ran])
                        # set as no-contact if CB,CA,CB1,CA1 not exist
                        continue

                      if 'CB' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
                        atomNm_col = 'CB'
                      elif 'CA' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
                        atomNm_col = 'CA'
                      elif 'CB1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
                        atomNm_col = 'CB1'
                      elif 'CA1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
                        atomNm_col = 'CA1'
                      else:
                        '''
                        # debug
                        print(resiId_dict[resIdx_pdbAuth_col])
                        for atm in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
                          print(atm.get_name)
                        '''
                        #raise Exception("No atoms CB,CA,CB1,CA1")
                        noExpAtom_pdb.append([best_pdb,chain_id,resiId_dict[resIdx_pdbAuth_col],pfamAcc,unpAcc,ran])
                        # set as no-contact if CB,CA,CB1,CA1 not exist
                        continue

                      #print('{},{}'.format(resiId_dict[resIdx_pdbAuth_row],resiId_dict[resIdx_pdbAuth_col]))
                      if pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]][atomNm_row]-pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]][atomNm_col] <= cutoff:
                        contact_mat[valReg_row][valReg_col] = 1
                      else:
                        continue
                #build json
                tmp_data_dict['contact-map'] = contact_mat
                packed_data.append(tmp_data_dict)
                exp_num += 1
              else:
                fail_expNum += 1
                noValidReg_pdb.append([best_pdb,chain_id,pfamAcc,unpAcc,ran])
      except BaseException as err:
        print('>>>Exception occurs: {}'.format(err))
        fail_expNum += 1
        noAnyErr_pdb.append([best_pdb,chain_id,pfamAcc,unpAcc,ran,str(err)])
  print('Processed {}, success: {}'.format(fail_expNum+exp_num,exp_num))
  print('Exception occurs: obso_pdb:{},noRsp_pdb-{},noAlign_pdb-{},noValidReg_pdb-{},noAsymId_pdb-{},noSomeErr_pdb-{}'.format(len(obsolete_pdb),len(noRsp_pdb),len(noAlign_pdb),len(noValidReg_pdb),len(noAsymId_pdb), len(noAnyErr_pdb)))
  # save all data
  with open('{}/pdbmap_sets/pdbmap_contactData/{}.json'.format(working_dir,uniAcc_range_nm),'w') as fl:
    json.dump(packed_data,fl,cls=NumpyArrayEncoder)
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoRsps/{}.csv'.format(working_dir,uniAcc_range_nm),noRsp_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoAligns/{}.csv'.format(working_dir,uniAcc_range_nm),noAlign_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoValids/{}.csv'.format(working_dir,uniAcc_range_nm),noValidReg_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoAsymIds/{}.csv'.format(working_dir,uniAcc_range_nm),noAsymId_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoAnyErrs/{}.csv'.format(working_dir,uniAcc_range_nm),noAnyErr_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactObso/{}.csv'.format(working_dir,uniAcc_range_nm),obsolete_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactNoExpAtoms/{}.csv'.format(working_dir,uniAcc_range_nm),noExpAtom_pdb,fmt='%s',delimiter=';')

def unmodel_pdb_idx(unobserved_residues,unobserved_atoms):
  unmodelResi_pdbIdx_list,unmodelAtom_pdbIdx_list = [],[]
  if unobserved_residues is not None:
    for resi_idx_dict in unobserved_residues:
      beg_seq_id = resi_idx_dict['beg_seq_id']
      end_seq_id = resi_idx_dict['end_seq_id']
      for i in range(beg_seq_id,end_seq_id+1):
        unmodelResi_pdbIdx_list.append(i)
  
  if unobserved_atoms is not None:
    for atom_idx_dict in unobserved_atoms:
      beg_seq_id = atom_idx_dict['beg_seq_id']
      end_seq_id = atom_idx_dict['end_seq_id']
      for i in range(beg_seq_id,end_seq_id+1):
        unmodelAtom_pdbIdx_list.append(i)
  
  return unmodelResi_pdbIdx_list,unmodelAtom_pdbIdx_list
   
def get_unp_pdb_seqIdx_mapping(aligned_regions):
  """
  generate unp-pdb seq index mapping of aligned region
  Output:
  - unp_pdb_seqIdx_dict: e.g. {1(unp):3(pdb),2:4,3:5,...}
  """
  unp_pdb_seqIdx_dict = {}
  for align_dict in aligned_regions:
    pdb_begin = int(align_dict['query_begin'])
    pdb_end = int(align_dict['query_end'])
    unp_begin = int(align_dict['target_begin'])
    unp_end = int(align_dict['target_end'])
    assert (pdb_end - pdb_begin) == (unp_end - unp_begin), '>>>aligned region not same length'
    pdb_idx = pdb_begin
    for unp_idx in range(unp_begin,unp_end+1):
      unp_pdb_seqIdx_dict[unp_idx] = pdb_idx
      pdb_idx += 1
  
  return unp_pdb_seqIdx_dict

def compare_two_range(unp_startIdx:int,unp_endIdx:int,unp_begin:int,unp_end:int):
  """
  return intersection of two ranges
  """
  if unp_endIdx <= unp_begin or unp_startIdx >= unp_end:
    return None
  elif unp_startIdx <= unp_begin and unp_endIdx >= unp_end:
    return [unp_begin,unp_end]
  elif unp_startIdx >= unp_begin and unp_endIdx <= unp_end:
    return [unp_startIdx,unp_endIdx]
  elif unp_startIdx >= unp_begin and unp_startIdx <= unp_end:
    return [unp_startIdx,unp_end]
  elif unp_endIdx >= unp_begin and unp_endIdx <= unp_end:
    return [unp_begin,unp_endIdx]
  else:
    return None

def check_valid_pos(unp_startIdx:int,unp_endIdx:int,aligned_regions:list,unmodelResi_pdbIdxs:list,unp_pdb_seqIdx_mapping:dict):
  """
  compare pfam provided unp_range with the unp_range from aligned_regions to
  select intersection region, indices of unmodelled residues are not included
  Output:
  - valid_unpIdx_list:, uniprot indices of modelled residues in intersection of pfam unp_range and pdb-unp aligned region.
  """
  valid_unpIdx_list = []     
  for align_dict in aligned_regions:
    unp_begin = int(align_dict['target_begin'])
    unp_end = int(align_dict['target_end'])
    val_region_start_end = compare_two_range(unp_startIdx,unp_endIdx,unp_begin,unp_end)
    if val_region_start_end is not None:
      for i in range(val_region_start_end[0],val_region_start_end[1]+1):
        pdb_i = unp_pdb_seqIdx_mapping[i]   
        if pdb_i not in unmodelResi_pdbIdxs:
          valid_unpIdx_list.append(i)
    else:
      continue
  return valid_unpIdx_list

def vis_contactMap(working_dir,flNm):
  with open('{}/pdbmap_sets/pdbmap_contactData/{}.json'.format(working_dir,flNm),'r') as fl:
    contact_data = json.load(fl)
  for exp_dict in contact_data[:5]:
    pfamAcc = exp_dict["pfamAcc"]
    unpAcc = exp_dict["unpAcc"]
    ran = exp_dict["range"]
    best_pdb = exp_dict["best_pdb"]
    contMap = np.array(exp_dict["contact-map"])
    fig = plt.figure()
    plt.matshow(contMap)
    plt.savefig('{}/pdbmap_sets/pdbmap_contactFigs/{}_{}_{}_{}.png'.format(working_dir,pfamAcc,unpAcc,ran,best_pdb))

def prune_seqNeibor_wrap(working_dir, zeroOut_list):
  allData_jsonList = []
  # load data file list
  fl_list = np.loadtxt('{}/pdbmap_sets/pdbmap_contactData/data_list'.format(working_dir),dtype='str')
  # load json data
  for one_fl in fl_list:
    print('>>{}'.format(one_fl))
    with open('{}/pdbmap_sets/pdbmap_contactData/{}'.format(working_dir,one_fl),'r') as dt_fl:
      dt_json = json.load(dt_fl)
    for one_dt in dt_json:
      ori_cm = np.array(one_dt["contact-map"])
      diagIdx = kth_diag_indices(ori_cm,zeroOut_list)
      ori_cm[diagIdx] = 0
      one_dt["contact-map"] = ori_cm
      allData_jsonList.append(one_dt)
  # save json list
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_zero_01.json'.format(working_dir),'w') as fl:
    json.dump(allData_jsonList,fl,cls=NumpyArrayEncoder)

def filterByLen(working_dir, lowLenCut, highLenCut):
  allData_jsonList = []
  minLen, maxLen = 100,0
  # load data file list
  fl_list = np.loadtxt('{}/pdbmap_sets/pdbmap_contactData/data_list'.format(working_dir),dtype='str')
  # load json data
  for one_fl in fl_list:
    print('>>{}'.format(one_fl))
    with open('{}/pdbmap_sets/pdbmap_contactData/{}'.format(working_dir,one_fl),'r') as dt_fl:
      dt_json = json.load(dt_fl)
    for one_dt in dt_json:
      seqLen = int(one_dt["targetSeq_len"])
      seq = one_dt['target_seq']
      if 'X' in seq or 'x' in seq:
        continue
      if seqLen < minLen:
        minLen = seqLen
      if seqLen > maxLen:
        maxLen = seqLen
      if seqLen >=  lowLenCut and seqLen <= highLenCut:
        allData_jsonList.append(one_dt)
  print('total number: {}'.format(len(allData_jsonList)))
  print('minLen: {}, maxLen: {}'.format(minLen,maxLen))
  # save json list
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l{}h{}.json'.format(working_dir,lowLenCut,highLenCut),'w') as fl:
    json.dump(allData_jsonList,fl,cls=NumpyArrayEncoder)
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l{}h{}_sample.json'.format(working_dir,lowLenCut,highLenCut),'w') as fl:
    json.dump(allData_jsonList[:30],fl,cls=NumpyArrayEncoder)

def split_data(working_dir):
  rng = random.Random(25)
  # load test pfamAcc
  hold_pfam_list = np.loadtxt('{}/holdOut_sets/muta_pfam_small_set.txt'.format(working_dir), dtype='str')
  hold_clan_list = np.loadtxt('{}/holdOut_sets/muta_clan_small_set.txt'.format(working_dir), dtype='str')

  # build pfam-clan pair
  pfam_clan_dict = {}
  with open('{}/Pfam-A.clans.tsv'.format(working_dir), 'r') as fl:
    for line in fl:
      line_split = re.split(r'\t', line)
      fam = line_split[0]
      clan = line_split[1]
      pfam_clan_dict[fam] = clan
  # save this pfam_clan dict
  with open('{}/Pfam-A.clans.json'.format(working_dir), 'w') as fl:
    json.dump(pfam_clan_dict, fl)
  
  # init var
  train_list = []
  val_list = []
  holdOut_list = []
  num_count = [0, 0, 0]

  # load josn data
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500.json'.format(working_dir),'r') as fl:
    allData = json.load(fl)

  # loop data list
  for dt_dict in allData:
    famId = dt_dict['pfamAcc']
    clanId = pfam_clan_dict[famId]
    if famId in hold_pfam_list or clanId in hold_clan_list:
      dt_dict['id'] = num_count[2]
      holdOut_list.append(dt_dict)
      num_count[2] += 1
    else:
      rand_num =  rng.random()
      if rand_num < 0.1:
        dt_dict['id'] = num_count[1]
        val_list.append(dt_dict)
        num_count[1] += 1
      else:
        dt_dict['id'] = num_count[0]
        train_list.append(dt_dict)
        num_count[0] += 1
  # save to json
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_train.json'.format(working_dir),'w') as fl:
    json.dump(train_list, fl)
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_valid.json'.format(working_dir),'w') as fl:
    json.dump(val_list, fl)
  with open('{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_holdout.json'.format(working_dir),'w') as fl:
    json.dump(holdOut_list, fl)
  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_json.stat".format(sum(num_count),num_count[0],num_count[1],num_count[2],working_dir))

def add_valid_mask(working_dir: str = None,
                   file_list: List = None,
                   input_format: str = 'lmdb',
                   save_json: bool = True,
                   save_lmdb: bool = True,
                   save_sample: bool = True):
  '''
  build 'valid_mask' for each sequence
  output file format: lmdb
  '''
  # load json file
  #train_fl = '{}/allData_lenCut_l8h500_train'.format(working_dir)
  #valid_fl = '{}/allData_lenCut_l8h500_valid'.format(working_dir)
  #holdout_fl = '{}/allData_lenCut_l8h500_holdout'.format(working_dir)

  for fl_i in file_list:
    print('>_process {}'.format(fl_i))
    fl_nm = '{}/{}'.format(working_dir,fl_i)
    
    ## load data
    if input_format == 'json':
      with open('{}.{}'.format(fl_nm,input_format), 'r') as fl:
        dt_list = json.load(fl)
    elif input_format == 'lmdb':
      dt_list = []
      input_env = lmdb.open('{}.{}'.format(fl_nm,input_format), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          dt_list.append(item)
    
    ## build valid_mask and save data 
    output_env = lmdb.open('{}_new.lmdb'.format(fl_nm), map_size=(1024 * 50) * (2 ** 20))
    outlier_list = []
    idx_n = 0 
    with output_env.begin(write=True) as txn:
      for dt in dt_list:
        if idx_n % 100 == 0:
          print('>_{}'.format(idx_n))
        ran = dt['range']
        unp_pfam_range = dt['unp_pfam_range']
        valid_unpIdxs = dt['valid_unpIdxs']
        targetSeq_len = dt['targetSeq_len']
        # 'range' 'unp_pfam_range' not same
        if ran != unp_pfam_range:
          outlier_list.append(dt)
        
        unp_pfam_range_split = re.split('-',unp_pfam_range)
        unp_pfam_range_start, unp_pfam_range_end = int(unp_pfam_range_split[0]), int(unp_pfam_range_split[1])
        assert int(targetSeq_len) == unp_pfam_range_end - unp_pfam_range_start + 1
        valid_mask = [False]*targetSeq_len
        valid_pos = [val_i - unp_pfam_range_start for val_i in valid_unpIdxs]
        for i in valid_pos:
          valid_mask[i] = True
        dt['valid_mask'] = valid_mask
        txn.put(str(idx_n).encode(), pkl.dumps(dt))
        idx_n += 1
      txn.put(b'num_examples', pkl.dumps(idx_n))
    
    if save_json:
      with open('{}_new.json'.format(fl_nm),'w') as fl:
        json.dump(dt_list, fl)
    
    if save_sample:
      with open('{}_sample_new.json'.format(fl_nm),'w') as fl:
        json.dump(dt_list[:10], fl)

    # save outliers
    if len(outlier_list) > 0:
      with open('{}_outlier_addValidMask.json'.format(fl_nm),'w') as fl:
        json.dump(outlier_list, fl)

def contact_precision_fig(fig_name):
  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  n_head, n_layer = 8, 4 #bert-4
  log_dir = '{}/job_logs'.format(working_dir)
  metric_list = ['max_precision_all_5','max_precision_all_2','max_precision_all_1','max_precision_short_5','max_precision_short_2','max_precision_short_1','max_precision_medium_5','max_precision_medium_2','max_precision_medium_1','max_precision_long_5','max_precision_long_2','max_precision_long_1','mean_precision_all_5','mean_precision_all_2','mean_precision_all_1','mean_precision_short_5','mean_precision_short_2','mean_precision_short_1','mean_precision_medium_5','mean_precision_medium_2','mean_precision_medium_1','mean_precision_long_5','mean_precision_long_2','mean_precision_long_1','apc_precision_all_5','apc_precision_all_2','apc_precision_all_1','apc_precision_short_5','apc_precision_short_2','apc_precision_short_1','apc_precision_medium_5','apc_precision_medium_2','apc_precision_medium_1','apc_precision_long_5','apc_precision_long_2','apc_precision_long_1']
  
  metric_all_distri_list = ['all_pred_distribution_apc_1_corr_S-M-L','all_pred_distribution_apc_1_corr','all_pred_distribution_apc_1_total',
      'all_pred_distribution_apc_2_corr_S-M-L','all_pred_distribution_apc_2_corr','all_pred_distribution_apc_2_total',
      'all_pred_distribution_apc_5_corr_S-M-L','all_pred_distribution_apc_5_corr', 'all_pred_distribution_apc_5_total',
      'all_pred_distribution_max_1_corr_S-M-L','all_pred_distribution_max_1_corr', 'all_pred_distribution_max_1_total',
      'all_pred_distribution_max_2_corr_S-M-L','all_pred_distribution_max_2_corr', 'all_pred_distribution_max_2_total',
      'all_pred_distribution_max_5_corr_S-M-L','all_pred_distribution_max_5_corr', 'all_pred_distribution_max_5_total',
      'all_pred_distribution_mean_1_corr_S-M-L','all_pred_distribution_mean_1_corr','all_pred_distribution_mean_1_total',
      'all_pred_distribution_mean_2_corr_S-M-L','all_pred_distribution_mean_2_corr','all_pred_distribution_mean_2_total',
      'all_pred_distribution_mean_5_corr_S-M-L','all_pred_distribution_mean_5_corr', 'all_pred_distribution_mean_5_total']

  metric_esm_logis_list = ['logisContact_esm_all_5','logisContact_esm_all_2','logisContact_esm_all_1','logisContact_esm_short_5','logisContact_esm_short_2','logisContact_esm_short_1','logisContact_esm_medium_5','logisContact_esm_medium_2','logisContact_esm_medium_1','logisContact_esm_long_5','logisContact_esm_long_2','logisContact_esm_long_1']
  metric_logis_list = ['lgr_test_prec_all', 'lgr_test_prec_short', 'lgr_test_prec_medium', 'lgr_test_prec_long']
  metric_logis_layers_list = ['lgr_test_prec_all_layerwise','lgr_test_prec_short_layerwise','lgr_test_prec_medium_layerwise','lgr_test_prec_long_layerwise'] # [4,3]([n_layer,topk])
  metric_other_list = ['contact_background_prec_all','contact_background_prec_short','contact_background_prec_medium','contact_background_prec_long']

  sele_flag = {
      5: np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      6: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      7: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
      2: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      }


  df_list = []
  
   
  # myModel: load overall logistic precision
  rp_set = 'rp15_all'
  data_name = 'logistic_datasets'
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  init_epoch = '330'
  blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  parIdx_list = [5,6,7,2]
  chlIdx_list = [1,0,4,5,6,8]
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          #log_fl = 'biSuper_bert_rp15_all_{}_torch_eval_{}_{}.{}-{}.0.out'.format(init_epoch,mode_pair,par_idx,set_nm,set_type)
          print('>log file:{}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system('rm tmp_rec')
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce_blc_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_logis_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[3]
            mode = mode_pair
            #topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm
            
            metric_value = np.array(metric_json[metric_nm])
            metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
            metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])

            for val_i in range(len(metric_value)):
              topK = 'L/{}'.format(topk_list[val_i])
              precision = metric_value[val_i]
              precision_indiv_mean = metric_value_indiv_mean[val_i]
              precision_indiv_std = metric_value_indiv_std[val_i]
              df_list.append([precision,precision_indiv_mean,precision_indiv_std,mode,range_nm,topK,headSele_idx,para_idx,dt_set])
  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','mode','range','topK','headSele_idx','para_idx','dt_set'])
   
  '''  
  # myModel: load layerwise logistic precision
  data_name = 'logistic_datasets'
  mode_pair = 'con' # con, nonCon
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  init_epoch = '10'
  parIdx_list = [5,6,7,2]
  chlIdx_list = [1,0,4,5,6,8]
  topk_list = [1,2,5]
  layer_list = [1,2,3,4]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_rp15_all_{}_torch_eval_{}_{}.{}-{}.0.out'.format(init_epoch,mode_pair,par_idx,set_nm,set_type)
          print('>log file:{}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f11 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system('rm tmp_rec')
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          with open('{}/results_to_keep/rp15/{}/results_metrics_{}_{}.json'.format(working_dir,tar_dir,data_name,set_nm),'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_logis_layers_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[3]
            mode = mode_pair
            #topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm
            
            metric_value = np.array(metric_json[metric_nm]) # [n_layer,top_k] ([4,3])
            for ly_i in range(metric_value.shape[0]):
              for val_i in range(metric_value.shape[1]):
                layer_idx = layer_list[ly_i]
                topK = 'L/{}'.format(topk_list[val_i])
                precision = metric_value[ly_i][val_i]
                df_list.append([precision,mode,range_nm,topK,headSele_idx,para_idx,dt_set,layer_idx])
  df = pd.DataFrame(df_list,columns=['precision','mode','range','topK','headSele_idx','para_idx','dt_set','layer_idx'])
  '''
  '''
  # load correct prediction distribution in 'ALL' range
  data_name = 'pdbmap_contactData' # pdbmap_contactData, logistic_datasets
  set_type = 'contact' # contact, logistic
  init_epoch = '20'
  chlIdx_list = [1,0,4,5,6,8]
  parIdx_list = [5,6,7,2]
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        chl_idx = chlIdx_list[chl_i]
        ppar_idx = parIdx_list[ppar_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_rp15_all_{}_torch_eval_{}_{}.{}-{}.0.out'.format(init_epoch,mode_pair,par_idx,set_nm,set_type)
          print('>log file: {}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f11 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system("rm tmp_rec")
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          with open('{}/results_to_keep/rp15/{}/results_metrics_{}_{}.json'.format(working_dir,tar_dir,data_name,set_nm),'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1-7; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_all_distri_list:
            name_split = re.split('_',metric_nm)
            if name_split[-1] == 'S-M-L':
              symm_way = name_split[3]
              mode = mode_pair
              topK = 'L/{}'.format(name_split[4])
              headSele_idx = ppar_i + 1 # convert ot 1,2,3...
              para_idx = chl_i + 1 # convert to 1,2,3...
              dt_set = set_nm

              metric_value = np.array(metric_json[metric_nm]) # [n_layer,n_head,3(S,M,L)]
              for lay in range(metric_value.shape[0]):
                for hea in range(metric_value.shape[1]):
                  head_idx = hea + 1
                  layer_idx = lay + 1
                  reg_flag = sele_flag[ppar_idx][lay][hea]
                  corr_num_s = metric_value[lay][hea][0]
                  corr_num_m = metric_value[lay][hea][1]
                  corr_num_l = metric_value[lay][hea][2]
                  corr = corr_num_s + corr_num_m + corr_num_l
                  df_list.append([corr_num_s/corr,mode,'short',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])
                  df_list.append([corr_num_m/corr,mode,'medium',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])
                  df_list.append([corr_num_l/corr,mode,'long',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])

  df = pd.DataFrame(df_list,columns=['corr_num','mode','range','topK','headSele_idx','para_idx','dt_set','head_idx','layer_idx','symm','reg_flag'])
  '''
  ''' 
  # load head-wise precision
  data_name = 'pdbmap_contactData' # pdbmap_contactData, logistic_datasets
  mode_pair = 'con' # con, nonCon
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'contact' # contact, logistic
  init_epoch = '83'
  chlIdx_list = [1,0,4,5,6,8]
  parIdx_list = [5,6,7,2]
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        chl_idx = chlIdx_list[chl_i]
        ppar_idx = parIdx_list[ppar_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_rp15_all_{}_torch_eval_{}_{}.{}-{}.0.out'.format(init_epoch,mode_pair,par_idx,set_nm,set_type)
          print('>log file: {}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f11 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system("rm tmp_rec")
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          with open('{}/results_to_keep/rp15/{}/results_metrics_{}_{}.json'.format(working_dir,tar_dir,data_name,set_nm),'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1-7; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[2]
            symm_way = name_split[0]
            mode = mode_pair
            topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1 # convert ot 1,2,3...
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm

            metric_value = np.array(metric_json[metric_nm])
            for lay in range(metric_value.shape[0]):
              for hea in range(metric_value.shape[1]):
                head_idx = hea + 1
                layer_idx = lay + 1
                reg_flag = sele_flag[ppar_idx][lay][hea]
                precision = metric_value[lay][hea]
                df_list.append([precision,mode,range_nm,topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])
  df = pd.DataFrame(df_list,columns=['precision','mode','range','topK','headSele_idx','para_idx','dt_set','head_idx','layer_idx','symm','reg_flag'])
  '''
  ''' 
  # load esm data
  for set_nm in ['train','valid','holdout']:
    with open('{}/data_process/pfam_32.0/pdbmap_sets/pdbmap_contactData/esm_models/esm1_t6_43M_UR50S_results_metrics_pdbmap_contactData_{}.json'.format(working_dir,set_nm),'r') as f:
      metric_json = json.load(f)
      # convert json to dataFrame format
      # symm: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
      # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
      # dt_set: train,valid,holdout
      for metric_nm in metric_list:
        name_split = re.split('_',metric_nm)
        range_nm = name_split[2]
        symm = name_split[0]
        topK = 'L/{}'.format(name_split[3])
        dt_set = set_nm

        metric_value = np.array(metric_json[metric_nm])
        for lay in range(metric_value.shape[0]):
          for hea in range(metric_value.shape[1]):
            layer_idx, head_idx = lay+1, hea+1
            prec = metric_value[lay,hea]
            df_list.append([prec,layer_idx,head_idx,symm,range_nm,topK,dt_set])
  df = pd.DataFrame(df_list,columns=['precision','layer_idx','head_idx','symm','range','topK','dt_set'])
  '''
  '''
  # load best logistic model weights
  init_epoch = '83'
  parIdx_list = [5,6,7,2]
  chlIdx_list = [1,0,4,5,6,8]
  topk_list = [1,2,5]
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  l1_weight_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  for mode_pair in ['con','nonCon']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        par_idx = '{}_{}'.format(ppar_idx,chl_idx)
        print('loading set: {}_{}'.format(mode_pair,par_idx))
        log_fl = 'biSuper_bert_rp15_all_{}_torch_eval_{}_{}.{}-{}.0.out'.format(init_epoch,mode_pair,par_idx,set_nm,set_type)
        print('>log file:{}'.format(log_fl))
        os.system("grep 'loading weights file' {}/{} | cut -d'/' -f11 > tmp_rec".format(log_dir,log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system('rm tmp_rec')
        print('>model dir:',tar_dir)
        for ran_str in ['all','short','medium','long']:
          gridSearch = np.loadtxt('{}/data_process/pfam_32.0/pdbmap_sets/pdbmap_contactData/logistic_datasets/logistic_models/{}/gridSear_model_{}.csv'.format(working_dir,tar_dir,ran_str),dtype='float',delimiter=',')
          valid_prec_best_idx = np.unravel_index(gridSearch.argmax(), gridSearch.shape)
          best_num = valid_prec_best_idx[0]+1
          best_l1w = l1_weight_list[valid_prec_best_idx[1]]
          print('> {}, num: {}, l1w: {}'.format(ran_str,best_num,best_l1w))
          best_mdl = joblib.load('{}/data_process/pfam_32.0/pdbmap_sets/pdbmap_contactData/logistic_datasets/logistic_models/{}/model_{}_{}'.format(working_dir,tar_dir,best_num,best_l1w))
          #print('> best coef: ',best_mdl.coef_)
          best_coef = best_mdl.coef_.reshape(n_layer,n_head)
          best_intercept = best_mdl.intercept_[0]
          for coef_layer in range(best_coef.shape[0]):
            for coef_head in range(best_coef.shape[1]):
              df_list.append([best_intercept,best_coef[coef_layer][coef_head],coef_layer+1,coef_head+1,ran_str,mode_pair,ppar_i+1,chl_i+1,set_nm])
  df = pd.DataFrame(df_list,columns=['intercept','coef','layer_idx','head_idx','range','mode','headSele_idx','para_idx','dt_set'])
  '''
  '''
  # load pretrain model head-wise precision
  data_name = 'pdbmap_contactData' # pdbmap_contactData, logistic_datasets
  set_type = 'contact' # contact, logistic
  epoch_dir = {'rp15':330,'rp35':130,'rp55':'90','rp75':60}
  topk_list = [1,2,5]
  for rp_set in ['rp15','rp35','rp55','rp75']:
    for set_nm in ['holdout']:
      print('loading set: {}'.format(rp_set))
      log_fl = 'baseline_bert_{}_torch_eval.{}_{}.0.{}.out'.format(rp_set,set_nm,set_type,epoch_dir[rp_set])
      print('>log file: {}'.format(log_fl))
      os.system("grep 'loading weights file' {}/{} | cut -d'/' -f11 > tmp_rec".format(log_dir,log_fl))
      with open('tmp_rec', 'r') as f:
        tar_dir = f.read()[:-1]
      os.system("rm tmp_rec")
      print('>model dir:',tar_dir)
      print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
      with open('{}/results_to_keep/{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,tar_dir,data_name,set_nm),'r') as f:
        metric_json = json.load(f)
      # convert json to dataFrame format
      # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
      # headSele_idx: 1-7; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
      # dt_set: train,valid,holdout
      dt_list = []
      for metric_nm in metric_list:
        name_split = re.split('_',metric_nm)
        range_nm = name_split[2]
        symm_way = name_split[0]
        mode = mode_pair
        topK = 'L/{}'.format(name_split[3])
        dt_set = set_nm

        metric_value = np.array(metric_json[metric_nm])
        for lay in range(metric_value.shape[0]):
          for hea in range(metric_value.shape[1]):
            head_idx = hea + 1
            layer_idx = lay + 1
            precision = metric_value[lay][hea]
            df_list.append([precision,rp_set,range_nm,topK,dt_set,head_idx,layer_idx,symm_way])
  df = pd.DataFrame(df_list,columns=['precision','rp_set','range','topK','dt_set','head_idx','layer_idx','symm'])
  '''
  # load background precision
  prec_bg_dict = {
      'all': 0.0186,
      'short': 0.0528,
      'medium': 0.0363,
      'long': 0.0145 
      }
    
  # draw figs
  if fig_name == 'colormap':
    max_val = 0
    for metric_nm in metric_list:
      mmax = np.amax(metric_json[metric_nm])
      if max_val < mmax:
        max_val = mmax
    for metric_nm in metric_list:
      metric_value = np.squeeze(np.array(metric_json[metric_nm]))
      layer_ave = np.sum(metric_value, axis=-1) / n_head
      #max_val = np.amax(metric_value)
      min_val = 0
      gs = gridspec.GridSpec(2,1, height_ratios=[1,2])
      fig = plt.figure(figsize=(13,13))
      ax1 = plt.subplot(gs[0])
      ##cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
      ax1_mat = ax1.matshow(metric_value, cmap=plt.cm.hot_r, vmin=min_val, vmax=max_val)
      ax1.set_xticks(np.arange(0,n_head))
      ax1.set_yticks(np.arange(0,n_layer))
      ax1.set_yticklabels(np.arange(1,n_layer+1))
      ax1.set_xticklabels(np.arange(1,n_head+1))
      for (i, j), z in np.ndenumerate(metric_value):
        ax1.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

      ax2 = plt.subplot(gs[1])
      ax2.barh(np.arange(n_layer),layer_ave,align='center')
      ax2.invert_yaxis()
      #ax2_yStart, ax2_yEnd = ax2.get_ylim()
      #ax2.set_yticks(np.arange(ax2_yStart+0.5,ax2_yEnd,1))
      #ax2.set_yticklabels(np.arange(1,n_layer+1))
      ax2.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.50)
      
      cb = fig.colorbar(ax1_mat, ax=[ax1,ax2], location='left')

      if not os.path.isdir('{}/results_to_keep/figures/{}_{}_{}_{}'.format(working_dir,mode_pair,par_idx,set_nm,set_type)):
        os.mkdir('{}/results_to_keep/figures/{}_{}_{}_{}'.format(working_dir,mode_pair,par_idx,set_nm,set_type))
      plt.savefig('{}/results_to_keep/figures/{}_{}_{}_{}/{}.png'.format(working_dir,mode_pair,par_idx,set_nm,set_type,metric_nm))
      plt.close(fig)
  
  elif fig_name == 'lgs_heatmap':
    # prepare weight matrix
    hs_idx, pra_idx = 4, 2
    vmax=65
    filter_df = df.loc[(df["headSele_idx"]==hs_idx) & (df["para_idx"]==pra_idx) & (df["mode"]=='con')]

    def facet_heatmap(data,color,**kwargs):
      dt_mat = data.pivot(index='head_idx', columns='layer_idx', values='coef')
      sns.heatmap(dt_mat,cmap='vlag',**kwargs)
    
    #sns.set(font_scale=5.5)
    gax = sns.FacetGrid(filter_df, col="range", height=2, aspect=0.6)
    cbar_ax = gax.fig.add_axes([.92, .3, .02, .4])  # <-- Create a colorbar axes
    gax = gax.map_dataframe(facet_heatmap, cbar_ax=cbar_ax, vmin=-vmax, vmax=vmax)
    gax.set_titles(col_template="{col_name}", fontweight='bold')
    gax.fig.subplots_adjust(right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
    tar_fig_dir = 'logistic'
    if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
      os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
    gax.savefig('{}/results_to_keep/figures/{}/logis_weight_{}_{}.png'.format(working_dir,tar_fig_dir,hs_idx,pra_idx))
    plt.clf()


  elif fig_name == 'catplot':
    #sns.set_theme(style="whitegrid")
    sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
    range_togo_list = ['all','short','medium','long']
    topK_togo_list = ['1','2','5']
    symm_togo_list = ['apc','max','mean']
    for range_togo in ['all']:
      for topK_togo in topK_togo_list:
        for symm_togo in ['apc']:
          #filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["symm"]==symm_togo) & (df["reg_flag"]==0)] # all TP distribition
          #filter_df = df.loc[(df["dt_set"]=='holdout') & (df["symm"]==symm_togo)] # pretrained model 
          filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo)]
          #filter_df = df.loc[(df["headSele_idx"]==1) & (df["para_idx"]==3) & (df["mode"]=='con') & (df["dt_set"]=='holdout')]
          #filter_df = df.loc[(df["topK"]=='L/'+topK_togo) & (df["symm"]=='apc') & (df["range"]==range_togo) & (df["dt_set"]=='holdout')]
          
          # pretrained model - 1
          #gax = sns.catplot(x="range", y="precision", hue="layer_idx",
          #                 row="rp_set", col='topK', data=filter_df, kind='box',
          #                 height=4, aspect=1.5,palette=sns.color_palette("hls", 4),
          #                 order=['all','short','medium','long'],hue_order=[1,2,3,4],row_order=['rp15','rp35','rp55','rp75'],col_order=['L/1','L/2','L/5'],
          #                 whis=10.0,width=0.5)
          
          # pretrained model - 2
          #gax = sns.catplot(x="layer_idx", y="precision", hue="rp_set",
          #                 row="range", col='topK', data=filter_df, kind='box',
          #                 height=4, aspect=1.5,palette=sns.color_palette("hls", 4),
          #                 order=[1,2,3,4],hue_order=['rp15','rp35','rp55','rp75'],row_order=['all','short','medium','long'],col_order=['L/1','L/2','L/5'],
          #                 whis=10.0,width=0.5)

          # head-wise 
          #gax = sns.catplot(x="para_idx", y="precision", hue="layer_idx",
          #                 row="mode", col='headSele_idx', data=filter_df, kind='box',
          #                 height=4, aspect=1.5,palette=sns.color_palette(),
          #                 order=[1,2,3,4,5,6],hue_order=[1,2,3,4],row_order=['con','nonCon','ce'],col_order=[1,2,3,4],
          #                 whis=10.0,width=0.5)
          #gax.map(sns.pointplot, "para_idx", "precision", "layer_idx", ci=None, dodge=0.4, linestyles='-.',
          #        estimator=np.median, order=[1,2,3,4,5,6],hue_order=[1,2,3,4],palette=sns.color_palette())
         
          # all TP distribution 
          #gax = sns.catplot(x="para_idx", y="corr_num", hue="range",
          #                 row="mode", col='headSele_idx', data=filter_df, kind='box',
          #                 height=4, aspect=1.5,palette=sns.color_palette(),
          #                 order=[1,2,3,4,5,6],hue_order=['short','medium','long'],row_order=['con','nonCon','ce'],col_order=[1,2,3,4],
          #                whis=10.0,width=0.5)

          # logistic layer-wise
          #gax = sns.catplot(x="para_idx", y="precision", hue="layer_idx",
          #                 row="mode", col='headSele_idx', data=filter_df, kind='point',
          #                 height=4,aspect=1.5,palette=sns.color_palette(),ci=None,dodge=0.4,linestyles='--',
          #                 order=[1,2,3,4,5,6],hue_order=[1,2,3,4],row_order=['con','nonCon','ce'],col_order=[1,2,3,4])

          # logistic,own attention
          gax = sns.catplot(x="para_idx", y="precision", hue="range",
                           row="mode", col='headSele_idx', data=filter_df, kind='point',
                           height=4, aspect=1.5, ci=None, dodge=0.4,
                           order=[1,2,3,4,5,6],hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=[1,2,3,4])
          
          #gax.set(ylim=(0.01, None))
          #gax.map(sns.pointplot, "para_idx", "precision", "range", ci=None, dodge=0.6, linestyles='--',
          #        palette=sns.color_palette(), estimator=np.min, order=[1,2,3,4,5,6],hue_order=['all','short','medium','long'])
          '''
          # head-wise esm
          ax = sns.catplot(x="range", y="precision",hue="layer_idx",
                           col="dt_set", data=filter_df, kind='box',
                           height=8, aspect=.9, 
                           col_order=['train','valid','holdout'],hue_order=[1,2,3,4,5,6],order=['all','short','medium','long'],legend=False)
          '''
          '''
          ax = sns.catplot(x="topK", y="precision", hue="symm",
                           col="range", data=filter_df, kind='box',
                           height=8, aspect=.8, 
                           order=['L/1','L/2','L/5'],hue_order=['apc','mean','max'],col_order=['all','short','medium','long'])
          '''
          #sns.stripplot(x="topK", y="precision", hue="symm",data=df[df['range']=='all'],color='.3',ax=ax.axes[0][0])
          #sns.stripplot(x="topK", y="precision", hue="symm",data=df[df['range']=='short'], color='.3',ax=ax.axes[0][1])
          #sns.stripplot(x="topK", y="precision", hue="symm",data=df[df['range']=='medium'], color='.3',ax=ax.axes[0][2])
          #sns.stripplot(x="topK", y="precision", hue="symm",data=df[df['range']=='long'], color='.3',ax=ax.axes[0][3])

         
          #gax.map(plt.axhline,y=prec_bg_dict[range_togo],color='gray',ls='-.',lw=1.,label='bg')

          gax.map(plt.axhline,y=0.0186,color='gray',ls='solid',lw=1.,label='all_bg')
          gax.map(plt.axhline,y=0.0528,color='gray',ls='dotted',lw=1.,label='short_bg')
          gax.map(plt.axhline,y=0.0363,color='gray',ls='dashed',lw=1.,label='medium_bg')
          gax.map(plt.axhline,y=0.0145,color='gray',ls='dashdot',lw=1.,label='long_bg')

          #ax.axes[0][0].axhline(0.0186,color='red',ls='--')
          #ax.axes[0][1].axhline(0.0528,color='red',ls='--')
          #ax.axes[0][2].axhline(0.0364,color='red',ls='--')
          #ax.axes[0][3].axhline(0.0145,color='red',ls='--')
         
          #ax.axes[0][0].axhline(0.0364,color='red',ls='--')
          #ax.axes[0][1].axhline(0.0364,color='red',ls='--')
          #ax.axes[1][0].axhline(0.0364,color='red',ls='--')
          #ax.axes[1][1].axhline(0.0364,color='red',ls='--')
          
          #gax.axes[2][2].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
          
          def to_percent(y, position):
            s = '{:0.2f}'.format(100*y)
            return s + '%'
          formatter = FuncFormatter(to_percent)
          plt.gca().yaxis.set_major_formatter(formatter)
          
          tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
          if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
            os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
          #gax.savefig('{}/results_to_keep/figures/{}/{}_logis_layer_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
          gax.savefig('{}/results_to_keep/figures/{}/overall_logis_{}.png'.format(working_dir,tar_fig_dir,topK_togo))
          #gax.savefig('{}/results_to_keep/figures/{}/all_distri_nonRegLayer_{}_{}.png'.format(working_dir,tar_fig_dir,symm_togo,topK_togo))
          #gax.savefig('{}/results_to_keep/figures/{}/{}_layer_apc_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
          #gax.savefig('{}/results_to_keep/figures/{}/contact_precision_rpwise_{}.png'.format(working_dir,tar_fig_dir,symm_togo))
          plt.clf()
  
  elif fig_name == 'boxplot-dt_set':
    sns.set_theme(style="whitegrid")
    for h_idx in [1,2,3,4]:
      for p_idx in [5,6,7,8]:
        par_idx = '{}_{}'.format(h_idx,p_idx)
        filter_df = df.loc[(df["headSele_idx"]==h_idx) & (df["para_idx"]==p_idx) & (df["mode"]=='apc')]
        ax = sns.catplot(x="topK", y="precision", hue="dt_set", col="range",
                         data=filter_df, kind="box", height=4, aspect=.8,
                         order=["L/1","L/2","L/5"],hue_order=["train","valid","holdout"],col_order=["all","short","medium","long"])
        if not os.path.isdir('{}/results_to_keep/figures/{}_{}_{}'.format(working_dir,mode_pair,par_idx,set_type)):
          os.mkdir('{}/results_to_keep/figures/{}_{}_{}'.format(working_dir,mode_pair,par_idx,set_type))
        ax.savefig('{}/results_to_keep/figures/{}_{}_{}/train_val_test_apc_boxplot.png'.format(working_dir,mode_pair,par_idx,set_type))
        plt.clf()

def to_percent(y, position):
  s = '{:0.2f}'.format(100*y)
  return s + '%'

def errbar(x,y,yerr,**kwargs):
  ax=plt.gca()
  ax.errorbar(x, y, yerr=yerr, fmt='', zorder=-1)

def contact_precision_fig_overall_logistic_pretrain(rp_set: str = 'rp15_all',
                                                    rp_mdl_idx: str = '1',
                                                    tar_fig_dir: str = None,
                                                    prec_bg_dict: dict = None,
                                                    logIdx_dict: dict = None,
                                                    epoch_dict: dict = None):
  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_baseline_bert_eval'.format(working_dir)

  metric_logis_list = ['lgr_test_prec_all', 'lgr_test_prec_short', 'lgr_test_prec_medium', 'lgr_test_prec_long']

  
  ##list for dataframe
  df_list = []

  # myModel: load overall logistic precision
  data_name = 'logistic_datasets'
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  topk_list = [1,2,5]
  for log_sub_i in logIdx_dict[rp_mdl_idx]:
    for epoch_i in epoch_dict['{}_{}'.format(rp_mdl_idx,log_sub_i)]:
      for set_nm in ['holdout']:
        log_fl = 'baseline_bert_{}_{}_torch_eval.holdout_logistic.0.{}.out'.format(rp_set,rp_mdl_idx,epoch_i+1)
        print('>log file:{}'.format(log_fl))
        os.system("grep 'loading weights file' {}/{} | cut -d'/' -f12 > tmp_rec".format(log_dir,log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system('rm tmp_rec')
        print('>model dir:',tar_dir)
        print('>json file: results_metrics_{}_{}_{}.json'.format(data_name,set_nm,epoch_i))
        json_fl='{}/results_to_keep/{}/{}_pretrain_{}_models/{}/results_metrics_{}_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,tar_dir,data_name,set_nm,epoch_i)
        with open(json_fl,'r') as f:
          metric_json = json.load(f)
        # convert json to dataFrame format
        # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
        # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
        # dt_set: train,valid,holdout
        for metric_nm in metric_logis_list:
          name_split = re.split('_',metric_nm)
          range_nm = name_split[3]
          #topK = 'L/{}'.format(name_split[3])
          dt_set = set_nm
          
          metric_value = np.array(metric_json[metric_nm])
          metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
          metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])

          for val_i in range(len(metric_value)):
            topK = 'L/{}'.format(topk_list[val_i])
            precision = metric_value[val_i]
            precision_indiv_mean = metric_value_indiv_mean[val_i]
            precision_indiv_std = metric_value_indiv_std[val_i]
            df_list.append([precision,precision_indiv_mean,precision_indiv_std,range_nm,topK,dt_set,epoch_i])
  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','range','topK','dt_set','epoch'])
  
  ## select model; long@topL
  ran_sel = 'long'
  filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/1') & (df["range"]==ran_sel)]
  row_sel = df.iloc[filter_df['precision_mean'].idxmax()]
  print(row_sel.to_string())
  
  ## draw figure
  #sns.set_theme(style="whitegrid")
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  #symm_togo_list = ['apc','max','mean']
  for topK_togo in topK_togo_list:
    #filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df['epoch'] <= 100)]
    filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo)]
    #print(filter_df.to_string())
    ## micro precision
    gax = sns.catplot(x="epoch", y="precision", hue="range",
                     data=filter_df, kind='point',
                     height=5, aspect=6.0, ci=None,
                     hue_order=['all','short','medium','long'])
    
    gax.map(plt.axhline,y=prec_bg_dict['all'],color='gray',ls='solid',lw=1.,label='all_bg')
    gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
    gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
    gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
    
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    #tar_fig_dir = 'rp15_all_blc@330' #logistics overall
    if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
      os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
    gax.savefig('{}/results_to_keep/figures/{}/overall_logis_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
    plt.close()
    
    ## precision - mean/std of each example
    gax = sns.catplot(x="epoch", y="precision_mean", hue="range",
                     data=filter_df, kind='point',
                     height=5, aspect=6.0, ci=None,
                     hue_order=['all','short','medium','long'])
    
    # draw error bar
    #gax.map(errbar, 'para_idx', 'precision_mean', 'precision_std')

    gax.map(plt.axhline,y=prec_bg_dict['all_mean'],color='gray',ls='solid',lw=1.,label='all_bg')
    gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
    gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
    gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')
    
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
    if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
      os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
    gax.savefig('{}/results_to_keep/figures/{}/overall_logis_meanstd_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
    plt.close()

def contact_precision_fig_overall_logistic(rp_set: str = 'rp15_all',
                                           rp_mdl_idx: str = '_1',
                                           tar_fig_dir: str = None,
                                           prec_bg_dict: dict = None,
                                           init_epoch: str = '100',
                                           blc_flag: str = '',
                                           stre_wgt: str = '1',
                                           parIdx_list: List = [5,6,7,2], 
                                           chlIdx_list: List = [1,0,4,5,6,8]
                                           ):
  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_biSuper_bert_eval'.format(working_dir)

  metric_logis_list = ['lgr_test_prec_all', 'lgr_test_prec_short', 'lgr_test_prec_medium', 'lgr_test_prec_long']

  
  ##list for dataframe
  df_list = []

  # myModel: load overall logistic precision
  data_name = 'logistic_datasets'
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  if len(blc_flag) > 0 :
    blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  else:
    blc_set={'con':'','nonCon':'','ce':''}
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          print('>log file:{}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system('rm tmp_rec')
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce{}_strethWeight{}_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_logis_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[3]
            mode = mode_pair
            #topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm
            
            metric_value = np.array(metric_json[metric_nm])
            metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
            metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])

            for val_i in range(len(metric_value)):
              topK = 'L/{}'.format(topk_list[val_i])
              precision = metric_value[val_i]
              precision_indiv_mean = metric_value_indiv_mean[val_i]
              precision_indiv_std = metric_value_indiv_std[val_i]
              df_list.append([precision,precision_indiv_mean,precision_indiv_std,mode,range_nm,topK,headSele_idx,para_idx,dt_set])
  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','mode','range','topK','headSele_idx','para_idx','dt_set'])
  
  ## select model; long@topL
  ran_sel = 'all'
  for m in ['con','nonCon','ce']:
    print('>>mode:',m)
    filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/1') & (df["range"]==ran_sel) & (df["mode"]==m)]
    row_sel = df.iloc[filter_df['precision_mean'].idxmax()]
    print(row_sel.to_string())

  
  ## draw figure
  #sns.set_theme(style="whitegrid")
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  symm_togo_list = ['apc','max','mean']
  for range_togo in ['all']:
    for topK_togo in topK_togo_list:
      for symm_togo in ['apc']:
        filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo)]
        ## micro precision
        gax = sns.catplot(x="para_idx", y="precision", hue="range",
                         row="mode", col='headSele_idx', data=filter_df, kind='point',
                         height=4, aspect=1.5, ci=None, dodge=0.4,
                         order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
        
        gax.map(plt.axhline,y=prec_bg_dict['all'],color='gray',ls='solid',lw=1.,label='all_bg')
        gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
        gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
        gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        
        #tar_fig_dir = 'rp15_all_blc@330' #logistics overall
        if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
          os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
        gax.savefig('{}/results_to_keep/figures/{}/overall_logis_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        plt.close()
        
        ## precision - mean/std of each example
        gax = sns.catplot(x="para_idx", y="precision_mean", hue="range",
                         row="mode", col='headSele_idx', data=filter_df, kind='point',
                         height=4, aspect=1.5, ci=None, dodge=0.4,
                         order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
        
        
        # draw error bar
        #gax.map(errbar, 'para_idx', 'precision_mean', 'precision_std')

        gax.map(plt.axhline,y=prec_bg_dict['all_mean'],color='gray',ls='solid',lw=1.,label='all_bg')
        gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
        gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
        gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        
        #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
        if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
          os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
        gax.savefig('{}/results_to_keep/figures/{}/overall_logis_meanstd_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        plt.close()
  
def contact_precision_fig_layersuper_logistic(rp_set: str='rp15_all',
                                              rp_mdl_idx: str='_1',
                                              init_epoch: str='330',
                                              tar_fig_dir: str=None,
                                              prec_bg_dict: dict=None,
                                              blc_flag: str = '',
                                              stre_wgt: str = '1',
                                              parIdx_list: List = [5,6,7,2],
                                              chlIdx_list: List = [1,0,4,5,6,8]
                                              ):

  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_biSuper_bert_eval'.format(working_dir)
  metric_logis_supervise_list = ['lgr_test_prec_all_layersupervise',
                              'lgr_test_prec_short_layersupervise',
                              'lgr_test_prec_medium_layersupervise',
                              'lgr_test_prec_long_layersupervise']
  
  df_list = []
  # myModel: load layerSupervised logistic precision
  data_name = 'logistic_datasets'
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  if len(blc_flag) > 0 :
    blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  else:
    blc_set={'con':'','nonCon':'','ce':''}
  topk_list = [1,2,5]
  layer_list = range(1,len(parIdx_list)+1)
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          print('>log file:{}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system('rm tmp_rec')
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce{}_strethWeight{}_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_logis_supervise_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[3]
            mode = mode_pair
            #topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1 # convert to 1,2,3...
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm
            
            metric_value = np.array(metric_json[metric_nm]) # [n_layer,top_k] ([4,3])
            metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
            metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])

            for val_i in range(len(metric_value)):
              topK = 'L/{}'.format(topk_list[val_i])
              precision = metric_value[val_i]
              precision_indiv_mean = metric_value_indiv_mean[val_i]
              precision_indiv_std = metric_value_indiv_std[val_i]
              df_list.append([precision,precision_indiv_mean,precision_indiv_std,mode,range_nm,topK,headSele_idx,para_idx,dt_set])
            
  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','mode','range','topK','headSele_idx','para_idx','dt_set'])

  # logistic layer-wise
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  symm_togo_list = ['apc','max','mean']
  for range_togo in ['all']:
    for topK_togo in topK_togo_list:
      for symm_togo in ['apc']:
        filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo)]
        ## micro precision
        gax = sns.catplot(x="para_idx", y="precision", hue="range",
                         row="mode", col='headSele_idx', data=filter_df, kind='point',
                         height=4, aspect=1.5, ci=None, dodge=0.4,
                         order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
        
        gax.map(plt.axhline,y=prec_bg_dict['all'],color='gray',ls='solid',lw=1.,label='all_bg')
        gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
        gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
        gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        
        #tar_fig_dir = 'rp15_all_blc@330' #logistics overall
        if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
          os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
        gax.savefig('{}/results_to_keep/figures/{}/supervise_logis_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        plt.close()
        
        ## precision - mean/std of each example
        gax = sns.catplot(x="para_idx", y="precision_mean", hue="range",
                         row="mode", col='headSele_idx', data=filter_df, kind='point',
                         height=4, aspect=1.5, ci=None, dodge=0.4,
                         order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
        
        
        # draw error bar
        #gax.map(errbar,'para_idx','precision_mean','precision_std')

        gax.map(plt.axhline,y=prec_bg_dict['all_mean'],color='gray',ls='solid',lw=1.,label='all_bg')
        gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
        gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
        gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        
        #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
        if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
          os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
        gax.savefig('{}/results_to_keep/figures/{}/supervise_logis_meanstd_topK{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        plt.close()
  
def contact_precision_fig_layerwise_logistic(rp_set: str='rp15_all',
                                             rp_mdl_idx: str='_1',
                                             init_epoch: str='330',
                                             tar_fig_dir: str=None,
                                             prec_bg_dict: dict=None,
                                             n_head: int=8,
                                             n_layer: int=4,
                                             blc_flag: str = '',
                                             stre_wgt: str = '1',
                                             parIdx_list: List = [5,6,7,2],
                                             chlIdx_list: List = [1,0,4,5,6,8]
                                             ):

  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_biSuper_bert_eval'.format(working_dir)

  metric_logis_layers_list = ['lgr_test_prec_all_layerwise','lgr_test_prec_short_layerwise','lgr_test_prec_medium_layerwise','lgr_test_prec_long_layerwise']
  #[3,]([topk,])

  df_list = []

  # myModel: load layerSupervised logistic precision
  data_name = 'logistic_datasets'
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'logistic' # contact, logistic
  if len(blc_flag) > 0 :
    blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  else:
    blc_set={'con':'','nonCon':'','ce':''}
  topk_list = [1,2,5]
  layer_list = range(1,len(parIdx_list)+1)
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        ppar_idx = parIdx_list[ppar_i]
        chl_idx = chlIdx_list[chl_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          print('>log file:{}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system('rm tmp_rec')
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce{}_strethWeight{}_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1,2,3,4; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_logis_layers_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[3]
            mode = mode_pair
            #topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1 # convert to 1,2,3...
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm
            
            metric_value = np.array(metric_json[metric_nm]) # [n_layer,top_k] ([4,3])
            metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
            metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])

            for ly_i in range(metric_value.shape[0]):
              for val_i in range(metric_value.shape[1]):
                layer_idx = layer_list[ly_i]
                topK = 'L/{}'.format(topk_list[val_i])
                precision = metric_value[ly_i][val_i]
                precision_indiv_mean = metric_value_indiv_mean[ly_i][val_i]
                precision_indiv_std = metric_value_indiv_std[ly_i][val_i]
                df_list.append([precision,precision_indiv_mean,precision_indiv_std,mode,range_nm,topK,headSele_idx,para_idx,dt_set,layer_idx])
  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','mode','range','topK','headSele_idx','para_idx','dt_set','layer_idx'])

  # logistic layer-wise
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  for range_togo in range_togo_list:
    for topK_togo in topK_togo_list:
      filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["range"]==range_togo)]
      gax = sns.catplot(x="para_idx", y="precision", hue="layer_idx",
                       row="mode", col='headSele_idx', data=filter_df, kind='point',
                       height=4,aspect=1.5,palette=sns.color_palette(),ci=None,dodge=0.4,linestyles='--',
                       order=range(1,len(chlIdx_list)+1),hue_order=range(1,n_layer+1),row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
      
      gax.map(plt.axhline,y=prec_bg_dict[range_togo],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/layerwise_logis_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()
      
      ## precision mean/std
      gax = sns.catplot(x="para_idx", y="precision_mean", hue="layer_idx",
                       row="mode", col='headSele_idx', data=filter_df, kind='point',
                       height=4,aspect=1.5,palette=sns.color_palette(),ci=None,dodge=0.4,linestyles='--',
                       order=range(1,len(chlIdx_list)+1),hue_order=range(1,n_layer+1),row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
      
      # draw error bar
      #gax.map(errbar,'para_idx','precision_mean','precision_std')

      gax.map(plt.axhline,y=prec_bg_dict['{}_mean'.format(range_togo)],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')

      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/layerwise_logis_meanstd_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()

  for layer_togo in range(1,n_layer+1):
    for topK_togo in topK_togo_list:
      filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["layer_idx"]==layer_togo)]
      gax = sns.catplot(x="para_idx", y="precision", hue="range",
                       row="mode", col='headSele_idx', data=filter_df, kind='point',
                       height=4,aspect=1.5,palette=sns.color_palette(),ci=None,dodge=0.4,linestyles='--',
                       order=range(1,len(chlIdx_list)+1),hue_order=range(1,n_layer+1),row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
      
      gax.map(plt.axhline,y=prec_bg_dict['all'],color='gray',ls='solid',lw=1.,label='all_bg')
      gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
      gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/layerwise_logis_layer{}_topK{}.png'.format(working_dir,tar_fig_dir,layer_togo,topK_togo))
      plt.close()
      
      ## precision mean/std
      gax = sns.catplot(x="para_idx", y="precision_mean", hue="range",
                       row="mode", col='headSele_idx', data=filter_df, kind='point',
                       height=4,aspect=1.5,palette=sns.color_palette(),ci=None,dodge=0.4,linestyles='--',
                       order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1))
      
      # draw error bar
      #gax.map(errbar,'para_idx','precision_mean','precision_std')

      gax.map(plt.axhline,y=prec_bg_dict['all_mean'],color='gray',ls='solid',lw=1.,label='all_bg')
      gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
      gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')

      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/layerwise_logis_meanstd_layer{}_topK{}.png'.format(working_dir,tar_fig_dir,layer_togo,topK_togo))
      plt.close()

def contact_precision_fig_headwise(rp_set: str='rp15_all',
                                   rp_mdl_idx: str='_1',
                                   init_epoch: str='330',
                                   tar_fig_dir: str=None,
                                   prec_bg_dict: dict=None,
                                   n_head: int=8,
                                   n_layer: int=4,
                                   blc_flag: str = '',
                                   stre_wgt: str = '1',
                                   parIdx_list: List = [5,6,7,2],
                                   chlIdx_list: List = [1,0,4,5,6,8]
                                   ):

  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_biSuper_bert_eval'.format(working_dir)
  metric_list = ['max_precision_all_5','max_precision_all_2','max_precision_all_1','max_precision_short_5','max_precision_short_2','max_precision_short_1','max_precision_medium_5','max_precision_medium_2','max_precision_medium_1','max_precision_long_5','max_precision_long_2','max_precision_long_1','mean_precision_all_5','mean_precision_all_2','mean_precision_all_1','mean_precision_short_5','mean_precision_short_2','mean_precision_short_1','mean_precision_medium_5','mean_precision_medium_2','mean_precision_medium_1','mean_precision_long_5','mean_precision_long_2','mean_precision_long_1','apc_precision_all_5','apc_precision_all_2','apc_precision_all_1','apc_precision_short_5','apc_precision_short_2','apc_precision_short_1','apc_precision_medium_5','apc_precision_medium_2','apc_precision_medium_1','apc_precision_long_5','apc_precision_long_2','apc_precision_long_1']
  
  if rp_mdl_idx == '':
    sele_flag = {
      5: np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      6: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      7: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
      2: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      }
  elif rp_mdl_idx == '_4':
    sele_flag = {
      1: np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      2: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      3: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
      4: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      }
  else:
    sele_flag = None

  df_list = []


  # myModel: load head-wise precision
  data_name = 'pdbmap_contactData' # pdbmap_contactData, logistic_datasets
  set_nm = 'holdout' # holdout, valid, tr-val
  set_type = 'contact' # contact, logistic
  if len(blc_flag) > 0 :
    blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  else:
    blc_set={'con':'','nonCon':'','ce':''}
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        chl_idx = chlIdx_list[chl_i]
        ppar_idx = parIdx_list[ppar_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          print('>log file: {}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system("rm tmp_rec")
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce{}_strethWeight{}_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1-7; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[2]
            symm_way = name_split[0]
            mode = mode_pair
            topK = 'L/{}'.format(name_split[3])
            headSele_idx = ppar_i + 1 # convert ot 1,2,3...
            para_idx = chl_i + 1 # convert to 1,2,3...
            dt_set = set_nm

            metric_value = np.array(metric_json[metric_nm])
            metric_value_indiv_mean = np.array(metric_json[metric_nm+'_indiv_mean'])
            metric_value_indiv_std = np.array(metric_json[metric_nm+'_indiv_std'])
            
            for lay in range(metric_value.shape[0]):
              for hea in range(metric_value.shape[1]):
                head_idx = hea + 1
                layer_idx = lay + 1
                reg_flag = sele_flag[ppar_idx][lay][hea]
                precision = metric_value[lay][hea]
                precision_indiv_mean = metric_value_indiv_mean[lay][hea]
                precision_indiv_std = metric_value_indiv_std[lay][hea]
                df_list.append([precision,precision_indiv_mean,precision_indiv_std,mode,range_nm,topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])

  df = pd.DataFrame(df_list,columns=['precision','precision_mean','precision_std','mode','range','topK','headSele_idx','para_idx','dt_set','head_idx','layer_idx','symm','reg_flag'])

  # Draw figure: logistic layer-wise
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  for range_togo in range_togo_list:
    for topK_togo in topK_togo_list:
      filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["range"]==range_togo)]
      gax = sns.catplot(x="para_idx", y="precision", hue="layer_idx",
                        row="mode", col='headSele_idx', data=filter_df, kind='box',
                        height=4,aspect=1.5,palette=sns.color_palette(),
                        order=range(1,len(chlIdx_list)+1),hue_order=range(1,n_layer+1),
                        row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                        whis=10.0,width=0.5)

      gax.map(plt.axhline,y=prec_bg_dict[range_togo],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' # head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()
      
      ## precision mean/std
      gax = sns.catplot(x="para_idx", y="precision_mean", hue="layer_idx",
                       row="mode", col='headSele_idx', data=filter_df, kind='box',
                       height=4,aspect=1.5,palette=sns.color_palette(),
                       order=range(1,len(chlIdx_list)+1),hue_order=range(1,n_layer+1),
                       row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                       whis=10.0,width=0.5)
      
      gax.map(plt.axhline,y=prec_bg_dict['{}_mean'.format(range_togo)],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')

      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_mean_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()
    
      ## layerSV / layerNonSV
      filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["range"]==range_togo)]
      gax = sns.catplot(x="para_idx", y="precision", hue="reg_flag",
                        row="mode", col='headSele_idx', data=filter_df, kind='box',
                        height=4,aspect=1.5,palette=sns.color_palette(),
                        order=range(1,len(chlIdx_list)+1),hue_order=[1,0],
                        row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                        whis=10.0,width=0.5)

      gax.map(plt.axhline,y=prec_bg_dict[range_togo],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' # head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_SV-NSV_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()
      
      ## precision mean/std
      gax = sns.catplot(x="para_idx", y="precision_mean", hue="reg_flag",
                       row="mode", col='headSele_idx', data=filter_df, kind='box',
                       height=4,aspect=1.5,palette=sns.color_palette(),
                       order=range(1,len(chlIdx_list)+1),hue_order=[1,0],
                       row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                       whis=10.0,width=0.5)
      
      gax.map(plt.axhline,y=prec_bg_dict['{}_mean'.format(range_togo)],color='gray',ls='solid',lw=1.,label='{}_bg'.format(range_togo))
      #gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      #gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')

      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_mean_SV-NSV_range{}_topK{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
      plt.close()

  for layer_togo in range(1,n_layer+1):
    for topK_togo in topK_togo_list:
      filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["layer_idx"]==layer_togo)]
      gax = sns.catplot(x="para_idx", y="precision", hue="range",
                       row="mode", col='headSele_idx', data=filter_df, kind='box',
                       height=4,aspect=1.5,palette=sns.color_palette(),
                       order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],
                       row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                       whis=10.0,width=0.5)
      
      gax.map(plt.axhline,y=prec_bg_dict['all'],color='gray',ls='solid',lw=1.,label='all_bg')
      gax.map(plt.axhline,y=prec_bg_dict['short'],color='gray',ls='dotted',lw=1.,label='short_bg')
      gax.map(plt.axhline,y=prec_bg_dict['medium'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      gax.map(plt.axhline,y=prec_bg_dict['long'],color='gray',ls='dashdot',lw=1.,label='long_bg')
        
      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_layer{}_topK{}.png'.format(working_dir,tar_fig_dir,layer_togo,topK_togo))
      plt.close()
      
      ## precision mean/std
      gax = sns.catplot(x="para_idx", y="precision_mean", hue="range",
                       row="mode", col='headSele_idx', data=filter_df, kind='box',
                       height=4,aspect=1.5,palette=sns.color_palette(),
                       order=range(1,len(chlIdx_list)+1),hue_order=['all','short','medium','long'],
                       row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                       whis=10.0,width=0.5)
      
      gax.map(plt.axhline,y=prec_bg_dict['all_mean'],color='gray',ls='solid',lw=1.,label='all_bg')
      gax.map(plt.axhline,y=prec_bg_dict['short_mean'],color='gray',ls='dotted',lw=1.,label='short_bg')
      gax.map(plt.axhline,y=prec_bg_dict['medium_mean'] ,color='gray',ls='dashed',lw=1.,label='medium_bg')
      gax.map(plt.axhline,y=prec_bg_dict['long_mean'],color='gray',ls='dashdot',lw=1.,label='long_bg')

      formatter = FuncFormatter(to_percent)
      plt.gca().yaxis.set_major_formatter(formatter)
        
      #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
      if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
        os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
      gax.savefig('{}/results_to_keep/figures/{}/headwise_mean_layer{}_topK{}.png'.format(working_dir,tar_fig_dir,layer_togo,topK_togo))
      plt.close()

def contact_precision_fig_all_distribution(rp_set: str='rp15_all',
                                           rp_mdl_idx: str='_1',
                                           init_epoch: str='330',
                                           tar_fig_dir: str=None,
                                           prec_bg_dict: dict=None,
                                           n_head: int=8,
                                           n_layer: int=4,
                                           blc_flag: str = '',
                                           stre_wgt: str = '1',
                                           parIdx_list: List = [5,6,7,2],
                                           chlIdx_list: List = [1,0,4,5,6,8]
                                           ):

  working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  log_dir = '{}/job_logs/archive_biSuper_bert_eval'.format(working_dir)
  metric_all_distri_list = ['all_pred_distribution_apc_1_corr_S-M-L','all_pred_distribution_apc_1_corr','all_pred_distribution_apc_1_total',
      'all_pred_distribution_apc_2_corr_S-M-L','all_pred_distribution_apc_2_corr','all_pred_distribution_apc_2_total',
      'all_pred_distribution_apc_5_corr_S-M-L','all_pred_distribution_apc_5_corr', 'all_pred_distribution_apc_5_total',
      'all_pred_distribution_max_1_corr_S-M-L','all_pred_distribution_max_1_corr', 'all_pred_distribution_max_1_total',
      'all_pred_distribution_max_2_corr_S-M-L','all_pred_distribution_max_2_corr', 'all_pred_distribution_max_2_total',
      'all_pred_distribution_max_5_corr_S-M-L','all_pred_distribution_max_5_corr', 'all_pred_distribution_max_5_total',
      'all_pred_distribution_mean_1_corr_S-M-L','all_pred_distribution_mean_1_corr','all_pred_distribution_mean_1_total',
      'all_pred_distribution_mean_2_corr_S-M-L','all_pred_distribution_mean_2_corr','all_pred_distribution_mean_2_total',
      'all_pred_distribution_mean_5_corr_S-M-L','all_pred_distribution_mean_5_corr', 'all_pred_distribution_mean_5_total']
  
  if rp_mdl_idx == '':
    sele_flag = {
      5: np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      6: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      7: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
      2: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      }
  elif rp_mdl_idx == '_4':
    sele_flag = {
      1: np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      2: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]),
      3: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]),
      4: np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
      }
  else:
    sele_flag = None
  
  if len(blc_flag) > 0 :
    blc_set={'con':'_blc','nonCon':'_blc','ce':'_blc_gm2.0'}
  else:
    blc_set={'con':'','nonCon':'','ce':''}
  
  df_list = []

  # load correct prediction distribution in 'ALL' range
  data_name = 'pdbmap_contactData' # pdbmap_contactData, logistic_datasets
  set_type = 'contact' # contact, logistic
  topk_list = [1,2,5]
  for mode_pair in ['con','nonCon','ce']:
    for ppar_i in range(len(parIdx_list)):
      for chl_i in range(len(chlIdx_list)):
        chl_idx = chlIdx_list[chl_i]
        ppar_idx = parIdx_list[ppar_i]
        for set_nm in ['holdout']:
          par_idx = '{}_{}'.format(ppar_idx,chl_idx)
          print('loading set: {}_{}'.format(mode_pair,par_idx))
          log_fl = 'biSuper_bert_{}{}_{}_torch_eval_{}{}_{}.{}-{}.0.out'.format(rp_set,rp_mdl_idx,init_epoch,mode_pair,blc_set[mode_pair],par_idx,set_nm,set_type)
          print('>log file: {}'.format(log_fl))
          os.system("grep 'loading weights file' {}/{} | cut -d'/' -f13 > tmp_rec".format(log_dir,log_fl))
          with open('tmp_rec', 'r') as f:
            tar_dir = f.read()[:-1]
          os.system("rm tmp_rec")
          print('>model dir:',tar_dir)
          print('>json file: results_metrics_{}_{}.json'.format(data_name,set_nm))
          json_fl='{}/results_to_keep/{}/{}{}_{}_con_non_ce{}_strethWeight{}_models/{}{}/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt,mode_pair,blc_set[mode_pair],tar_dir,data_name,set_nm)
          with open(json_fl,'r') as f:
            metric_json = json.load(f)
          # convert json to dataFrame format
          # mode: max,mean,apc; range: all,short,medium,long; topK: L,L/2,L/5
          # headSele_idx: 1-7; para_idx: 0-9; layer_idx: 1-4; head_idx: 1-8,
          # dt_set: train,valid,holdout
          for metric_nm in metric_all_distri_list:
            name_split = re.split('_',metric_nm)
            if name_split[-1] == 'S-M-L':
              symm_way = name_split[3]
              mode = mode_pair
              topK = 'L/{}'.format(name_split[4])
              headSele_idx = ppar_i + 1 # convert ot 1,2,3...
              para_idx = chl_i + 1 # convert to 1,2,3...
              dt_set = set_nm

              metric_value = np.array(metric_json[metric_nm]) # [n_layer,n_head,3(S,M,L)]
              for lay in range(metric_value.shape[0]):
                for hea in range(metric_value.shape[1]):
                  head_idx = hea + 1
                  layer_idx = lay + 1
                  reg_flag = sele_flag[ppar_idx][lay][hea]
                  corr_num_s = metric_value[lay][hea][0]
                  corr_num_m = metric_value[lay][hea][1]
                  corr_num_l = metric_value[lay][hea][2]
                  corr = corr_num_s + corr_num_m + corr_num_l
                  df_list.append([corr_num_s/corr,mode,'short',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])
                  df_list.append([corr_num_m/corr,mode,'medium',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])
                  df_list.append([corr_num_l/corr,mode,'long',topK,headSele_idx,para_idx,dt_set,head_idx,layer_idx,symm_way,reg_flag])

  df = pd.DataFrame(df_list,columns=['corr_num','mode','range','topK','headSele_idx','para_idx','dt_set','head_idx','layer_idx','symm','reg_flag'])
  
  ## Draw fig: prediction ditribution in ALL range
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['short','medium','long']
  topK_togo_list = ['1','2','5']
  symm_togo_list = ['apc','max','mean']
  reg_togo_list = [0,1]
  for topK_togo in topK_togo_list:
    for symm_togo in symm_togo_list:
      for reg_togo in reg_togo_list:
        filter_df = df.loc[(df["dt_set"]=='holdout') & (df["topK"]=='L/'+topK_togo) & (df["symm"]==symm_togo) & (df["reg_flag"]==reg_togo)] # all TP distribition
        gax = sns.catplot(x="para_idx", y="corr_num", hue="range",
                          row="mode", col='headSele_idx', data=filter_df, kind='box',
                          height=4, aspect=1.5,palette=sns.color_palette(),
                          order=range(1,len(chlIdx_list)+1),hue_order=['short','medium','long'],
                          row_order=['con','nonCon','ce'],col_order=range(1,len(parIdx_list)+1),
                          whis=10.0,width=0.5)

        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        
        #tar_fig_dir = 'rp15_all_blc@330' #logistics head-wise
        if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
          os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
        gax.savefig('{}/results_to_keep/figures/{}/alldistri_topK{}_symm{}_reg{}.png'.format(working_dir,tar_fig_dir,topK_togo,symm_togo,reg_togo))
        plt.close()

def generate_logis_data(working_dir):
  train_maxNum = 20
  valid_maxNum = 20
  train_dataDir = '{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_train.lmdb'.format(working_dir)
  valid_dataDir = '{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_valid.lmdb'.format(working_dir)
  
  train_env = lmdb.open(str(train_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
  valid_env = lmdb.open(str(valid_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

  # loop all data to find seq with length longer than 250
  lenCut = 250
  train_candIdx = []
  valid_candIdx = []
  with train_env.begin(write=False) as txn:
    train_num_examples = pkl.loads(txn.get(b'num_examples'))
    for idx in range(train_num_examples):
      item = pkl.loads(txn.get(str(idx).encode()))
      if item['targetSeq_len'] >= lenCut:
        train_candIdx.append(item['id'])
  with valid_env.begin(write=False) as txn:
    valid_num_examples = pkl.loads(txn.get(b'num_examples'))
    for idx in range(valid_num_examples):
      item = pkl.loads(txn.get(str(idx).encode()))
      if item['targetSeq_len'] >= lenCut:
        valid_candIdx.append(item['id'])

 
  # random select 20 protein for logistic train, 20 for logistic validation
  np.random.seed(seed=0)
  train_seleIdxs = np.random.choice(train_candIdx,size=train_maxNum)
  valid_seleIdxs = np.random.choice(valid_candIdx,size=valid_maxNum)
  
  # save selected indices
  np.savetxt('{}/pdbmap_sets/pdbmap_contactData/logistic_datasets/lenCut_l8h500_rand_train.1.idx.csv'.format(working_dir),
             train_seleIdxs,fmt='%i',delimiter=',')
  np.savetxt('{}/pdbmap_sets/pdbmap_contactData/logistic_datasets/lenCut_l8h500_rand_valid.1.idx.csv'.format(working_dir),
             valid_seleIdxs,fmt='%i',delimiter=',')


  # gather examples
  whole_set = []
  with train_env.begin(write=False) as txn:
    for idx in train_seleIdxs:
      item = pkl.loads(txn.get(str(idx).encode()))
      item['type_flag'] = 1
      whole_set.append(item)
  with valid_env.begin(write=False) as txn:
    for idx in valid_seleIdxs:
      item = pkl.loads(txn.get(str(idx).encode()))
      item['type_flag'] = 0
      whole_set.append(item)

  whole_wrtDir = '{}/pdbmap_sets/pdbmap_contactData/logistic_datasets/allData_lenCut_l8h500_tr-val.lmdb'.format(working_dir)
  map_size = (1024 * 5) * (2 ** 20) # 5G
  whole_wrtEnv = lmdb.open(whole_wrtDir, map_size=map_size)
  
  with whole_wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(whole_set):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  whole_wrtEnv.close()
  
def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    #shape = [batch_size] + [475]*len(sequences[0].shape)


    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def train_esm_logistic(working_dir,model_name):
  # load data
  data_dir = '{}/pdbmap_sets/pdbmap_contactData/logistic_datasets'.format(working_dir)
  with open('{}/logistic_models_esm/{}_output_predictions_logistic_datasets_tr-val.pkl'.format(data_dir,model_name), 'rb') as f:
    dt_list = pkl.load(f)
  print('tr-val size:{}'.format(len(dt_list)))
  contactMap,attentionMat,valid_mask,seq_length,type_flag = [],[],[],[],[]
  for one_dt in dt_list:
    #print('target_contact:',one_dt['target_contact'].shape)
    #print('attenMat:',one_dt['attenMat'].shape)
    #print('val_mask:',one_dt['val_mask'].shape)
    #print('seq_length:',one_dt['seq_length'].shape)
    #print('type_flag:',one_dt['type_flag'].shape)
    contactMap.append(one_dt['target_contact'])
    attentionMat.append(one_dt['attenMat'])
    valid_mask.append(one_dt['val_mask'])
    seq_length.append(one_dt['seq_length'])
    type_flag.append(one_dt['type_flag'])
  
  contactMap = pad_sequences(contactMap,0)
  attentionMat = pad_sequences(attentionMat,0)
  valid_mask = pad_sequences(valid_mask,False)
  seq_length = np.asarray(seq_length)
  type_flag = np.asarray(type_flag)

  #print('contactMap:',contactMap.shape)
  #print('attentionMat:',attentionMat.shape)
  #print('valid_mask:',valid_mask.shape)
  #print('seq_length:',seq_length.shape)
  #print('type_flag:',type_flag.shape)
  
  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

  #print('val_all_mask_mat:',np.sum(val_all_mask_mat,axis=(-1,-2)))


  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  #print('(y_ind - x_ind) >= 6',(y_ind - x_ind) >= 6)
  val_all_mask_mat = np.logical_and(val_all_mask_mat, (y_ind - x_ind)>=6)
  val_short_mask_mat = np.logical_and(val_short_mask_mat, np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
  val_medium_mask_mat = np.logical_and(val_medium_mask_mat, np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
  val_long_mask_mat = np.logical_and(val_long_mask_mat, (y_ind - x_ind) >= 24)
 
  #print('val_all_mask_mat:',np.sum(val_all_mask_mat,axis=(-1,-2)))
  #print('val_short_mask_mat:',np.sum(val_short_mask_mat,axis=(-1,-2)))
  #print('val_medium_mask_mat:',np.sum(val_medium_mask_mat,axis=(-1,-2)))
  #print('val_long_mask_mat:',np.sum(val_long_mask_mat,axis=(-1,-2)))

  # contactMap size: [bs, L_max, L_max]
  # attentionMat size: [bs, n_layer, n_head, L_max, L_max]
  # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
  attenMat_symm = 0.5 * (attentionMat + np.transpose(attentionMat, (0,1,2,4,3)))
  # rowSum/colSum size: [bs,n_layer,n_head,L_max]; allSum [bs,n_layer,n_head];
  # all broadcast to [bs,n_layer,n_head,L_max,L_max]
  attenMat_symm_rowSum = np.repeat(np.sum(attenMat_symm, axis=-1)[:,:,:,:,None],attenMat_symm.shape[-1],axis=-1)
  attenMat_symm_colSum = np.repeat(np.sum(attenMat_symm, axis=-2)[:,:,:,None,:],attenMat_symm.shape[-1],axis=-2)
  attenMat_symm_allSum = np.repeat(np.repeat(np.sum(attenMat_symm,axis=(-2,-1))[:,:,:,None,None],attenMat_symm.shape[-1],axis=-1),attenMat_symm.shape[-1],axis=-2)
  # attenMat_symm_apc [bs, n_layer, n_head, L_max, L_max]
  #print('attenMat_symm:',attenMat_symm.shape)
  #print('attenMat_symm_rowSum:',attenMat_symm_rowSum.shape)
  #print('attenMat_symm_colSum:',attenMat_symm_colSum.shape)
  #print('attenMat_symm_allSum:',attenMat_symm_allSum.shape)
  attenMat_symm_apc = attenMat_symm-attenMat_symm_rowSum*attenMat_symm_colSum/attenMat_symm_allSum
  
  # split train / valid set
  train_idx = np.squeeze(np.argwhere(type_flag == 1))
  valid_idx = np.squeeze(np.argwhere(type_flag == 0))
  # prepare validation set (all, short, medium, long)
  valid_contMap = contactMap[valid_idx]
  valid_atteMat = attentionMat[valid_idx]
  #print('valid_idx:',valid_idx)
  valid_all_valMask = val_all_mask_mat[valid_idx]  # [n_valid,L_max,L_max]
  valid_short_valMask = val_short_mask_mat[valid_idx]
  valid_medium_valMask = val_medium_mask_mat[valid_idx]
  valid_long_valMask = val_long_mask_mat[valid_idx]

  #valid pair numbers for each example
  valid_all_valMask_count = np.sum(valid_all_valMask,axis=(-1,-2)) # [n_valid,]
  #print('valid_all_valMask_count:',valid_all_valMask_count)
  valid_short_valMask_count = np.sum(valid_short_valMask,axis=(-1,-2))
  valid_medium_valMask_count = np.sum(valid_medium_valMask,axis=(-1,-2))
  valid_long_valMask_count = np.sum(valid_long_valMask,axis=(-1,-2))

  def sele_pair(atteM, valM):
    #print('atteM:',atteM.shape)
    #print('valM:',valM.shape)
    #print('valM:',valM)
    fil_atte = atteM[:,:,valM]
    fil_atte_shape = fil_atte.shape
    #print('fil_atte_shape:',fil_atte_shape)
    return fil_atte.reshape(-1,fil_atte_shape[-1]).transpose(1,0)

  def calc_prec(targs,preds,mask_count,lengths,topk):
    """
    targ: true contactMap with value as 0/1, [total num of valid pairs,]
    pred: predicted prob for class 1, [total num of valid pairs,]
    mask_count: num of valid pairs for each example, [n_examples,] 
    topk: L/topk
    """
    correct = 0.
    total = 0.
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      targ_set = targs[cunt_sum:cunt_sum+count]
      pred_set = preds[cunt_sum:cunt_sum+count]
      cunt_sum += count
      most_likely_idx = np.argpartition(-pred_set,kth=length//topk)[:(length//topk)+1]
      selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) #size[seq_length,]
      correct += np.sum(selected)
      total += length
    return correct / total

  # select working pairs
  valid_all_valContMap = valid_contMap[valid_all_valMask]
  valid_short_valContMap = valid_contMap[valid_short_valMask]
  valid_medium_valContMap = valid_contMap[valid_medium_valMask]
  valid_long_valContMap = valid_contMap[valid_long_valMask]

  valid_all_valAtteMat = np.vstack(list(map(sele_pair,valid_atteMat,valid_all_valMask)))
  valid_short_valAtteMat = np.vstack(list(map(sele_pair,valid_atteMat,valid_short_valMask)))
  valid_medium_valAtteMat = np.vstack(list(map(sele_pair,valid_atteMat,valid_medium_valMask)))
  valid_long_valAtteMat = np.vstack(list(map(sele_pair,valid_atteMat,valid_long_valMask)))
 
  valid_prec_all = []
  valid_prec_short = []
  valid_prec_medium = []
  valid_prec_long = []

  l1_weight_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  #l1_weight_list = [0.001,1,50]
  
  for train_num in range(1,len(train_idx)+1):
  #for train_num in range(1,3):
    print('train_num',train_num)
    train_contMap = contactMap[train_idx[0:train_num]] # [n_train,L_max,L_max]
    train_atteMat = attentionMat[train_idx[0:train_num]] # [n_train/valid,n_layer,n_head,L_max,L_max]
    train_valMask = val_all_mask_mat[train_idx[0:train_num]] # [n_train,L_max,L_max]
 
    train_valContMap = train_contMap[train_valMask] # [num of all valid pairs of all examples,]
    train_valAtteMat = np.vstack(list(map(sele_pair,train_atteMat,train_valMask)))
    #print('label:',np.unique(train_valContMap))

    # tune l1 weight
    valid_prec_all_l1 = []
    valid_prec_short_l1 = []
    valid_prec_medium_l1 = []
    valid_prec_long_l1 = []
    for l1w in l1_weight_list:
      print('l1_weight:',l1w)
      mdl = LogisticRegression(penalty = 'l1',C=1.0/l1w,random_state=0,tol=0.001,solver='saga',max_iter=5000).fit(train_valAtteMat,train_valContMap)
      joblib.dump(mdl,'{}/logistic_models_esm/model_{}_{}'.format(data_dir,train_num,l1w))
      posi_label_idx = np.argwhere(mdl.classes_ == 1)[0][0]

      valid_all_contM_pred = mdl.predict_proba(valid_all_valAtteMat)
      valid_short_contM_pred = mdl.predict_proba(valid_short_valAtteMat)
      valid_medium_contM_pred = mdl.predict_proba(valid_medium_valAtteMat)
      valid_long_contM_pred = mdl.predict_proba(valid_long_valAtteMat)
      # calculate precision
      prec_all = calc_prec(valid_all_valContMap,valid_all_contM_pred[:,posi_label_idx],valid_all_valMask_count,seq_length,1)    
      prec_short = calc_prec(valid_short_valContMap,valid_short_contM_pred[:,posi_label_idx],valid_short_valMask_count,seq_length,1)    
      prec_medium = calc_prec(valid_medium_valContMap,valid_medium_contM_pred[:,posi_label_idx],valid_medium_valMask_count,seq_length,1) 
      prec_long = calc_prec(valid_long_valContMap,valid_long_contM_pred[:,posi_label_idx],valid_long_valMask_count,seq_length,1)    
      # record precision
      valid_prec_all_l1.append(prec_all)
      valid_prec_short_l1.append(prec_short)
      valid_prec_medium_l1.append(prec_medium)
      valid_prec_long_l1.append(prec_long)
    # record precision
    valid_prec_all.append(valid_prec_all_l1)
    valid_prec_short.append(valid_prec_short_l1)
    valid_prec_medium.append(valid_prec_medium_l1)
    valid_prec_long.append(valid_prec_long_l1)
  # select best model
  valid_prec_all = np.array(valid_prec_all)
  valid_prec_short = np.array(valid_prec_short)
  valid_prec_medium = np.array(valid_prec_medium)
  valid_prec_long = np.array(valid_prec_long)

  valid_prec_all_best_idx = np.unravel_index(valid_prec_all.argmax(), valid_prec_all.shape)
  valid_prec_short_best_idx = np.unravel_index(valid_prec_short.argmax(), valid_prec_short.shape)
  valid_prec_medium_best_idx = np.unravel_index(valid_prec_medium.argmax(), valid_prec_medium.shape)
  valid_prec_long_best_idx = np.unravel_index(valid_prec_long.argmax(), valid_prec_long.shape)

  # retrain and save best model
  valid_mode = ['all','short','medium','long']
  valid_prec_idxs = [valid_prec_all_best_idx,valid_prec_short_best_idx,valid_prec_medium_best_idx,valid_prec_long_best_idx]
  valid_prec_mat = [valid_prec_all,valid_prec_short,valid_prec_medium,valid_prec_long]
  valid_prec_max, valid_prec_min = np.amax(valid_prec_mat), np.amin(valid_prec_mat)
  valid_prec_best = []
  for i in range(len(valid_mode)):
    idx_tuple = valid_prec_idxs[i]
    valid_prec_best.append(np.amax(valid_prec_mat[i]))
    num_best = idx_tuple[0]
    l1w_best = l1_weight_list[idx_tuple[1]]
    mode = valid_mode[i]
    print('>_mode: {}; num_best: {}; l1w_best: {}'.format(mode,num_best+1,l1w_best))
 
    # joblib.load(filename)

    # plot gridsearch figure
    fig,ax = plt.subplots(figsize=(16,20))
    ax_mat = ax.matshow(valid_prec_mat[i], cmap=plt.cm.hot_r, vmin=valid_prec_min, vmax=valid_prec_max)
    ax.set_xticks(np.arange(len(l1_weight_list)))
    ax.set_yticks(np.arange(valid_prec_mat[i].shape[0]))
    ax.set_yticklabels(np.arange(1,valid_prec_mat[i].shape[0]+1), rotation=45)
    ax.set_xticklabels(l1_weight_list, rotation=45)
    for (ii, jj), z in np.ndenumerate(valid_prec_mat[i]):
          ax.text(jj, ii, '{:0.3f}'.format(z), ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    cb = fig.colorbar(ax_mat)
    plt.savefig('{}/logistic_models_esm/gridSear_model_{}.png'.format(data_dir,mode))
    plt.close(fig)

    # save validation set precision 
    np.savetxt('{}/logistic_models_esm/gridSear_model_{}.csv'.format(data_dir,mode),valid_prec_mat[i],fmt='%.3f',delimiter=',')
  return valid_prec_best

def json2lmdb(jsonFile_names: List = None,
              working_dir: str = None,
              map_size_G: int = 15):
  '''convert name.json to name.lmdb
  '''
  map_size = (1024 * map_size_G) * (2 ** 20) # 15G
  
  for jfl in jsonFiles:
    # load josn data
    with open('{}/{}.json'.format(working_dir,jfl),'r') as fl:
      allData = json.load(fl)
    
    wrtDir = '{}/{}.lmdb'.format(working_dir,jfl)
    wrtEnv = lmdb.open(wrtDir, map_size=map_size)
    
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(allData):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i + 1))
    wrtEnv.close()

  return None

def seq_lmdb_subtract(working_dir):
  sbtor_dataDir = '{}/seq_json_rp75/pfam_holdout_lenCut.lmdb'.format(working_dir)
  sbtee_dataDir = '{}/seq_json_rp15/pfam_holdout_lenCut.lmdb'.format(working_dir)

  sbtor_env = lmdb.open(str(sbtor_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
  sbtee_env = lmdb.open(str(sbtee_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

  # collect sbtee's unpIden
  sbtee_unpIden = []
  with sbtee_env.begin(write=False) as txn:
    sbtee_num_examples = pkl.loads(txn.get(b'num_examples'))
    print('rp15 holdout num: {}'.format(sbtee_num_examples))
    for idx in range(sbtee_num_examples):
      item = pkl.loads(txn.get(str(idx).encode()))
      sbtee_unpIden.append(item['unpIden'])

  # assemble new set
  new_set = []
  with sbtor_env.begin(write=False) as txn:
    sbtor_num_examples = pkl.loads(txn.get(b'num_examples'))
    print('rp75 holdout num: {}'.format(sbtor_num_examples))
    for idx in range(sbtor_num_examples):
      item = pkl.loads(txn.get(str(idx).encode()))
      if item['unpIden'] not in sbtee_unpIden:
        new_set.append(item)

  newSet_wrtDir = '{}/seq_json_rp75/pfam_holdout-filrp15_lenCut.lmdb'.format(working_dir)
  map_size = (1024 * 15) * (2 ** 20) # 15G
  newSet_wrtEnv = lmdb.open(newSet_wrtDir, map_size=map_size)
  with newSet_wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(new_set):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  print('new rp75 holdout num: {}'.format(i+1))

def range_set_count(working_dir):
  """
  to count number of valid residue pairs in each range (short, medium, long)
  sub-short pairs (|i-j|<6) are ignored
  """
  set_name = 'holdout'
  set_dataDir = '{}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_{}.lmdb'.format(working_dir,set_name)
  set_env = lmdb.open(str(set_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
  count_short,count_medium,count_long = 0, 0, 0
  count_short_con, count_medium_con, count_long_con = [0,0], [0,0], [0,0]
  with set_env.begin(write=False) as txn:
    set_num_examples = pkl.loads(txn.get(b'num_examples'))
    print('in total, {} examples'.format(set_num_examples))
    for idx in range(set_num_examples):
      if idx % 1000 == 0:
        print('> at exp {}'.format(idx))
      item = pkl.loads(txn.get(str(idx).encode()))
      #print(item.keys())
      valid_mask = np.array(item["valid_mask"])
      contactMap = np.array(item["contact-map"])
      #seq_length = item["seq_length"]
      assert valid_mask.shape[0] == contactMap.shape[0]
      valid_mask_mat = valid_mask[:,None] & valid_mask[None,:] # size [L,L]
      seqpos = np.arange(valid_mask.shape[0])
      x_ind, y_ind = np.meshgrid(seqpos, seqpos)
      valid_mask_mat_short = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
      valid_mask_mat_medium = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
      valid_mask_mat_long = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 24)
      # count pairs in ranges (each pair is counted only once)
      count_short += np.sum(valid_mask_mat_short)
      count_medium += np.sum(valid_mask_mat_medium)
      count_long += np.sum(valid_mask_mat_long)
      # count contact and non-contact pairs in each range
      valid_conM_short = contactMap[valid_mask_mat_short]
      valid_conM_medium = contactMap[valid_mask_mat_medium]
      valid_conM_long = contactMap[valid_mask_mat_long]
      
      count_short_con[0] += np.sum(valid_conM_short)
      count_medium_con[0] += np.sum(valid_conM_medium)
      count_long_con[0] += np.sum(valid_conM_long)
      
      count_short_con[1] += np.sum(1-valid_conM_short)
      count_medium_con[1] += np.sum(1-valid_conM_medium)
      count_long_con[1] += np.sum(1-valid_conM_long)


  count_all = count_short+count_medium+count_long
  print('all pairs: {}; short pairs: {}; medium pairs: {}; long pairs: {}'.format(count_all,count_short,count_medium,count_long))
  print('short-con: {}, short-nonCon: {};'.format(count_short_con[0],count_short_con[1]))
  print('medium-con: {}, medium-nonCon: {};'.format(count_medium_con[0],count_medium_con[1]))
  print('long-con: {}, long-nonCon: {};'.format(count_long_con[0],count_long_con[1]))

  os.system("echo 'all pairs: {}; short pairs: {}; medium pairs: {}; long pairs: {}' > {}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_range_count_{}.txt".format(count_all,count_short,count_medium,count_long,working_dir,set_name)) 
  os.system("echo 'short-con: {}, short-nonCon: {};\nmedium-con: {}, medium-nonCon: {};\nlong-con: {}, long-nonCon: {};' >> {}/pdbmap_sets/pdbmap_contactData/allData_lenCut_l8h500_range_count_{}.txt".format(count_short_con[0],count_short_con[1],count_medium_con[0],count_medium_con[1],count_long_con[0],count_long_con[1],working_dir,set_name)) 

if __name__ == "__main__":
  # load background precision
  prec_bg_dict = {
      'all': 0.0186,
      'short': 0.0528,
      'medium': 0.0363,
      'long': 0.0145,
      'all_mean': 0.0306,
      'all_std': 0.0213,
      'short_mean': 0.0552,
      'short_std': 0.0221,
      'medium_mean': 0.0403,
      'medium_std': 0.0233,
      'long_mean': 0.0257,
      'long_std': 0.0210
      }


  working_dir = "/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_32.0"
  #uniAcc_range_jsonVer(working_dir)
  #fl2load = sys.argv[1]
  #print(fl2load)
  #bestPdb_resolution(working_dir,fl2load)
  #print(queryApi_pdbInfo(working_dir,'2IGY','A','Q17SC2'))
  #pfam_contact(working_dir,fl2load)
  #prune_seqNeibor_wrap(working_dir, [-1,0,1])
  #filterByLen(working_dir, 8, 500)
  #vis_contactMap(working_dir,fl2load)
  #split_data(working_dir)
  add_valid_mask(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_32.0/pdbmap_sets/pdbmap_contactData',
              file_list = ['allData_lenCut_l8h500'],
              input_format = 'json',
              save_json = True,
              save_lmdb = True,
              save_sample = True)
  
  ### evaluate pretrained models
  '''
  rp_set = 'rp15_all'
  rp_mdl_idx ='4'
  tar_fig_dir = '{}_{}_pretrain'.format(rp_set,rp_mdl_idx)
  n_head = 12
  n_layer = 6
  logIdx_dict = {'1':[0,1,2,3,4,5],
                 '2':[0,1,2,3],
                 '3':[0,1,2,3],
                 '4':[0,1,2]
                }
  epoch_dict = {'1_0': [0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169],
                '1_1': [179,189,199,209,219,229,239,249,259,269,279,289,299,309,319,329,339],
                '1_2': [349,359,369,379,389,399,409,419,429,439,449,459,469],
                '1_3': [479,489,499,509,519,529,539],
                '1_4': [549,559,569,579,589,599,609,619,629,639,649,659,669,679,689,699,709],
                '1_5': [719,729],
                '2_0': [0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209,219],
                '2_1': [229,239,249,259,269,279,289,299,309,319,329,339,349,359,369,379,389,399,409,419,429,439,449],
                '2_2': [459,469,479,489,499,509,519,529,539,549,559,569,579,589,599,609,619,629,639,649,659,669,679],
                '2_3': [689,699,709,719,729],
                '3_0': [0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209],
                '3_1': [219,229,239,249,259,269,279,289,299,309,319,329,339,349,359,369,379,389,399,409,419,429,439,449,459],
                '3_2': [469,479,489,499,509,519,529,539,549,559,569,579,589,599,609,619,629,639,649,659,669,679,689,699,709],
                '3_3': [719,729],
                '4_0': [0,9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209,219,229,239,249],
                '4_1': [259,269,279,289,299,309,319,329,339,349,359,369,379,389,399,409,419,429,439,449,459,469,479,489,499],
                '4_2': [509,519,529,539,549,559,569,579,589,599,609,619,629,639,649,659,669,679,689,699,709,719,729]      
               }
  
  contact_precision_fig_overall_logistic_pretrain(rp_set = rp_set,
                                                    rp_mdl_idx = rp_mdl_idx,
                                                    tar_fig_dir = tar_fig_dir,
                                                    prec_bg_dict = prec_bg_dict,
                                                    logIdx_dict = logIdx_dict,
                                                    epoch_dict = epoch_dict)
  '''
  ### evaluate AS models
  #contact_precision_fig('catplot')
  '''
  rp_set = 'rp15_all'
  rp_mdl_idx ='_1'
  init_epoch ='300'
  blc_flag = '_blc'
  stre_wgt = '1'
  tar_fig_dir = '{}{}@{}{}_strethW{}'.format(rp_set,rp_mdl_idx,init_epoch,blc_flag,stre_wgt)
  n_head = 12
  n_layer = 6
  parIdx_list = [1,2,3,4,5,6]    #[5,6,7,2]
  chlIdx_list = [1,0,2,4,3,5,7,6,8]                  #[1,0,4,5,6,8]
   
  contact_precision_fig_layersuper_logistic(rp_set = rp_set,
                                            rp_mdl_idx = rp_mdl_idx,
                                            init_epoch = init_epoch,
                                            tar_fig_dir = tar_fig_dir,
                                            prec_bg_dict = prec_bg_dict,
                                            blc_flag = blc_flag,
                                            stre_wgt = stre_wgt,
                                            parIdx_list = parIdx_list,
                                            chlIdx_list = chlIdx_list)
  
  contact_precision_fig_overall_logistic(rp_set = rp_set,
                                         rp_mdl_idx = rp_mdl_idx,
                                         tar_fig_dir = tar_fig_dir,
                                         prec_bg_dict = prec_bg_dict,
                                         init_epoch = init_epoch,
                                         blc_flag = blc_flag,
                                         stre_wgt = stre_wgt,
                                         parIdx_list = parIdx_list, 
                                         chlIdx_list = chlIdx_list)
  '''
  '''
  contact_precision_fig_headwise(rp_set = rp_set,
                                 rp_mdl_idx = rp_mdl_idx,
                                 init_epoch = init_epoch,
                                 tar_fig_dir = tar_fig_dir,
                                 prec_bg_dict = prec_bg_dict,
                                 n_head = n_head,
                                 n_layer = n_layer,
                                 blc_flag = blc_flag,
                                 stre_wgt = stre_wgt,
                                 parIdx_list = parIdx_list,
                                 chlIdx_list = chlIdx_list)
  
  contact_precision_fig_layerwise_logistic(rp_set = rp_set,
                                           rp_mdl_idx = rp_mdl_idx,
                                           init_epoch = init_epoch ,
                                           tar_fig_dir = tar_fig_dir,
                                           prec_bg_dict = prec_bg_dict,
                                           n_head = n_head,
                                           n_layer = n_layer,
                                           blc_flag = blc_flag,
                                           stre_wgt = stre_wgt,
                                           parIdx_list = parIdx_list,
                                           chlIdx_list = chlIdx_list)
  
  contact_precision_fig_all_distribution(rp_set = rp_set,
                                         rp_mdl_idx = rp_mdl_idx,
                                         init_epoch = init_epoch,
                                         tar_fig_dir = tar_fig_dir,
                                         prec_bg_dict = prec_bg_dict,
                                         n_head = n_head,
                                         n_layer= n_layer,
                                         blc_flag = blc_flag,
                                         stre_wgt = stre_wgt,
                                         parIdx_list = parIdx_list,
                                         chlIdx_list = chlIdx_list)
  '''
  #generate_logis_data(working_dir)
  #train_esm_logistic(working_dir,'esm1_t6_43M_UR50S')
  #seq_lmdb_subtract(working_dir)
  #range_set_count(working_dir)
  #json2lmdb(jsonFile_names=['allData_lenCut_l8h500'], path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_32.0/pdbmap_sets/pdbmap_contactData', map_size_G = 15)
