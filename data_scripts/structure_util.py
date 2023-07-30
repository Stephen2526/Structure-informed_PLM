"""
Author:
    Yuanfei Sun

Description:
    utility functions to support protein structure data processing

    SIFTS data provides index mapping (https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html)
      pdb_chain_uniprot: A summary of the PDBe to UniProt residue level mapping, showing the start and end residues of the mapping using SEQRES, PDB sequence and UniProt numbering
      ------
      E.g. 5wiu records in the file
      5wiu,A,P0ABE7,233,284,1002,1053,24,75
      5wiu,A,P0ABE7,287,337,1056,1106,78,128
      5wiu,A,P21917,38,180,34,176,34,176
      5wiu,A,P21917,186,232,182,1001,182,228
      5wiu,A,P21917,338,417,383,462,335,414
      ------

Functions:

"""
import requests

def queryApi_pdbInfo(pdb_id,chain_id,unpAcc):
  """
  Description:
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

  Input:
    * pdb_id: 4-letter pdb id
    * chain_id: asym_id, not author defined chain id
    * unpAcc: uniprot accesion id
  
  Output:
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




def indexMap_api(pdb_id,chain_id,unpAcc):
  """
  Description:
    query index mapping for uniprot-seq, pdb-seq, author-pdb-index from RSCB API
  
  Params:
    * pdb_id:
    * chain_id: should be label_asym_id
    * unpAcc:
  
  Outputs:
    * auth_pdbSeq_mapping: list,author defined residue indices from start to end, e.g. ['-3','-2',.,'1','2',.'40','1000','1001',.,'1020','65','66',...]
    * aligned_regions: list of list, each list contains 4 values [pdb_begin,pdb_end, uniprot_begin, uniprot_end], e.g. [[5,232,1,228],[338,422,335,419]]
  """
  rcsb1d_url = "https://1d-coordinates.rcsb.org/graphql"
  pdb_instance = '{}.{}'.format(pdb_id,chain_id)
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
  res_idxMap = requests.post(rcsbBase_url,json={'query':query_idxMap})
  res_align = requests.post(rcsb1d_url,json={'query':query_align})