import json
from posixpath import split
from ssl import RAND_status
import time
from enum import Enum
from tkinter.font import families
from urllib.request import build_opener
import Bio.PDB,os,requests,sys,logging,lmdb
from matplotlib.pyplot import contour
from requests.adapters import HTTPAdapter, Retry
from prody import extendVector
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.Align import substitution_matrices
from Bio import SeqIO
import numpy as np
from json import JSONEncoder
import pickle as pkl
import pandas as pd
from typing import Sequence, List, Optional, Tuple, Union
from itertools import combinations_with_replacement
import multiprocessing as mp
from functools import partial



BLOSUM62_MAT = substitution_matrices.load("BLOSUM62")
# Default fill values for missing features
HSAAC_DIM = 42  # We have 2 + (2 * 20) HSAAC values from the two instances of the unknown residue symbol '-'
DEFAULT_MISSING_FEAT_VALUE = np.nan
DEFAULT_MISSING_SS = '-'
DEFAULT_MISSING_RSA = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_RSA_CLA = '-'
DEFAULT_MISSING_RD = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_PROTRUSION_INDEX = [DEFAULT_MISSING_FEAT_VALUE for _ in range(6)]
DEFAULT_MISSING_HSAAC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(HSAAC_DIM)]
DEFAULT_MISSING_CN = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_SEQUENCE_FEATS = np.array([DEFAULT_MISSING_FEAT_VALUE for _ in range(27)])
DEFAULT_MISSING_NORM_VEC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(3)]

# Abbreviations for SS
SS3_HELIX = 'H'
SS3_STRAND = 'E'
SS3_COIL = 'C'
# Abbrevistions for RSA
RSA2_BURIED = 'B'
RSA2_EXPOSE = 'E'

# SS8 to SS3
SS8_TO_SS3 = {'G':'H', 'I':'H', 'H':'H', 'B':'E', 'E':'E', 'T':'C', 'S':'C', '-':'-'}

# VDW radii values
#
# NOTE: 
#   - The first character in the PDB atom name is the element of the atom.
#
VDW_RADII = { "N" : 1.55, "C" : 1.70,"H" : 1.20, "O" : 1.52, "S": 1.85 }

# RSA cutoff, buried: 0 - BINARY_RSA_CUTOFF
BINARY_RSA_CUTOFF = 0.25

# mmCif struct_conf_type to SS3 mapping
struct_conf_type_to_ss3 = {
  'HELX_LH_27_P': SS3_HELIX, #left-handed 2-7 helix (protein)
  'HELX_LH_3T_P': SS3_HELIX, #left-handed 3-10 helix (protein)
  'HELX_LH_AL_P': SS3_HELIX, #left-handed alpha helix (protein)
  'HELX_LH_GA_P': SS3_HELIX, #left-handed gamma helix (protein)
  'HELX_LH_OM_P': SS3_HELIX, #left-handed omega helix (protein)
  'HELX_LH_OT_P': SS3_HELIX, #left-handed helix with type that does not conform to an accepted category (protein)
  'HELX_LH_P': SS3_HELIX, #left-handed helix with type not specified (protein)
  'HELX_LH_PI_P': SS3_HELIX, #left-handed pi helix (protein)
  'HELX_LH_PP_P': SS3_HELIX, #left-handed polyproline helix (protein)
  'HELX_OT_P': SS3_HELIX, #helix with handedness and type that do not conform to an accepted category (protein)
  'HELX_P': SS3_HELIX, #helix with handedness and type not specified (protein)
  'HELX_RH_27_P': SS3_HELIX, #right-handed 2-7 helix (protein)
  'HELX_RH_3T_P': SS3_HELIX, #right-handed 3-10 helix (protein)
  'HELX_RH_AL_P': SS3_HELIX, #right-handed alpha helix (protein)
  'HELX_RH_GA_P': SS3_HELIX, #right-handed gamma helix (protein)
  'HELX_RH_OM_P': SS3_HELIX, #right-handed omega helix (protein)
  'HELX_RH_OT_P': SS3_HELIX, #right-handed helix with type that does not conform to an accepted category (protein)
  'HELX_RH_P': SS3_HELIX, #right-handed helix with type not specified (protein)
  'HELX_RH_PI_P': SS3_HELIX, #right-handed pi helix (protein)
  'HELX_RH_PP_P': SS3_HELIX, #right-handed polyproline helix (protein)
  'STRN': SS3_STRAND, #beta strand (protein)
  'TURN_OT_P': SS3_COIL, #turn with type that does not conform to an accepted category (protein)
  'TURN_P': SS3_COIL, #turn with type not specified (protein)
  'TURN_TY1P_P': SS3_COIL, #type I prime turn (protein)
  'TURN_TY1_P': SS3_COIL, #type I turn (protein)
  'TURN_TY2P_P': SS3_COIL, #type II prime turn (protein)
  'TURN_TY2_P': SS3_COIL, #type II turn (protein)
  'TURN_TY3P_P': SS3_COIL, #type III prime turn (protein)
  'TURN_TY3_P': SS3_COIL, #type III turn (protein)
  'BEND': SS3_COIL, #region with high backbone curvature without specific hydrogen bonding, a bend at residue i occurs when the angle between $C\_alpha(i)-C_\alpha(i-2) and C_\alpha(i+2) - C_\alpha(i)$ is greater than 70 degrees (protein)
}



# Exception level Enum
class enum_exception_level(Enum):
  ERROR=1
  WARNING=2
  INFO=3

def print_exception(err: Exception,
                    level: Enum,
                    message: str = None):
  exception_message = str(err)
  exception_type, exception_object, exception_traceback = sys.exc_info()
  if level == enum_exception_level.ERROR:
    logger.error(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, {message}")
  elif level == enum_exception_level.WARNING:
    logger.warning(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, {message}")    
  elif level == enum_exception_level.INFO:
    logger.info(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, {message}")  

def read_ss_from_file(struct_file: str = None,
                      file_format: str = None):
  """Extract Second Structure(SS) information from structure file.

  Args:
    struct_file (str): full file path for input structure (file can be full file or only header part).
    file_format (str): structure file format, one of {pdb, mmCif}.
  
  Returns:
    dict<tuple,string>: tuple identifier (chain_id,residue_id) to SS element {'H','E','C'} dictionary.
      residue id use author_seq_id;
      Helix(H); Strand(E), Coil(C)
  
  Raises:

  """
  ss_dict = dict()
  if file_format == 'pdb':
    helix_words, sheet_words = '^HELIX', '^SHEET'
    #1-based
    helix_chain_idx = [20,20]
    sheet_chain_idx = [22,22]
    helix_beginId_idx, helix_endId_idx = [22,25], [34,37] 
    sheet_beginId_idx, sheet_endId_idx = [23,26], [34,37]

    try:
      ## grab ss lines
      helix_lines = os.popen(f"grep -E '{helix_words}' {struct_file}").read().strip('\n').split('\n')
      sheet_lines = os.popen(f"grep -E '{sheet_words}' {struct_file}").read().strip('\n').split('\n')

      assert len(helix_lines) > 0
      assert len(sheet_lines) > 0

      for line in helix_lines:
          ## split columns
          chain = line[helix_chain_idx[0]-1 : helix_chain_idx[1]]
          startId = int(line[helix_beginId_idx[0]-1 : helix_beginId_idx[1]])
          endId = int(line[helix_endId_idx[0]-1 : helix_endId_idx[1]])
          for idx in range(startId,endId+1):
            ss_dict[chain,str(idx)] = SS3_HELIX

      for line in sheet_lines:
          ## split columns
          chain = line[sheet_chain_idx[0]-1 : sheet_chain_idx[1]]
          startId = int(line[sheet_beginId_idx[0]-1 : sheet_beginId_idx[1]])
          endId = int(line[sheet_endId_idx[0]-1 : sheet_endId_idx[1]])
          for idx in range(startId,endId+1):
            ss_dict[chain,str(idx)] = SS3_STRAND
            
    except Exception as ex:
      pdb_id, tmp = os.path.splitext(os.path.basename(struct_file))
      print_exception(err=ex,level=enum_exception_level.WARNING,message=f'{struct_file}')

  elif file_format == 'mmCif':
    try:
      mmcif_header=Bio.PDB.MMCIF2Dict.MMCIF2Dict(struct_file)
      # helix, strand, turn
      for r in range(len(mmcif_header['_struct_conf.conf_type_id'])):
        type_id = mmcif_header['_struct_conf.conf_type_id'][r]
        chain = mmcif_header['_struct_conf.beg_auth_asym_id'][r]
        startId = int(mmcif_header['_struct_conf.beg_auth_seq_id'][r])
        endId = int(mmcif_header['_struct_conf.end_auth_seq_id'][r])
        for idx in range(startId,endId+1):
          ss_dict[chain,str(idx)] = struct_conf_type_to_ss3[type_id]
      # strand (extra information for double check)
      if '_struct_sheet_range.sheet_id' in mmcif_header.keys():
        for r in range(len(mmcif_header['_struct_sheet_range.sheet_id'])):
          chain = mmcif_header['_struct_sheet_range.beg_auth_asym_id'][r]
          startId = int(mmcif_header['_struct_sheet_range.beg_auth_seq_id'][r])
          endId = int(mmcif_header['_struct_sheet_range.end_auth_seq_id'][r])
          for idx in range(startId,endId+1):
            if (chain,str(idx)) not in ss_dict.keys():
              ss_dict[chain,str(idx)] = SS3_STRAND
    except Exception as ex:
      pdb_id, tmp = os.path.splitext(os.path.basename(struct_file))
      print_exception(err=ex,level=enum_exception_level.WARNING,message=f'{struct_file}')

  return ss_dict

def read_conPdbSeq_from_mmcif(pdb_file: str,
                              entity_id: int):
  """Read canonical pdb sequence from mmcif file

  Args:
    pdb_file (str): pdb file path

  Returns:
    str: canonical pdb sequence

  Raises:
    KeyError: '_entity_poly.pdbx_seq_one_letter_code_can' section not included in header part.
  """
  con_pdbSeq = None
  try:
    pdb_id, file_format = os.path.splitext(os.path.basename(pdb_file))
    mmcif_header=Bio.PDB.MMCIF2Dict.MMCIF2Dict(pdb_file)
    con_pdbSeq = mmcif_header['_entity_poly.pdbx_seq_one_letter_code_can'][entity_id-1].replace('\n','')
  except Exception as err:
    print_exception(err=err, level=enum_exception_level.ERROR, message=f'pdb seq error: {pdb_id}')
  return con_pdbSeq


def get_dssp_outputs(struct_file: str,file_format: str):
  """Get DSSP output dict for input structure file

  Args:
    struct_file (str): structure file path

  Returns:
    dict: DSSP output, when exception happens, return None
      DSSP ouput is a dict of key-value pairs. Key is a tuple: (chain_id, (' ', res_id, ' ')). 'chain_id' and 'res_id' are author defined.

      value is feature list: 
        [dssp index, amino acid, secondary structure, relative ASA, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy]

  Raises:

  """
  pdb_id, ext = os.path.splitext(os.path.basename(struct_file))
  pdb_id = pdb_id.lower()
  try:
    if file_format.lower() == 'pdb':
      parser = Bio.PDB.PDBParser(QUIET=True)
      struct = parser.get_structure(pdb_id, struct_file)
      model = struct[0]                    
    elif file_format.lower() == 'mmcif':
      parser = Bio.PDB.FastMMCIFParser(QUIET=True)
      struct = parser.get_structure(pdb_id, struct_file)
      model = struct[0]                    
    
    #dssp_tuple = dssp_dict_from_pdb_file(struct_file,DSSP='mkdssp')
    dssp = Bio.PDB.DSSP(model, struct_file, dssp='mkdssp', acc_array='Sander', file_type=file_format.upper())

    return dssp
  
  except Exception as err:
    print_exception(err=err,level=enum_exception_level.ERROR)
    return None

def get_dssp_ss_residue(dssp_dict: dict,
                        chain: str,
                        residue_idx: str):
  dssp_value = DEFAULT_MISSING_SS
  try:
    if residue_idx.isnumeric():
      dssp_values = dssp_dict[chain, int(residue_idx)]
    else:
      pass
      #dssp_values = dssp_dict[chain, (' ', int(residue_idx), ' ')]
    dssp_value = dssp_values[2]
  except KeyError:
    print_exception(err=KeyError,level=enum_exception_level.WARNING,message=f"residue id key ({chain}, (' ', {residue_idx}, ' ')) not exists")
  except Exception as err:
    print_exception(err=err,level=enum_exception_level.WARNING)
  return dssp_value

def get_dssp_rsa_residue(dssp_dict: dict,
                        chain: str,
                        residue_idx: str,
                        thres: float = 0.25):
  """RSA from dssp calculation
  classification criteria from 'Comprehensive characterization of amino acid positions in protein structures reveals molecular effect of missense variants':
  five types: (i) core (RSA < 5%); (ii) buried (5% <= RSA < 25%); (iii) medium-buried (25% <= RSA < 50%); (iv) medium-exposed (50% <= RSA < 75%); (v) exposed (RSA >= 75%).

  2 classes: use 25% as cutoff

  Args:
    dssp_dict (dict): DSSP output in dictionary format.
    chain (str): author defined chain id.
    residue_idx (str): author defined residue index.
    thres (float): cutoff for 2 classes, default: 0.25.

  Returns:
    float: RSA value for residue
    str: RSA class for residue, one of ['B, 'E', '-']
  """
  dssp_value = DEFAULT_MISSING_RSA
  dssp_class = DEFAULT_MISSING_RSA_CLA
  try:
    if residue_idx.isnumeric():
      dssp_values = dssp_dict[chain, int(residue_idx)]
      if isinstance(dssp_values[3], float):
        dssp_value = dssp_values[3] 
        dssp_class = RSA2_BURIED if dssp_values[3] < thres else RSA2_EXPOSE
    else:
      pass
      #dssp_values = dssp_dict[chain, (' ', int(residue_idx), ' ')]
  except KeyError:
    print_exception(err=KeyError,level=enum_exception_level.WARNING,message=f"residue id key ({chain}, (' ', {residue_idx}, ' ')) not exists")
  except Exception as err:
    print_exception(err=KeyError,level=enum_exception_level.WARNING)
  return dssp_value, dssp_class

def get_dssp_rsa_instance(dssp_dict: dict,
                          struct_model: Bio.PDB.Chain.Chain):
  
  # loop over residues
  # for resi in struct_model:
  #   try:
      
  return 0

# def get_residueDepth(struct_file: str = None,
#             chain_ref: str = None,
#             file_format: str = None,
#             exception_file: str = None,
#             set_nm: str = None):

# def get_protrusionIndex(struct_file: str = None,
#             chain_ref: str = None,
#             file_format: str = None,
#             exception_file: str = None,
#             set_nm: str = None):

def get_index_mapping(
        pdb_id: str = None,
        chain_id: str = None,
        unpAcc: str = None,
        logger: logging.Logger = None):
  """
  Modified based on function pdbmap_processs.queryApi_pdbInfo

  Use RSCB API to get index mapping between
  * uniprot residue index (continuous, need two value: begin_idx, end_idx)
  * PDB residue index (continuous, need two value: begin_idx, end_idx)
  * author defined PDB residue index(not continuous; need a list of numbers)

  For insertion positions in pdb seq, use -1 as index in uniprot row
  
  
  API returns author defined indices for the whole sequence, the positions covered by uniprot sequence are extracted as return
  
  Args:
    chain_id (str): _label_asym_id in PDBx/mmCIF schema (not auth chain id!)
  
  Returns:
    List: index mapping between three sets, size 3*L
      1st row: uniprot indices of each amino acid from N-ter to c_ter
      2nd row: PDB sequence indices of each amino acid from N-ter to c_ter
      3rd row: author defined PDB sequence indices of each amino acid from N-ter to c_ter
    str: uniprot sequence
    str: pdb sequence
    str: auth chain id
    str: entity id

  """
  pdb_id, chain_id, unpAcc = pdb_id.upper(), chain_id.upper(), unpAcc.upper()
  rcsbBase_url = "https://data.rcsb.org/graphql"
  rcsb1d_url = "https://1d-coordinates.rcsb.org/graphql"
  pdb_instance = '{}.{}'.format(pdb_id.upper(),chain_id.upper())
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
  
  threeIdxSetMap = []
  
  ## pdb-uniprot idx mapping
  try:
    unp_seq,pdb_seq,aligned_regions = None, None, None
    res_align = req_sess.post(rcsb1d_url,json={'query':query_align})
    res_align_json = res_align.json()
    pdb_seq=res_align_json['data']['alignment']['query_sequence']
    # one pdb seq could have more than 1 unp correspondence
    for d in res_align_json['data']['alignment']['target_alignment']:
      if d['target_id'] == unpAcc.upper():
        unp_seq=d['target_sequence']
        aligned_regions=d['aligned_regions']
    if unp_seq is None: 
      # no such unpAcc under this pdb,
      #print(f"WARNING: {pdb_instance} has no matching seq for {unpAcc}", file=sys.stderr)
      logger.warning(f"{pdb_instance} has no matching seq for {unpAcc}")
    # loop over aligned regions
    pdb_idxs = []
    unp_idxs = [] 
    if aligned_regions is not None:
      for ali_reg in aligned_regions:
        pdb_idxs.extend([str(tmpi) for tmpi in range(ali_reg['query_begin'],ali_reg['query_end']+1)])
        unp_idxs.extend([str(tmpi) for tmpi in range(ali_reg['target_begin'],ali_reg['target_end']+1)])
    ## insert -1 for pdb insertion positions
    new_pdb_idxs = []
    new_unp_idxs = []
    for pdb_i in range(int(pdb_idxs[0]),int(pdb_idxs[-1])+1):
      new_pdb_idxs.append(str(pdb_i))
      curr_i = np.where(np.array(pdb_idxs)==str(pdb_i))[0]
      if len(curr_i) > 0:
        new_unp_idxs.append(unp_idxs[curr_i[0]])
      else:
        new_unp_idxs.append('-1')

    threeIdxSetMap.append(new_unp_idxs)
    threeIdxSetMap.append(new_pdb_idxs)
    assert len(new_unp_idxs) == len(new_pdb_idxs)
    ## author defined pdb idx - pdb idx
    res_idxMap = req_sess.post(rcsbBase_url,json={'query':query_idxMap})
    res_idxMap_json = res_idxMap.json()
    auth_pdbSeq_mapping=res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_to_entity_poly_seq_mapping']
    auth_pdbSeq = [auth_pdbSeq_mapping[int(pdb_i)-1] for pdb_i in new_pdb_idxs] #insert case: e.g. '1A'
    assert len(new_unp_idxs) == len(auth_pdbSeq)
    threeIdxSetMap.append(auth_pdbSeq)

    auth_chain_id = res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id']
    entity_id = res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['entity_id']
    
    return threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id
  except KeyError as err:
    exception_message = str(err)
    exception_type, exception_object, exception_traceback = sys.exc_info()
    logger.warning(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, pdb:{pdb_id}_{chain_id}")
  except Exception as err:
    exception_message = str(err)
    exception_type, exception_object, exception_traceback = sys.exc_info()
    logger.warning(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, pdb:{pdb_id}_{chain_id}")
    
  return threeIdxSetMap, None, None, None, None

def get_residues(pdb_fn, pdb_id, chain_ids=None, model_num=0):
    """Build a simple list of residues from a single chain of a PDB file.
    Args:
        pdb_fn: The path to a PDB file.
        chain_ids: A list of single-character chain identifiers.
        dssp: Path to the dssp executable (optional).
        model_num: The model number in the PDB file to use (optional)
    Returns:
        A list of Bio.PDB.Residue objects.
    """

    if pdb_fn is not None:
      pdb_id = os.path.splitext(os.path.basename(pdb_fn))[0]
    
    bio_pdbList = Bio.PDB.PDBList()
    pdb_flNm = bio_pdbList.retrieve_pdb_file(pdb_id,pdir=pdb_download_dir,file_format='mmCif',overwrite=True)
    parser = Bio.PDB.FastMMCIFParser(QUIET=True)
    struct = parser.get_structure(pdb_id, pdb_flNm)
    model = struct[0]
       

    if chain_ids is None:
        # get residues from every chain.
        chains = model.get_list()
    else:
        chains = [ model[ch_id] for ch_id in chain_ids ]

    residues = []

    # To do, handle:
    # - disordered residues/atoms (todo)
    # - non-amino acids
    # - non-standard amino acids
    for ch in chains:
        for res in filter(lambda r : Bio.PDB.is_aa(r), ch.get_residues()):
            if not Bio.PDB.is_aa(res, standard=True):
                sys.stderr.write("WARNING: non-standard AA at %r%s"
                        % (res.get_id(), os.linesep))
            residues.append(res)

    return residues, model

def get_atom_coord(res, atom_name, verbose=False):
    """Get the atomic coordinate of a single atom in a residue. This function wraps the
    ``Bio.PDB.Residue.get_coord()`` function to infer CB coordinates if required.
    Args:
        res: A Bio.PDB.Residue object.
        atom_name: The name of the atom (e.g. "CA" for alpha-carbon).
        verbose: Display diagnostic messages to stderr (optional).
    Returns:
        The coordinate of the specified atom.
    """

    try:
        coord = res[atom_name].get_coord()
    except KeyError:
        if atom_name != "CB":
            # Return the first/only available atom.
            atom = res.child_dict.values()[0]
            sys.stderr.write(
                "WARNING: {} atom not found in {}".format(atom_name, res)
                    + os.linesep
            )
            return atom.get_coord()

        if verbose:
            sys.stderr.write(
                "WARNING: computing virtual {} coordinate.".format(atom_name)
                    + os.linesep)

        assert("N" in res)
        assert("CA" in res)
        assert("C" in res)

        # Infer the CB atom position described in http://goo.gl/OaNjxe
        #
        # NOTE:
        # These are Bio.PDB.Vector objects and _not_ numpy arrays.
        N = res["N"].get_vector()
        CA = res["CA"].get_vector()
        C = res["C"].get_vector()

        CA_N = N - CA
        CA_C = C - CA

        rot_mat = Bio.PDB.rotaxis(-np.pi * 120.0 / 180.0, CA_C)

        coord = (CA + CA_N.left_multiply(rot_mat)).get_array()

    return coord

def get_sidechain_atoms(residue: Residue):
  return None

def calc_center_of_mass(atom: Atom):
  return None

def calc_cmass_distance(res_a: Residue, res_b: Residue, sidechain_only: bool = False):
    """Compute the distance between the centres of mass of both residues.
    Args:
        res_a: A ``Bio.PDB.Residue`` object.
        res_b: A ``Bio.PDB.Residue`` object.
        sidechain_only: Set to True to consider only the sidechain atoms, False
            otherwise (optional).
    Returns:
        The distance between the centers of mass of ``res_a`` and ``res_b``.
    """

    if sidechain_only:
        atoms_a = get_sidechain_atoms(res_a)
        atoms_b = get_sidechain_atoms(res_b)
    else:
        atoms_a = res_a.get_list()
        atoms_b = res_b.get_list()

    A = calc_center_of_mass(atoms_a)
    B = calc_center_of_mass(atoms_b)

    return np.linalg.norm(A-B)

def calc_minvdw_distance(res_a, res_b):
    """Compute the minimum VDW distance between two residues, accounting for the
    VDW radii of each atom.
    Args:
        res_a: A ``Bio.PDB.Residue`` object.
        res_b: A ``Bio.PDB.Residue`` object.
    Returns:
        The minimum VDW distance between ``res_a`` and ``res_b``.
    """

    min_dist = None

    for a in res_a.get_iterator():
        for b in res_b.get_iterator():
            radii_a = VDW_RADII.get(a.get_id()[0], 0.0)
            radii_b = VDW_RADII.get(b.get_id()[0], 0.0)

            A = a.get_coord()
            B = b.get_coord()

            dist = np.linalg.norm(A - B) - radii_a - radii_b

            if (min_dist is None) or dist < min_dist:
                min_dist = dist

    return min_dist

def calc_distance(res_a, res_b, measure="CA"):
    """Calculate the (L2) Euclidean distance between a pair of residues
    according to a given distance metric.
    Args:
        res_a: A ``Bio.PDB.Residue`` object.
        res_b: A ``Bio.PDB.Residue`` object.
        measure: The inter-residue distance measure (optional).
    Returns:
        The distance between ``res_a`` and ``res_b``.
    """

    if measure in ("CA", "CB"):
        A = get_atom_coord(res_a, measure)
        B = get_atom_coord(res_b, measure)
        dist = np.linalg.norm(A-B)
    elif measure == "cmass":
        dist = calc_cmass_distance(res_a, res_b)
    elif measure == "sccmass":
        dist = calc_cmass_distance(res_a, res_b, sidechain_only=True)
    elif measure == "minvdw":
        dist = calc_minvdw_distance(res_a, res_b)
    else:
        raise NotImplementedError

    return dist

def distMap2class(distance_map: Union[np.ma.masked_array,np.ndarray]):
  """bin 2D distance map
  
  Args:
    distance_map (ndarray or masked_array): 2D distance map

  Returns:
    ndarray or masked_array: binned 2D distance map
  """
  PARAMS = {
    "DMIN" : 2.0,
    "DMAX" : 20.0,
    "DBINS": 36}

  # dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
  # db = torch.round((dist-params['DMIN']-dstep/2)/dstep)

  # db[db<0] = 0
  # db[db>params['DBINS']] = params['DBINS']
    
  # return db.long()  


def singleProtein_distMat(pdb_model: Bio.PDB.Chain.Chain,
                          threeIdxSetMap: List = None,
                          repr_atom: str = 'CB',
                          contact_thresh: float = 8.0,
                          symmetric: bool = True):
  """Generate distance/contact map for single protein structure.
  
  Missing residue will be reflected by residue numbering in BioPython, e.g. two continuous residue objects of 6VYB.A: <Residue HIS het=  resseq=69 icode= >, <Residue PRO het=  resseq=82 icode= >

  Args:
    pdb_model (biopython.PDB.Model): Biopython object for pdb strucuture.
    threeIdxSetMap (List): Index mapping between 1-uniprot indices, 2-pdb sequence, 3-author defined pdb sequence. Size 3*L.
    repr_atom (str): Representative (pseduo)atom for each amino acid. Defaults to 'CA'.
    contact_thresh (float): Threshold for converting distance map to contact map.
      Distance below contact_thresh are regarded as contact (True in contact map). Defaults to 10 \AA.
    symmetric (bool): If True, return a symmetric distance map.
  
  Returns:
    numpy.masked_array: distance map
    numpy.masked_array: contact map if contact_thresh is given, otherwise None.
  """
  threeIdxSetMap = np.array(threeIdxSetMap)
  seq_len = threeIdxSetMap.shape[1]
  dist_map = np.zeros((seq_len, seq_len), dtype="float64")
  dist_map[:] = np.nan
  contact_map = None
  
  ## index pair to upper-triangle
  pair_indices = combinations_with_replacement(range(seq_len), 2)
  for i, j in pair_indices:
    try:
      res_a = pdb_model[int(threeIdxSetMap[2,i])]
      res_b = pdb_model[int(threeIdxSetMap[2,j])]
      dist = calc_distance(res_a, res_b, repr_atom)
      dist_map[i,j] = dist
      if symmetric:
        dist_map[j,i] = dist
    except KeyError:
      #print_exception(err=KeyError, level=enum_exception_level.ERROR, message=f'res_a key: {threeIdxSetMap[2,i]}, res_b key: {int(threeIdxSetMap[2,j])}')
      pass
    except Exception as ex:
      print_exception(err=ex, level=enum_exception_level.ERROR)
  
  dist_map = np.ma.masked_array(dist_map, np.isnan(dist_map),fill_value=np.nan)
  ## convert to contact map
  if contact_thresh is not None:
        contact_map = dist_map < contact_thresh

  return dist_map, contact_map

def test_singleProtein_distMat():
  # logging.basicConfig(filename='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/dataProcess_scripts/exception.out', level=logging.DEBUG, 
  #                   format='%(asctime)s %(levelname)s %(name)s %(message)s')
  # logger=logging.getLogger(__name__)
  bio_parser = Bio.PDB.FastMMCIFParser(QUIET=True)
  pdb_struct = bio_parser.get_structure('2plq','/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/download_structures/AMIE_PSEAE/2plq.cif')
  pdb_model = pdb_struct[0]['A']
  threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id = get_index_mapping(pdb_id='2plq',chain_id='A',unpAcc='Q9L543')

  dist_map, contact_map = singleProtein_distMat(pdb_model=pdb_model,threeIdxSetMap=threeIdxSetMap,repr_atom='CA',contact_thresh=10.0)
  return dist_map,contact_map

def queryApi_pdbInfo(pdb_id,chain_id,unpAcc):
  """
  query information from RCSB PDB Data API & RCSB PDB 1D Coordinate Server API
  * Residue index mappings between author provided and pdb sequence positions
  * Sequence alignment between seq of uniprot and pdb
  * Unmodelled regions(residues;atoms)
  Input
  * pdb_id
  * chain_id(asym_id, not author defined)
  * unpAcc: uniprot accesion
  Output
  * hasRes: True - has response; False - no response
  * auth_pdbSeq_mapping: list,author defined residue indices from start to end, e.g. ['-3','-2',.,'1','2',.'40','1000','1001',.,'1020','65','66',...]
  * unp_seq: str,uniprot seq
  * pdb_seq: str,pdb seq
  * aligned_regions: list of dict,each dict has keys "query_begin","query_end","target_begin","target_end"(pdb residue index)
  * unobserved_residues: list of dict, each dict has keys "beg_seq_id","end_seq_id" (pdb residue index)
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
  res_test = req_sess.post(rcsbBase_url,json={'query':query_test})
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
  res_idxMap = req_sess.post(rcsbBase_url,json={'query':query_idxMap})
  res_align = req_sess.post(rcsb1d_url,json={'query':query_align})
  res_unmodel = req_sess.post(rcsb1d_url,json={'query':query_unmodel}) 
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

def unp_repr_pdbs(unp_id: str,
                  seqIdentity_percent: int = 100,
                  represent_by: str = 'score',
                  logger:logging.Logger = None) -> List:
  """
  use RSCB Search API to query representative/non-redundant pdb polymer entities for one uniprot id

  Args:
    - unp_id: uniprot accession number
    - seqIdentity_precent: sequence identity cutoff in percentage, 100(default)
    - represent_by: criteria to select representative one:
        entity_poly.rcsb_sample_sequence_length;
        rcsb_accession_info.initial_release_date;
        rcsb_entry_info.resolution_combined;
        score(default);
  
  Returns:
    - pdb_list: representative/non-redundant pdb list for input unp_id
    - total_num: total number of polymers for input unp_id
    - repr_num: number of polymers after sequence identity filtering for input unp_id

  """
  query_url = 'https://search.rcsb.org/rcsbsearch/v1/query'
  query_str = '''{{"query": {{
    "type": "group",
    "nodes": [
      {{
        "type": "terminal",
        "service": "text",
        "parameters": {{
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
          "operator": "in",
          "negation": false,
          "value": [
            "{unp_id}"
          ]
        }}
      }},
      {{
        "type": "terminal",
        "service": "text",
        "parameters": {{
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
          "operator": "exact_match",
          "value": "UniProt",
          "negation": false
        }}
      }}
    ],
    "logical_operator": "and",
    "label": "nested-attribute"
  }},
  "return_type": "polymer_entity",
  "request_options": {{
    "group_by_return_type": "representatives",
    "group_by": {{
      "aggregation_method": "sequence_identity",
      "ranking_criteria_type": {{
        "sort_by": "{ranking}",
        "direction": "desc"
      }},
      "similarity_cutoff": {iden_cutoff}
    }},
    "pager": {{
      "start": 0,
      "rows": 1000000
    }},
    "sort": [
      {{
        "sort_by": "{ranking}",
        "direction": "desc"
      }}
    ],
    "scoring_strategy": "combined"
    }}
  }}'''.format(unp_id=unp_id,iden_cutoff=seqIdentity_percent,ranking=represent_by)
 
  polymer_entity_list = []
  total_num, repr_num = 0,0
  ## launch query
  try:
    repr_pdbs = req_sess.post(query_url,data=query_str)
    if repr_pdbs.status_code == 200:
      logger.info(f"[unp_repr_pdbs] unp_id:{unp_id}, response okay")
    else:
      return polymer_entity_list, total_num, repr_num 
    
    repr_pbds_json = repr_pdbs.json()
    total_num = repr_pbds_json['total_count'] 
    repr_num = repr_pbds_json['group_by_count']
    result_set_list = repr_pbds_json['result_set']
    # loop result list
    for res in result_set_list:
      polymer_entity_list.append(res['identifier'])
  except KeyError as err:
    exception_message = str(err)
    exception_type, exception_object, exception_traceback = sys.exc_info()
    logger.warning(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, unp_id:{unp_id}")
  except Exception as err:
    exception_message = str(err)
    exception_type, exception_object, exception_traceback = sys.exc_info()
    logger.warning(f"Line {exception_traceback.tb_lineno}, {exception_type}:{exception_message}, unp_id:{unp_id}")
  return polymer_entity_list, total_num, repr_num


def unp_batch_repr_pdbs(unpId_list: List,
                        seqIdentity_percent: int = 100,
                        represent_by: str = 'entity_poly.rcsb_sample_sequence_length',
                        logger:logging.Logger = None) -> List:
  """
  use RSCB Search API to query representative/non-redundant pdb polymer entities for one uniprot id

  Args:
    - unp_id: uniprot accession number
    - seqIdentity_precent: sequence identity cutoff in percentage, 100(default)
    - represent_by: criteria to select representative one:
        entity_poly.rcsb_sample_sequence_length; (default)
        rcsb_accession_info.initial_release_date;
        rcsb_entry_info.resolution_combined;
        score;
  
  Returns:
    - pdb_list: representative/non-redundant pdb list for input unp_id
    - total_num: total number of polymers for input unp_id
    - repr_num: number of polymers after sequence identity filtering for input unp_id

  """
  unp_str = ''
  for i in range(len(unpId_list)-1):
    unp_str += f'"{unpId_list[i]}",'
  unp_str += f'"{unpId_list[-1]}"'
  ## https://search.rcsb.org/#search-example-8
  query_url = 'https://search.rcsb.org/rcsbsearch/v2/query'
  query_str = '''{{"query": {{
    "type": "group",
    "nodes": [
      {{
        "type": "terminal",
        "service": "text",
        "parameters": {{
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
          "operator": "in",
          "negation": false,
          "value": [
            {unp_ids}
          ]
        }}
      }},
      {{
        "type": "terminal",
        "service": "text",
        "parameters": {{
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
          "operator": "exact_match",
          "value": "UniProt",
          "negation": false
        }}
      }}
    ],
    "logical_operator": "and",
    "label": "nested-attribute"
  }},
  "return_type": "polymer_entity",
  "request_options": {{
    "group_by_return_type": "representatives",
    "group_by": {{
      "aggregation_method": "sequence_identity",
      "ranking_criteria_type": {{
        "sort_by": "{ranking}",
        "direction": "desc"
      }},
      "similarity_cutoff": {iden_cutoff}
    }},
    "paginate": {{
      "start": 0,
      "rows": 1000000
    }},
    "sort": [
      {{
        "sort_by": "{ranking}",
        "direction": "desc"
      }}
    ],
    "scoring_strategy": "combined"
    }}
  }}'''.format(unp_ids=unp_str,iden_cutoff=seqIdentity_percent,ranking=represent_by)
 
  polymer_entity_list = []
  total_num, repr_num = 0,0
  ## launch query
  try:
    repr_pdbs = req_sess.post(query_url,data=query_str)
    if repr_pdbs.status_code == 200:
      logger.info(f"[unp_repr_pdbs] unpId batch from {unpId_list[0]} to {unpId_list[-1]}, response okay")
    else:
      return polymer_entity_list, total_num, repr_num 
    
    repr_pbds_json = repr_pdbs.json()
    total_num = repr_pbds_json['total_count'] 
    repr_num = repr_pbds_json['group_by_count']
    result_set_list = repr_pbds_json['result_set']
    # loop result list
    for res in result_set_list:
      polymer_entity_list.append(res['identifier'])
  except KeyError as err:
    print_exception(err=err,level=enum_exception_level.WARNING)
  except Exception as err:
    print_exception(err=err,level=enum_exception_level.WARNING)
  return polymer_entity_list, total_num, repr_num


def select_entity_instance(entity_id:str,
                           filter_type:str = 'UNOBSERVED_RESIDUE_XYZ'):
  """One polymer entity usually have more than one instances. This function is to select one out using filter_type

  Args:
    entity_id (str): pdb polymer entity id
    filter_type (str): the instance feature type to select the top instance, default: "UNOBSERVED_RESIDUE_XYZ"

  Return:
    str: selected top one instance id, e.g. 1a2b.A
    List[str]: all instance ids
    str: uniprot id to this entity
  """
  query_url = "https://data.rcsb.org/graphql"
  query_str = '''
  {{polymer_entities(entity_ids: ["{entity}"]) {{
      rcsb_polymer_entity_container_identifiers {{
        asym_ids
        reference_sequence_identifiers {{
          database_accession
          database_isoform
          database_name
        }}
      }}
    }}
  }}'''.format(entity=entity_id)

  
  query_out = req_sess.post(query_url,json={'query':query_str})
  queryOut_json = query_out.json()
  pdb_id = entity_id.split('_')[0]
  try:
    entity_info = queryOut_json['data']['polymer_entities'][0]
    instanceId_list = entity_info['rcsb_polymer_entity_container_identifiers']['asym_ids']
    seqIden_list = entity_info['rcsb_polymer_entity_container_identifiers']['reference_sequence_identifiers']
    ## get unp id
    unpAcc_id = None
    for seq_iden in seqIden_list:
      db_name = seq_iden['database_name']
      db_accession = seq_iden['database_accession']
      if db_name.lower() == 'uniprot':
        unpAcc_id = db_accession        

    if len(instanceId_list) == 1:
      return f'{pdb_id}.{instanceId_list[0]}', [f'{pdb_id}.{instanceId_list[0]}'], unpAcc_id
    else:
      insId_list = []
      for insId in instanceId_list:
        insId_list.append(f"\"{pdb_id}.{insId}\"")
      insId_str = ','.join(insId_list)
      # query features 
      query_feature_str = '''
        {{polymer_entity_instances(instance_ids: [{insIds}]) {{
            rcsb_id
            rcsb_polymer_instance_feature_summary{{
              type,
              count,        
              coverage
            }}
          }}
        }}'''.format(insIds=insId_str)
      query_out = req_sess.post(query_url,json={'query':query_feature_str})
      queryOut_json = query_out.json()
      instance_feaList = queryOut_json['data']['polymer_entity_instances']
      top_insId = None
      top_coverage = 1.
      for fea_set in instance_feaList:
        ins_id = fea_set['rcsb_id']
        for fea in fea_set['rcsb_polymer_instance_feature_summary']:
          if fea['type'] == filter_type:
            fea_coverage = fea['coverage']
        if fea_coverage < top_coverage:
          top_insId = ins_id
          top_coverage = fea_coverage
    return top_insId, [f'{pdb_id}.{insId}' for insId in instanceId_list], unpAcc_id
  except Exception as err:
    print_exception(err=err,level=enum_exception_level.WARNING,message=f'entity_id:{entity_id}')
  return None, [], None

def query_struct_ShinData(
  set_list: List,
  set_file: str,
  unpId_batch_size: int,
  afDB_version: int,
  afAccessionPath: str,
  download_struct: bool = False,
  struct_format: str = 'mmCif',
  check_exp_struct: bool = False,
  check_AF_struct: bool = False,
  unp_pdb_map_path: str = None,
  logger: logging.Logger = None):
  """Collect structures for sequence set
  """
  ## read family list to process
  if set_list is None:
      set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
 
  if afAccessionPath.split('.')[-1] == 'pickle':
    with open(f'{afAccessionPath}', 'rb') as handle:
      afAccessionDict = pkl.load(handle)
  elif afAccessionPath.split('.')[-1] == 'csv':
    afAcc_map = pd.read_csv(afAccessionPath,sep=',',header=None,names=['unpAcc','start','end','afAcc','version'])

  # with open(f'{uniprotMap_path}', 'rb') as handle:
  #   uniprotDict = pkl.load(handle)
  # print('Finish loading mappings!')

  bio_pdbList = Bio.PDB.PDBList(verbose=False)

  ## loop over family list
  for setI in range(len(set_list)):
    setNm = set_list[setI]
    ## list to hold pdb ids to download
    pdbId_to_download = []
    afPdb_to_download = []
    ## save uniprot id, experiment pdb id mapping
    if unp_pdb_map_path is not None:
      if check_exp_struct:
        unp_expPdb_write = open(f'{unp_pdb_map_path}/{setNm}_unp_expPDB_map.tsv','w')
      if check_AF_struct:
        unp_afPdb_write = open(f'{unp_pdb_map_path}/{setNm}_unp_afPDB_map.tsv','w')
    
    ## statistic variables
    expStruct_numSeq = 0
    expStruct_numEntityAll = 0
    expStruct_numEntity100 = 0
    expStruct_numEntity100InstanceAll = 0
    expStruct_numEntity100InstanceReprRangeCover = []
    expStruct_num = 0
    af2Struct_num = 0
    
    ## load fasta seqs of Shin2021 Data
    fasta_file = os.popen(f"grep '{setNm}' /scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list | cut -f1").read().strip('\n')
    #fasta_file = set_list[setI,0]
    with open(f'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequences/{fasta_file}.fa') as handle:
      famSeqNum = 0
      unpId_list = []
      unpId_dict = {}
      unpId_key_key_mapping = {}
      for record in SeqIO.parse(handle,"fasta"):
        famSeqNum += 1
        if famSeqNum % 1000 == 0:
          print(f"set {setNm}, at seq #: {famSeqNum}",flush=True)
        seq_str = record.seq
        seqId_split = record.id.split(':')
        reweight_score = seqId_split[1]
        seq_range = seqId_split[0].split('/')[1]
        seq_range_start, seq_range_end = seq_range.split('-')
        ## get uniprot id
        if 'UniRef100' not in record.id: # AMIE_PSEAE/1-346:0.0714285714286
            unpId = os.popen(f"grep '{setNm}' /scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list | cut -f4").read().strip('\n')
        else: # UniRef100_Q7URG9/19-317:0.125
            unpId = seqId_split[0].split('/')[0].split('_')[1]
        unpId_list.append(f'{unpId}')
        unpId_range_key =f'{unpId}/{seq_range_start}-{seq_range_end}'
        if unpId_range_key not in unpId_dict.keys():
          unpId_dict[unpId_range_key] = {
            'seq_range_start': seq_range_start,
            'seq_range_end': seq_range_end,
            'weight_score': reweight_score,
            'unp_seq_segment': seq_str,
            'record_id': record.id}
        
        if unpId not in unpId_key_key_mapping.keys():
          unpId_key_key_mapping[unpId] = [unpId_range_key]
        else:
          unpId_key_key_mapping[unpId].append(unpId_range_key)
    
    tail_idx = 0
    case_with_exp_stucture = {}
    while tail_idx < len(unpId_list):
      expPDB_list, afPbdRecord_list= [], []
      current_list = unpId_list[tail_idx:tail_idx+unpId_batch_size]
      tail_idx += unpId_batch_size
      if check_exp_struct:
        ### rscb API batch query ###
        expPDB_list, total_num, repr_num = unp_batch_repr_pdbs(unpId_list=current_list,seqIdentity_percent=100,represent_by='entity_poly.rcsb_sample_sequence_length',logger=logger)
        #end = time.time()
        #print("The time of execution of above program is :", end-start)
        expStruct_numEntityAll += total_num
        expStruct_numEntity100 += repr_num

        ## loop over polymer entities
        for exp_entityId in expPDB_list:
          try:
            if len(exp_entityId) == 0:
              continue
            #pdbId, pdbChainId = exp_entityId.split(':')
            pdbId, pdbEntityId = exp_entityId.split('_')
            
            ## get top-1 polymer entity instance among all instances
            top_insId, all_indId, unp_acc = select_entity_instance(entity_id=exp_entityId)
            expStruct_numEntity100InstanceAll += len(all_indId)  
            ins_chainId = top_insId.split('.')[-1]
            
            ## check if the top instance cover segment range
            threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id = get_index_mapping(pdb_id=pdbId,chain_id=ins_chainId,unpAcc=unp_acc,logger=logger)
            real_unp_pos = np.where(np.array(threeIdxSetMap[0]).astype(int) > 0)[0]

            if unp_acc in unpId_key_key_mapping.keys():
              for unpId_key in unpId_key_key_mapping[unp_acc]:
                start_in_range_local = (int(threeIdxSetMap[0][real_unp_pos[0]]) <= int(unpId_dict[unpId_key]['seq_range_start'])) and (int(threeIdxSetMap[0][real_unp_pos[-1]]) >= int(unpId_dict[unpId_key]['seq_range_start']))
                end_in_range_local = (int(threeIdxSetMap[0][real_unp_pos[0]]) <= int(unpId_dict[unpId_key]['seq_range_end'])) and (int(threeIdxSetMap[0][real_unp_pos[-1]]) >= int(unpId_dict[unpId_key]['seq_range_end']))
                if start_in_range_local or end_in_range_local:
                  if f'{pdbId}_{ins_chainId}' not in expStruct_numEntity100InstanceReprRangeCover:
                    expStruct_numEntity100InstanceReprRangeCover.append(f'{pdbId}_{ins_chainId}')
                  if pdbId not in pdbId_to_download:
                    ## save pdb id
                    pdbId_to_download.append(pdbId)
                  if unpId_key not in case_with_exp_stucture.keys():
                    case_with_exp_stucture[unpId_key] = [f'{pdbId}_{ins_chainId}']
                    ## save record
                    expStruct_num += 1
                    if unp_pdb_map_path is not None:
                      unp_expPdb_write.write(f'{setNm}\t{unpId_key}:{unpId_dict[unpId_key]["weight_score"]}\t{exp_entityId}\t{pdbId}_{ins_chainId}/{threeIdxSetMap[0][0]}-{threeIdxSetMap[0][-1]}\n')
                  else:
                    if f'{pdbId}_{ins_chainId}' not in case_with_exp_stucture[unpId_key]:
                      case_with_exp_stucture[unpId_key].append(f'{pdbId}_{ins_chainId}')
                      ## save record
                      expStruct_num += 1
                      if unp_pdb_map_path is not None:
                        unp_expPdb_write.write(f'{setNm}\t{unpId_key}:{unpId_dict[unpId_key]["weight_score"]}\t{exp_entityId}\t{pdbId}_{ins_chainId}/{threeIdxSetMap[0][0]}-{threeIdxSetMap[0][-1]}\n')
          except Exception as err:
            print_exception(err=err,level=enum_exception_level.ERROR,message=f'loop expPDB_list: {exp_entityId}')
      if check_AF_struct:
        if afAccessionPath.split('.')[-1] == 'pickle':
          for unpId in current_list:
            if unpId in afAccessionDict.keys():
              afPbdRecord_list.extend(afAccessionDict[unpId])
        elif afAccessionPath.split('.')[-1] == 'csv':
          afPbdRecord_list.extend([','.join(row.astype(str).tolist()) for _, row in afAcc_map.loc[afAcc_map['unpAcc'].isin(current_list)].iterrows()])
        else:
          # if only use unpId as search pattern, will find matching substring in unpId which is not a desired match
          for unpId in current_list:
            afPbdRecord_list.extend([os.popen(f"grep -i '^{unpId},' {afAccessionPath}").read().strip('\n')])
        #if len(afPbdRecord_list[0]) > 0:
        #  print(f"{unpId};{afPbdRecord_list}")
        for afPdbRecord in afPbdRecord_list:
          if len(afPdbRecord) == 0:
            continue
          afPbdRecord_split = afPdbRecord.split(',')
          unpId = afPbdRecord_split[0]
          af_start_idx = int(afPbdRecord_split[1])
          af_end_idx = int(afPbdRecord_split[2])
          afPdb_to_download.append(afPbdRecord_split[3])
          if unpId in unpId_key_key_mapping.keys():
            for unpId_key in unpId_key_key_mapping[unpId]:
              start_in_range_local = (af_start_idx <= int(unpId_dict[unpId_key]['seq_range_start'])) and (af_end_idx >= int(unpId_dict[unpId_key]['seq_range_start']))
              end_in_range_local = (af_start_idx <= int(unpId_dict[unpId_key]['seq_range_end'])) and (af_end_idx >= int(unpId_dict[unpId_key]['seq_range_end']))
              if start_in_range_local or end_in_range_local:
                af2Struct_num += 1
                if unp_pdb_map_path is not None:
                  unp_afPdb_write.write(f"{setNm}\t{unpId_dict[unpId_key]['record_id']}\t{unpId}\t{afPbdRecord_split[3]}:{afPbdRecord_split[1]}-{afPbdRecord_split[2]}\n")
    ## download exp structure files
    download_dir = f'{root_dir}/data_process/structure/shin2021/download_structures/{setNm}'
    if not os.path.isdir(download_dir):
      os.mkdir(download_dir)
    if download_struct:
      ## download exp structure
      if len(pdbId_to_download) > 0:
        bio_pdbList.download_pdb_files(pdbId_to_download,pdir=download_dir,file_format=struct_format,overwrite=True)
      ## download af structure
      for af_id in afPdb_to_download:
        url_str=f'https://alphafold.ebi.ac.uk/files/{af_id}-model_v{afDB_version}.cif'
        pdb_flNm = f'{download_dir}/{af_id}-model_v{afDB_version}.cif'
        if (not os.path.isfile(pdb_flNm)) or (os.path.getsize(pdb_flNm) == 0):
          r = requests.get(url_str,allow_redirects=True)
          with open(pdb_flNm,'wb') as cif:
            cif.write(r.content)
    ## count sequence number having exp structure
    expStruct_numSeq = len(case_with_exp_stucture)
    print(f"{setNm},{famSeqNum},{expStruct_numEntityAll},{expStruct_numEntity100},{expStruct_numEntity100InstanceAll},{len(expStruct_numEntity100InstanceReprRangeCover)},{expStruct_numSeq},{expStruct_num},{af2Struct_num}",flush=True)
    if unp_pdb_map_path is not None:
      if check_exp_struct:
        unp_expPdb_write.close()
      if check_AF_struct:
        unp_afPdb_write.close()

def query_struct_pfam(
  set_list: List,
  set_file: str,
  data_path: str,
  unpId_batch_size: int,
  afDB_version: int,
  afAccessionPath: str,
  download_struct: bool = False,
  struct_format: str = 'mmCif',
  check_exp_struct: bool = False,
  check_AF_struct: bool = False,
  unp_pdb_map_path: str = None,
  logger: logging.Logger = None):
  """Collect structures for sequence set
  
  Args:
    set_list: list of family names
  
  """
  ## read family list to process
  if set_list is None:
      set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
 
  if afAccessionPath.split('.')[-1] == 'pickle':
    with open(f'{afAccessionPath}', 'rb') as handle:
      afAccessionDict = pkl.load(handle)
  elif afAccessionPath.split('.')[-1] == 'csv':
    afAcc_map = pd.read_csv(afAccessionPath,sep=',',header=None,names=['unpAcc','start','end','afAcc','version'])

  bio_pdbList = Bio.PDB.PDBList(verbose=False)

  ## loop over family list
  for setI in range(len(set_list)):
    setNm = set_list[setI]
    ## list to hold pdb ids to download
    pdbId_to_download = []
    afPdb_to_download = []
    ## save uniprot id, experiment pdb id mapping
    if unp_pdb_map_path is not None:
      if check_exp_struct:
        unp_expPdb_write = open(f'{unp_pdb_map_path}/{setNm}_unp_expPDB_map.tsv','w')
      if check_AF_struct:
        unp_afPdb_write = open(f'{unp_pdb_map_path}/{setNm}_unp_afPDB_map.tsv','w')
    
    ## statistic variables
    expStruct_numSeq = 0
    expStruct_numEntityAll = 0
    expStruct_numEntity100 = 0
    expStruct_numEntity100InstanceAll = 0
    expStruct_numEntity100InstanceReprRangeCover = []
    expStruct_num = 0
    af2Struct_num = 0
    
    seq_env = lmdb.open(f'{data_path}/{setNm}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
    famSeqNum = 0
    unpId_list = []
    unpId_dict = {}
    unpId_key_key_mapping = {}
    with seq_env.begin(write=False) as txn:
      num_examples = pkl.loads(txn.get(b'num_examples'))
      for i in range(num_examples):
        item = pkl.loads(txn.get(str(i).encode()))
        famSeqNum += 1
        if famSeqNum % 1000 == 0:
          #logger.info(f"set {setNm}, at seq #: {famSeqNum}")
          print(f"set {setNm}, at seq #: {famSeqNum}")
        unpId = item['unpIden'].split('.')[0]
        seq_range_start, seq_range_end = item['range'].split('-')
        unpId_list.append(unpId)
        if unpId not in unpId_dict.keys():
          unpId_dict[unpId] = [{'seq_range_start': seq_range_start,
                                'seq_range_end': seq_range_end,
                                'weight_score': item['seq_reweight'],
                                'unp_seq_segment': item['primary'],
                                'unp_seq_segment_align': item['msa_seq']}]
        else:
          unpId_dict[unpId].append({'seq_range_start': seq_range_start,
                                    'seq_range_end': seq_range_end,
                                    'weight_score': None,
                                    'unp_seq_segment': raw_seq_str,
                                    'unp_seq_segment_align': aligned_seq_str})
        try:
          afPbdRecord_list = afAccessionDict[f'{unpId}']
          #if len(afPbdRecord_list[0]) > 0:
          #  print(f"{unpId};{afPbdRecord_list}")
          for afPdbRecord in afPbdRecord_list:
            if len(afPdbRecord) == 0:
              continue
            afPbdRecord_split = afPdbRecord.split(',')
            if unp_pdb_map_path is not None:
              unp_afPdb_write.write(f'{setNm}\t{record.id}\t{unpId}\t{afPbdRecord_split[3]}:{afPbdRecord_split[1]}-{afPbdRecord_split[2]}\n')
            af2Struct_num += 1
        except KeyError as err:
          pass
        except Exception as err:
          print_exception(err=err,level=enum_exception_level.ERROR)


      ### rscb API batch query ###
      tail_idx = 0
      case_with_exp_stucture = []
      while tail_idx < len(unpId_list):
        current_list = unpId_list[tail_idx:tail_idx+unpId_batch_size]
        tail_idx += unpId_batch_size
        try:
          ## get polymer entity instances from uniport mapping file
          #expPDB_list = uniprotDict[f'UniRef100_{unpId}'][0].split('\t')[5].split('; ')
          
          ## get polymer entities from RCSB search query
          #expPDB_list, total_num, repr_num = unp_repr_pdbs(unp_id=unpId,seqIdentity_percent=100,represent_by='entity_poly.rcsb_sample_sequence_length',logger=logger)
          #start = time.time()
          expPDB_list, total_num, repr_num = unp_batch_repr_pdbs(unpId_list=current_list,seqIdentity_percent=100,represent_by='entity_poly.rcsb_sample_sequence_length',logger=logger)
          #end = time.time()
          #print("The time of execution of above program is :", end-start)
          expStruct_numEntityAll += total_num
          expStruct_numEntity100 += repr_num

          ## loop over polymer entities
          for exp_entityId in expPDB_list:
            if len(exp_entityId) == 0:
              continue
            #pdbId, pdbChainId = exp_entityId.split(':')
            pdbId, pdbEntityId = exp_entityId.split('_')
            
            ## get top-1 polymer entity instance among  
            top_insId, all_indId, unp_acc = select_entity_instance(entity_id=exp_entityId)
            expStruct_numEntity100InstanceRepr += 1
            expStruct_numEntity100InstanceAll += len(all_indId)  
            ins_chainId = top_insId.split('.')[-1]
            
            ## check if the top instance cover segment range
            threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id = get_index_mapping(pdb_id=pdbId,chain_id=ins_chainId,unpAcc=unp_acc,logger=logger)
            assert unp_acc in unpId_dict.keys()
            real_unp_pos = np.where(np.array(threeIdxSetMap[0]).astype(int) > 0)[0]
            start_in_range, end_in_range = False, False
            for unp_dict_case in unpId_dict[unp_acc]:
              start_in_range |= (int(threeIdxSetMap[0][real_unp_pos[0]]) <= int(unp_dict_case['seq_range_start'])) and (int(threeIdxSetMap[0][real_unp_pos[-1]]) >= int(unp_dict_case['seq_range_start']))
              end_in_range |= (int(threeIdxSetMap[0][real_unp_pos[0]]) <= int(unp_dict_case['seq_range_end'])) and (int(threeIdxSetMap[0][real_unp_pos[-1]]) >= int(unp_dict_case['seq_range_end']))
            if start_in_range or end_in_range :
              expStruct_numEntity100InstanceReprRangeCover += 1
              if f'{unp_acc}/{unpId_dict[unp_acc]["seq_range_start"]}_{unpId_dict[unp_acc]["seq_range_end"]}' not in case_with_exp_stucture:
                case_with_exp_stucture.append(f'{unp_acc}/{unpId_dict[unp_acc]["seq_range_start"]}_{unpId_dict[unp_acc]["seq_range_end"]}')
            else:
              continue
            if unp_pdb_map_path is not None:
              unp_expPdb_write.write(f'{setNm}\t{unp_acc}/{unpId_dict[unp_acc]["seq_range_start"]}-{unpId_dict[unp_acc]["seq_range_end"]}:{unpId_dict[unp_acc]["weight_score"]}\t{exp_entityId}\t{pdbId}_{ins_chainId}/{threeIdxSetMap[0][0]}-{threeIdxSetMap[0][-1]}\n')
          
        except KeyError as err:
          print_exception(err=err,level=enum_exception_level.ERROR)
        except AssertionError as err:
          print_exception(err=err,level=enum_exception_level.ERROR,message=f'unp_acc: {unp_acc}')
        except Exception as err:
          print_exception(err=err,level=enum_exception_level.ERROR,message=f'setNm: {setNm}')
      ## count sequence number having exp structure
      expStruct_numSeq = len(case_with_exp_stucture)
    print(f"{setNm},{famSeqNum},{expStruct_numEntityAll},{expStruct_numEntity100},{expStruct_numEntity100InstanceAll},{expStruct_numEntity100InstanceRepr},{expStruct_numEntity100InstanceReprRangeCover},{expStruct_numSeq},{af2Struct_num}",flush=True)
    if unp_pdb_map_path is not None:
      unp_expPdb_write.close()
      unp_afPdb_write.close()


def exp_struct_feature(family_name: str,
                      pdb_id: str,
                      unp_id: str,
                      struct_format: str,
                      pdbChain_id: str,
                      start_idx: int,
                      end_idx: int) -> Tuple[str,np.ma.masked_array,np.ma.masked_array,str,str,str,np.ma.masked_array]:
  """Represent structure as graph and prepare features.
  
  Graph constuction: residue as a node; edge exists if distance of C_alpha atom pairs <= 10 \AA.

  Features set and calculation methods:
    Node features:
      Amino Acid Identity [AA]: 20, use embeddings from pretrained language models
      Second Structure [SS]: 3-class (helix,sheet,coil) read from pdb/mmcif file or 8 class (H,B,E,G,I,T,S,-) use BioPython and DSSP
      Relative Solvent Accessibility [RSA]: scaler, [0-1], use BioPython and DSSP
      
      EXTENDED SET
      Residue Depth(RD): scalar, [0-1], BioPython and version 2.6.1 of MSMS
      Protrusion Index(PI): 6, [0-1], version 1.0 of PSAIA
      Sinusoidal positional encoding of index i: 1, [-1,1]
      Torsion angle encoding: 6, {sin, cos} * {\phi, \psi, \omega}
      
      NOT USED
      Residue Type(RT), 5-class
        positive charged: R,H,K
        negative charged: D,E
        polar uncharged: S,T,N,Q
        unique: C,G,P
        hydrophobic: A,I,L,M,F,W,Y,V
   
    Pairwise/Edge features:
      Relative positional encoding: scalar, sinusoidal encoding of abs(j-i)
      
      EXTENDED SET
      Pairwise distance: 16, Gaussian RBF encoded C_alpha-C_alpha distance spaced 0-20 \AA
      Angle between two amide planes' normal vectors with sinusoidal encoding: 2, [-1,1], {sin, cos} * angle

    Other edge geometric features, first define a residue local frame (Ingraham2019, Octavian2021, Morehead2021)
      Direction
      Orientation (quaternion)
    
    Reference:
      DIPS-PLUS
  
  Args:
    family_name (str): family name.
    pdb_id (str): 4-letter pdb id.
    unp_id (str): Uniprot accession number.
    pdbChain_id (str): pdb chain id (non auth chain id).
    struct_format (str): format of structure file. Choose from {'pdb', mmCif}. Defaults to 'mmCif'.
    start_idx (int): starting residue index of sequence segment (use uniprot index, 1-based).
    end_idx (int): ending residue index of sequence segment (use uniprot index, 1-based).
  


  Returns:
    str: target pdb sequence string.
    numpy masked_array: target pdb contact map(binary).
    numpy masked_array: target pdb distance map.
    str: target pdb secondary structure string(3 class).
    str: target pdb secondary structure string(8 class).
    numpy masked_array: target pdb relative relative solvent accessibility(real value)
    str: target pdb relative relative solvent accessibility(2 class)

  """
  
  bio_pdbList = Bio.PDB.PDBList()
  if struct_format.lower() == 'mmcif':
      bio_parser = Bio.PDB.FastMMCIFParser(QUIET=True)
  elif struct_format.lower() == 'pdb':
      bio_parser = Bio.PDB.PDBParser()
  
  pdbfile_ext = {'mmCif': 'cif',
                 'pdb': 'pdb'}

  ### acquire unp-pdb-auth index mapping ###
  threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id = get_index_mapping(
                                                                  pdb_id = pdb_id,
                                                                  chain_id = pdbChain_id,
                                                                  unpAcc = unp_id)

  #download_dir = f'{root_dir}/data_process/structure/shin2021/download_structures/{family_name}'
  #if not os.path.isdir(download_dir):
  #    os.mkdir(download_dir)
  #pdb_flNm = bio_pdbList.retrieve_pdb_file(pdb_id,pdir=download_dir,file_format=struct_format,overwrite=True)
  #print(pdb_flNm, flush=True)
  
  ## process pdb (if pdb file exists)
  pdb_flNm = f'{root_dir}/data_process/structure/shin2021/download_structures/{family_name}/{pdb_id.lower()}.{pdbfile_ext[struct_format]}'
  if os.path.isfile(pdb_flNm):
    print(f">>pdb: {pdb_id}-{pdbChain_id}/{auth_chain_id}", flush=True)
    pdb_struc = bio_parser.get_structure(pdb_id,pdb_flNm)
    pdb_model = pdb_struc[0][auth_chain_id] #Bio.PDB.Chain.Chain object
    
    # DEBUG
    #for resi in pdb_model.get_residues():
    #  print(resi.get_resname())

   
    ### get mapping index ###
    mapping_idx = []
    for i in range(len(threeIdxSetMap[0])):
      if int(threeIdxSetMap[0][i]) != -1:
        if int(threeIdxSetMap[0][i]) >= start_idx and int(threeIdxSetMap[0][i]) <= end_idx:
          mapping_idx.append(i)
    
    if len(mapping_idx) == 0:
      raise Exception(f'the given uniprot range {start_idx}-{end_idx} is not covered by pdb instance {pdb_id.upper()}-{pdbChain_id.upper()}')

    if pdb_seq is None:
      pdb_seq = read_conPdbSeq_from_mmcif(pdb_file = pdb_flNm)
    ### AA seq ###
    pdb_seq_target = pdb_seq[int(threeIdxSetMap[1][mapping_idx[0]])-1 : int(threeIdxSetMap[1][mapping_idx[-1]])]

    ### generate adjacency matrix ###
    distance_map, contact_map = singleProtein_distMat(
                                  pdb_model = pdb_model,
                                  threeIdxSetMap = threeIdxSetMap,
                                  repr_atom='CB',
                                  contact_thresh=10.0,
                                  symmetric=True)
    mapping_idx_range = range(mapping_idx[0],mapping_idx[-1]+1)
    map_cut_mesh = np.ix_(mapping_idx_range,mapping_idx_range)
    distance_map_target = distance_map[map_cut_mesh]
    contact_map_target = contact_map[map_cut_mesh]
    
    
    ### DSSP ###
    dssp_out = get_dssp_outputs(struct_file=pdb_flNm,file_format=struct_format)

    ### SS3 (return PDB residue index) ###
    ss3_dict = read_ss_from_file(
                struct_file = pdb_flNm,
                file_format = struct_format)
    ss3_seq_target = ''
    for i in range(mapping_idx[0],mapping_idx[-1]+1):
      if (auth_chain_id,threeIdxSetMap[2][i]) in ss3_dict.keys():
        ss3_seq_target += ss3_dict[auth_chain_id,threeIdxSetMap[2][i]]
      else:
        ss3_seq_target += DEFAULT_MISSING_SS

    ### SS8 ###
    ss8_seq_target = ''
    for i in range(mapping_idx[0],mapping_idx[-1]+1):
      ss8_seq_target += get_dssp_ss_residue(
                          dssp_dict=dssp_out,
                          chain=auth_chain_id,
                          residue_idx=threeIdxSetMap[2][i])

    ### RSA ###
    rsa_class_seq_target = ''
    rsa_value_seq_target_list = []
    for i in range(mapping_idx[0],mapping_idx[-1]+1):
      resi_rsa_value, resi_rsa_class = get_dssp_rsa_residue(
                                          dssp_dict=dssp_out,
                                          chain=auth_chain_id,
                                          residue_idx=threeIdxSetMap[2][i],
                                          thres=BINARY_RSA_CUTOFF)
      rsa_class_seq_target += resi_rsa_class
      rsa_value_seq_target_list.append(resi_rsa_value)
    rsa_value_seq_target = np.ma.masked_array(rsa_value_seq_target_list, np.isnan(rsa_value_seq_target_list),fill_value=np.nan)
    ## positional encoding will be handled by embedding module in model

    return pdb_seq_target, contact_map_target, distance_map_target, ss3_seq_target, ss8_seq_target, rsa_class_seq_target, rsa_value_seq_target

def af_structure_plddt_filer_mmcif(fl_name: str, plddt_thres: float=50.):
  """Select positions with acceptable pLDDT confidence score.

  Args:
    fl_name (str): path of structure mmcif file.
    plddt_thres (float): pLDDT confidence threshold. Accept the reside if its' pLDDT >= threshopld. Default: 50.0

  Returns:
    List: list of accepted residues' indices
    List: list of accepted residues' plddt score
  """
  try:
    pdb_id, file_format = os.path.splitext(os.path.basename(fl_name))
    mmcif_header=Bio.PDB.MMCIF2Dict.MMCIF2Dict(fl_name)
    pos_list = []
    plddt_list = []
    for r in range(len(mmcif_header['_atom_site.group_PDB'])):
      auth_seq_id = int(mmcif_header['_atom_site.auth_seq_id'][r])
      plddt_score = float(mmcif_header['_atom_site.B_iso_or_equiv'][r])
      if plddt_score >= plddt_thres and auth_seq_id not in pos_list:
        pos_list.append(auth_seq_id)
        plddt_list.append(plddt_score)
  except Exception as err:
    print_exception(err=err, level=enum_exception_level.ERROR, message=f'alphafold pLDDT score error: {pdb_id}')
  return pos_list, plddt_list

def alphafold_struct_feature(
      family_name: str,
      af_id: str,
      afDB_version: int,
      unp_id: str,
      struct_format: str,
      start_idx: int,
      end_idx: int,
      af_start_idx: int,
      af_end_idx: int) -> Tuple[str,np.ma.masked_array,np.ma.masked_array,str,str,np.ma.masked_array,str]:
  """Represent structure as graph and prepare features.
  
  Graph constuction: residue as a node; edge exists if distance of C_beta atom pairs <= 8 \AA.

  Features set and calculation methods:
    Node features:
      Amino Acid Identity [AA]: 20, use embeddings from pretrained language models
      Second Structure [SS]: 3-class (helix,sheet,coil) read from pdb/mmcif file or 8 class (H,B,E,G,I,T,S,-) use BioPython and DSSP
      Relative Solvent Accessibility [RSA]: scaler, [0-1], use BioPython and DSSP
      
      EXTENDED SET
      Residue Depth(RD): scalar, [0-1], BioPython and version 2.6.1 of MSMS
      Protrusion Index(PI): 6, [0-1], version 1.0 of PSAIA
      Sinusoidal positional encoding of index i: 1, [-1,1]
      Torsion angle encoding: 6, {sin, cos} * {\phi, \psi, \omega}
      
      NOT USED
      Residue Type(RT), 5-class
        positive charged: R,H,K
        negative charged: D,E
        polar uncharged: S,T,N,Q
        unique: C,G,P
        hydrophobic: A,I,L,M,F,W,Y,V
   
    Pairwise/Edge features:
      Relative positional encoding: scalar, sinusoidal encoding of abs(j-i)
      
      EXTENDED SET
      Pairwise distance: 16, Gaussian RBF encoded C_alpha-C_alpha distance spaced 0-20 \AA
      Angle between two amide planes' normal vectors with sinusoidal encoding: 2, [-1,1], {sin, cos} * angle

    Other edge geometric features, first define a residue local frame (Ingraham2019, Octavian2021, Morehead2021)
      Direction
      Orientation (quaternion)
    
    Reference:
      DIPS-PLUS
  
  Args:
    family_name (str): family name.
    pdb_id (str): 4-letter pdb id.
    unp_id (str): Uniprot accession number.
    pdbChain_id (str): pdb chain id (non auth chain id).
    struct_format (str): format of structure file. Choose from {'pdb', mmCif}. Defaults to 'mmCif'.
    start_idx (int): starting residue index of sequence segment (use uniprot index, 1-based).
    end_idx (int): ending residue index of sequence segment (use uniprot index, 1-based).
  


  Returns:
    str: target pdb sequence string.
    numpy masked_array: target pdb contact map(binary).
    numpy masked_array: target pdb distance map.
    str: target pdb secondary structure string(3 class).
    str: target pdb secondary structure string(8 class).
    numpy masked_array: target pdb relative relative solvent accessibility(real value)
    str: target pdb relative relative solvent accessibility(2 class)

  """
  bio_pdbList = Bio.PDB.PDBList()
  if struct_format.lower() == 'mmcif':
      bio_parser = Bio.PDB.FastMMCIFParser(QUIET=True)
  elif struct_format.lower() == 'pdb':
      bio_parser = Bio.PDB.PDBParser()
  
  pdbfile_ext = {'mmCif': 'cif',
                 'pdb': 'pdb'}
  chain_id = 'A'

  download_dir = f'{root_dir}/data_process/structure/shin2021/download_structures/{family_name}'
  if not os.path.isdir(download_dir):
      os.mkdir(download_dir)
  
  url_str=f'https://alphafold.ebi.ac.uk/files/{af_id}-model_v{afDB_version}.cif'
  pdb_flNm = f'{download_dir}/{af_id}-model_v{afDB_version}.cif'
  if (not os.path.isfile(pdb_flNm)) or (os.path.getsize(pdb_flNm) == 0):
    r = requests.get(url_str,allow_redirects=True)
    with open(pdb_flNm,'wb') as cif:
      cif.write(r.content)
  
  ## process pdb (if pdb file exists)
  if os.path.isfile(pdb_flNm):
    print(f">>pdb: {af_id}-{chain_id}", flush=True)
    pdb_struc = bio_parser.get_structure(af_id,pdb_flNm)
    pdb_model = pdb_struc[0][chain_id] #Bio.PDB.Chain.Chain object
    
    ## pLDDT scores
    good_positions,good_plddt = af_structure_plddt_filer_mmcif(fl_name=pdb_flNm, plddt_thres=50.)
    low_conf_idx = []
        
    af_range = list(range(af_start_idx, af_end_idx+1))
    target_range = list(range(start_idx, end_idx+1))
    mapping_idx = []
    for i in range(len(af_range)):
      if af_range[i] in target_range:
        mapping_idx.append(i)
      if af_range[i] not in good_positions:
        low_conf_idx.append(i)
    
    ## number of positions not covered by pdb in two ends
    uncover_num_start, uncover_num_end = 0, 0
    if start_idx < af_start_idx:
      uncover_num_start = af_start_idx - start_idx
    if end_idx > af_end_idx:
      uncover_num_end = end_idx - af_end_idx

    pdb_seq = read_conPdbSeq_from_mmcif(pdb_file = pdb_flNm,entity_id=1)
    ### AA seq ###
    pdb_seq_target = pdb_seq[af_range[mapping_idx[0]]-1:af_range[mapping_idx[-1]]]

    ## fake index map
    threeIdxSetMap = [[str(af_i) for af_i in af_range]]*3
    ### generate adjacency matrix ###
    distance_map, contact_map = singleProtein_distMat(
                                  pdb_model = pdb_model,
                                  threeIdxSetMap = threeIdxSetMap,
                                  repr_atom='CB',
                                  contact_thresh=8.0,
                                  symmetric=True)
    low_conf_mesh = np.ix_(low_conf_idx,low_conf_idx)
    distance_map[low_conf_mesh] = np.ma.masked
    contact_map[low_conf_mesh] = np.ma.masked 
    map_cut_mesh = np.ix_(mapping_idx,mapping_idx)
    distance_map_target = distance_map[map_cut_mesh]
    contact_map_target = contact_map[map_cut_mesh]
    ## pad uncovered positions
    if uncover_num_start > 0 or uncover_num_end > 0:
      distance_map_target = np.pad(distance_map_target.filled(fill_value=np.nan), ((uncover_num_start,uncover_num_end),(uncover_num_start,uncover_num_end)), 'constant', constant_values=np.nan)
      contact_map_target = np.pad(contact_map_target.filled(fill_value=np.nan), ((uncover_num_start,uncover_num_end),(uncover_num_start,uncover_num_end)), 'constant', constant_values=np.nan)
      # convert back to masked_array
      distance_map_target = np.ma.masked_array(distance_map_target, np.isnan(distance_map_target), fill_value=np.nan)
      contact_map_target = np.ma.masked_array(contact_map_target, np.isnan(contact_map_target), fill_value=np.nan)

    ### Residue identity embedding ###
    ## from pretrained model, dataloader will load directly from saved embedding files
    
    ### DSSP ###
    dssp_out = get_dssp_outputs(struct_file=pdb_flNm,file_format=struct_format)

    ### SS3 (return PDB residue index) ###
    ss3_dict = read_ss_from_file(
                struct_file = pdb_flNm,
                file_format = struct_format)
    ss3_seq_target = ''
    for i in mapping_idx:
      if (chain_id,threeIdxSetMap[2][i]) in ss3_dict.keys():
        ss3_seq_target += ss3_dict[chain_id,threeIdxSetMap[2][i]]
      else:
        ss3_seq_target += DEFAULT_MISSING_SS
    ## pad uncovered positions
    if uncover_num_start > 0 or uncover_num_end > 0:
      ss3_seq_target = DEFAULT_MISSING_SS*uncover_num_start + ss3_seq_target + DEFAULT_MISSING_SS*uncover_num_end

    ### SS8 ###
    ss8_seq_target = ''
    for i in mapping_idx:
      ss8_seq_target += get_dssp_ss_residue(
                          dssp_dict=dssp_out,
                          chain=chain_id,
                          residue_idx=threeIdxSetMap[2][i])
    ## pad uncovered positions
    if uncover_num_start > 0 or uncover_num_end > 0:
      ss8_seq_target = DEFAULT_MISSING_SS*uncover_num_start + ss8_seq_target + DEFAULT_MISSING_SS*uncover_num_end
    
    ### RSA ###
    rsa_class_seq_target = ''
    rsa_value_seq_target_list = []
    for i in mapping_idx:
      resi_rsa_value, resi_rsa_class = get_dssp_rsa_residue(
                                          dssp_dict=dssp_out,
                                          chain=chain_id,
                                          residue_idx=threeIdxSetMap[2][i],
                                          thres=BINARY_RSA_CUTOFF)
      rsa_class_seq_target += resi_rsa_class
      rsa_value_seq_target_list.append(resi_rsa_value)
    rsa_value_seq_target = np.ma.masked_array(rsa_value_seq_target_list, np.isnan(rsa_value_seq_target_list),fill_value=np.nan)
    ## pad uncovered positions
    if uncover_num_start > 0 or uncover_num_end > 0:
      rsa_class_seq_target = DEFAULT_MISSING_RSA_CLA*uncover_num_start + rsa_class_seq_target + DEFAULT_MISSING_RSA_CLA*uncover_num_end
      rsa_value_seq_target = np.pad(rsa_value_seq_target.filled(fill_value=np.nan),(uncover_num_start,uncover_num_end),'constant',constant_values=DEFAULT_MISSING_RSA)
      rsa_value_seq_target = np.ma.masked_array(rsa_value_seq_target, np.isnan(rsa_value_seq_target),fill_value=np.nan)
    ## positional encoding will be handled by embedding module in model ##

  return pdb_seq_target, contact_map_target, distance_map_target, ss3_seq_target, ss8_seq_target, rsa_class_seq_target, rsa_value_seq_target

def generate_afStruct_feature_ShinData(
      set_list: List,
      set_file: str,
      unp_pdb_map_path: str,
      feature_save_dir: str,
      min_len: int,
      afDB_version: int):

  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
  
  ## Load exp structure list
  for set_i in range(len(set_list)):
    set_name = set_list[set_i]
    print(f'set: {set_name}', flush=True)
    if os.path.getsize(f'{unp_pdb_map_path}/{set_name}_unp_afPDB_map.tsv') == 0:
      continue
    struct_list = np.loadtxt(f'{unp_pdb_map_path}/{set_name}_unp_afPDB_map.tsv',dtype='str',delimiter='\t')
    if len(struct_list.shape) == 1:
      struct_list = np.reshape(struct_list,(-1,struct_list.shape[-1]))

    ## Create a new LMDB environment
    map_size = (1024 * 100) * (2 ** 20) # 100G
    env = lmdb.open(f"{feature_save_dir}/{set_name}.lmdb", map_size=map_size)
    set_feature_list = []
    for strt_i in range(struct_list.shape[0]):
      try:
        family_nm = struct_list[strt_i,0]
        unpId_range_str, weight = struct_list[strt_i,1].split(':')
        _id,unp_range_str = unpId_range_str.split('/')
        range_start, range_end = unp_range_str.split('-')
        unp_id = struct_list[strt_i,2]
        af_id, af_range_str = struct_list[strt_i,3].split(':')
        af_start,af_end = af_range_str.split('-') 
        if int(range_start) >= int(af_end) or int(range_end) <= int(af_start):
          continue
        pdb_seq_target, contact_map_target, distance_map_target, \
        ss3_seq_target, ss8_seq_target, rsa_class_seq_target, \
        rsa_value_seq_target = alphafold_struct_feature(
                                family_name=family_nm,
                                af_id=af_id,
                                afDB_version=afDB_version,
                                unp_id=unp_id,
                                struct_format='mmCif',
                                start_idx=int(range_start),
                                end_idx=int(range_end),
                                af_start_idx=int(af_start),
                                af_end_idx=int(af_end))
        seq_len = len(pdb_seq_target)
        if seq_len < min_len:
            continue
        assert distance_map_target.shape[0] == seq_len
        if np.sum(~distance_map_target.mask) == 0:
          continue
        assert contact_map_target.shape[0] == seq_len
        assert np.sum(~contact_map_target.mask) > 0
        assert len(ss3_seq_target) == seq_len
        assert len (ss8_seq_target) == seq_len
        assert len(rsa_class_seq_target) == seq_len
        assert rsa_value_seq_target.shape[0] == seq_len
        assert np.sum(~rsa_value_seq_target.mask) > 0

        set_feature_list.append({
          'seq_id': f'{family_nm}/{unp_id}/{unp_range_str}/{af_id}',
          'seq_primary': pdb_seq_target,
          'seq_len': seq_len,
          'distance_map': distance_map_target.tobytes(fill_value=np.nan),
          'ss3': ss3_seq_target,
          'ss8': ss8_seq_target,
          'rsa_class': rsa_class_seq_target,
          'rsa_value': rsa_value_seq_target.tobytes(fill_value=np.nan),
          'seq_reweight': weight 
        })
      except Exception as err:
        print_exception(err=err,level=enum_exception_level.ERROR,message=f'set:{set_name},seq_id:{family_nm}/{unp_id}/{unp_range_str}/{af_id}')
    
    # start a new lmdb write transaction
    print(f'num: {len(set_feature_list)}')
    with env.begin(write=True) as txn:
      for i, entry in enumerate(set_feature_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    env.close()

def generate_afStruct_feature_ShinData_mapFunc(row_tuple,lmdb_path,afDB_version,min_len):
    curr_pro = mp.current_process()
    row = row_tuple[1]
    family_nm = row['protein_name']
    unpId_range_str, weight = row['info'].split(':')
    _id,unp_range_str = unpId_range_str.split('/')
    range_start, range_end = unp_range_str.split('-')
    unp_id = row['unp_id']
    af_id, af_range_str = row['afdb_id'].split(':')
    af_start,af_end = af_range_str.split('-') 
    if int(range_start) >= int(af_end) or int(range_end) <= int(af_start):
      return 0
    try:
      pdb_seq_target, contact_map_target, distance_map_target, \
      ss3_seq_target, ss8_seq_target, rsa_class_seq_target, \
      rsa_value_seq_target = alphafold_struct_feature(
                              family_name=family_nm,
                              af_id=af_id,
                              afDB_version=afDB_version,
                              unp_id=unp_id,
                              struct_format='mmCif',
                              start_idx=int(range_start),
                              end_idx=int(range_end),
                              af_start_idx=int(af_start),
                              af_end_idx=int(af_end))
      seq_len = len(pdb_seq_target)
      
      assert seq_len >= min_len, "seq_len >= min_len"
      assert len(ss3_seq_target) == seq_len, "len(ss3_seq_target) == seq_len"
      assert len (ss8_seq_target) == seq_len, "len (ss8_seq_target) == seq_len"
      assert len(rsa_class_seq_target) == seq_len, "len(rsa_class_seq_target) == seq_len"
      assert rsa_value_seq_target.shape[0] == seq_len, "rsa_value_seq_target.shape[0] == seq_len"
      assert np.sum(~rsa_value_seq_target.mask) > 0, "np.sum(~rsa_value_seq_target.mask) > 0"
      assert distance_map_target.shape[0] == seq_len, "distance_map_target.shape[0] == seq_len"
      assert contact_map_target.shape[0] == seq_len, "contact_map_target.shape[0] == seq_len"
      assert np.sum(~contact_map_target.mask) > 0, "np.sum(~contact_map_target.mask) > 0"
      assert np.sum(~distance_map_target.mask) > 0, "np.sum(~distance_map_target.mask) > 0"

      set_feature_dict = {
        'seq_id': f'{family_nm}/{unp_id}/{unp_range_str}/{af_id}',
        'seq_primary': pdb_seq_target,
        'seq_len': seq_len,
        'distance_map': distance_map_target.tobytes(fill_value=np.nan),
        'ss3': ss3_seq_target,
        'ss8': ss8_seq_target,
        'rsa_class': rsa_class_seq_target,
        'rsa_value': rsa_value_seq_target.tobytes(fill_value=np.nan),
        'seq_reweight': weight 
      }
      map_size = (1024 * 100) * (2 ** 20) # 100G
      env = lmdb.open(f'{lmdb_path}_{curr_pro._identity[0]}.lmdb', map_size=map_size)
      #with env.begin(write=False) as txn:
      #  num_count = pkl.loads(txn.get('num_examples'.encode()))
      # start a new lmdb write transaction
      with env.begin(write=True) as txn:
        num_count = int(env.stat()['entries'])
        txn.put(str(num_count).encode(), pkl.dumps(set_feature_dict))
        #txn.put(b'num_examples', pkl.dumps(num_count+1))
      env.close()
      return 1
    except Exception as err:
      print_exception(err=err,level=enum_exception_level.ERROR,message=f'set:{set_name},seq_id:{family_nm}/{unp_id}/{unp_range_str}/{af_id}')
      return 0
  

def generate_afStruct_feature_ShinData_multithread(
      set_list: List,
      set_file: str,
      unp_pdb_map_path: str,
      feature_save_dir: str,
      min_len: int,
      afDB_version: int,
      num_workers: int):
  
  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
  
  ## Load exp structure list
  for set_i in range(len(set_list)):
    set_name = set_list[set_i]
    print(f'set: {set_name}', flush=True)
    if os.path.getsize(f'{unp_pdb_map_path}/{set_name}_unp_afPDB_map.tsv') == 0:
      continue
    struct_df = pd.read_csv(f'{unp_pdb_map_path}/{set_name}_unp_afPDB_map.tsv', delimiter='\t', names=['protein_name','info','unp_id','afdb_id'])

    ## Create a new LMDB environment
    map_size = (1024 * 100) * (2 ** 20) # 100G
    env = lmdb.open(f"{feature_save_dir}/{set_name}.lmdb", map_size=map_size)
    #with env.begin(write=True) as txn:
    #  txn.put(b'num_examples', pkl.dumps(0))
    with mp.Pool(num_workers) as p:
      mp_returns = p.map(partial(generate_afStruct_feature_ShinData_mapFunc,lmdb_path=f"{feature_save_dir}/{set_name}",afDB_version=afDB_version,min_len=min_len),struct_df.iterrows())
    
    all_data_list = []
    for i in range(1,num_workers+1):
      sub_env = lmdb.open(f"{feature_save_dir}/{set_name}_{i}.lmdb", map_size=map_size)
      with sub_env.begin(write=False) as sub_txn:
        for key, value in sub_txn.cursor():
          item = pkl.loads(value)
          all_data_list.append(item)
      sub_env.close()
      os.system(f"rm -r {feature_save_dir}/{set_name}_{i}.lmdb")
    with env.begin(write=True) as txn:
      for i, entry in enumerate(all_data_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(len(all_data_list)))
    
    # with env.begin(write=False) as txn:
    #   num_count = pkl.loads(txn.get('num_examples'.encode()))
    print(f'num (returned by map): {np.sum(mp_returns)}')
    print(f'num (read from lmdb): {len(all_data_list)}')
  return None

def generate_expStruct_feature_ShinData(
      set_list: List,
      set_file: str, 
      unp_pdb_map_path: str,
      feature_save_dir: str,
      min_len: int = 10):

  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
  
  ## Load exp structure list
  for set_i in range(len(set_list)):
    set_name = set_list[set_i]
    print(f'set: {set_name}', flush=True)
    if os.path.getsize(f'{unp_pdb_map_path}/{set_name}_unp_expPDB_map.tsv') == 0:
        continue
    struct_list = np.loadtxt(f'{unp_pdb_map_path}/{set_name}_unp_expPDB_map.tsv',dtype='str',delimiter='\t')
    if len(struct_list.shape) == 1:
      struct_list = np.reshape(struct_list,(-1,struct_list.shape[-1]))

    ## Create a new LMDB environment
    map_size = (1024 * 15) * (2 ** 20) # 15G
    env = lmdb.open(f"{feature_save_dir}/{set_name}.lmdb", map_size=map_size)
    set_feature_list = []
    for strt_i in range(struct_list.shape[0]):
      try:
        family_nm = struct_list[strt_i,0]
        unpId_range_str, weight = struct_list[strt_i,1].split(':')
        unp_id,unp_range_str = unpId_range_str.split('/')
        range_start, range_end = unp_range_str.split('-')
        pdb_id, entity_id = struct_list[strt_i,2].split('_')
        ins_pdb_id, ins_range_str = struct_list[strt_i,3].split('/')
        tmp,chain_id = ins_pdb_id.split('_')
        
        pdb_seq_target, contact_map_target, distance_map_target, \
        ss3_seq_target, ss8_seq_target, rsa_class_seq_target, \
        rsa_value_seq_target = exp_struct_feature(
                                family_name=family_nm,
                                pdb_id=pdb_id,
                                unp_id=unp_id,
                                struct_format='mmCif',
                                pdbChain_id=chain_id,
                                start_idx=int(range_start),
                                end_idx=int(range_end))
        seq_len = len(pdb_seq_target)
        if seq_len < min_len:
          continue
        assert distance_map_target.shape[0] == seq_len
        if np.sum(~distance_map_target.mask) == 0:
          continue
        assert contact_map_target.shape[0] == seq_len
        assert np.sum(~contact_map_target.mask) > 0
        assert len(ss3_seq_target) == seq_len
        assert len (ss8_seq_target) == seq_len
        assert len(rsa_class_seq_target) == seq_len
        assert rsa_value_seq_target.shape[0] == seq_len
        assert np.sum(~rsa_value_seq_target.mask) > 0

        set_feature_list.append({
          'seq_id': f'{family_nm}/{unp_id}/{unp_range_str}/{pdb_id}-{chain_id}',
          'seq_primary': pdb_seq_target,
          'seq_len': seq_len,
          'distance_map': distance_map_target.tobytes(fill_value=np.nan),
          'ss3': ss3_seq_target,
          'ss8': ss8_seq_target,
          'rsa_class': rsa_class_seq_target,
          'rsa_value': rsa_value_seq_target.tobytes(fill_value=np.nan),
          'seq_reweight': weight
        })
      except Exception as err:
        print_exception(err=err,level=enum_exception_level.ERROR,message=f'set:{set_name},seq_id:{family_nm}/{unp_id}/{unp_range_str}/{pdb_id}-{chain_id}')

    # start a new lmdb write transaction
    print(f'num: {len(set_feature_list)}')
    with env.begin(write=True) as txn:
      for i, entry in enumerate(set_feature_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    
    env.close()

def XToJson(
  inputPath: str = None,
  delimiter: str = ',',
  keyCol: int = 0,
  outputPath: str = None,
  scanLine: bool = True):
  """
  Convert tabular data to json format
  """
  outputJson = {}
  i = 0
  if scanLine:
    with open(f'{inputPath}') as handle:
      for line in handle:
        i += 1
        line = line.strip('\n')
        line_split = line.split(delimiter)
        keyVal = line_split[keyCol]
        if i % 1000 == 0:
          print(keyVal, flush=True)
        if keyVal in outputJson.keys():
          outputJson[keyVal].append(line)
        else:
          outputJson[keyVal] = [line]
  else:
    sourceData = np.loadtxt(inputPath,dtype='str',delimiter=delimiter)
    for i in range(len(sourceData)):
      i += 1
      keyVal = sourceData[i,keyCol]
      if i % 1000 == 0:
        print(keyVal, flush=True)
      if keyVal in outputJson.keys():
        outputJson[keyVal].append(sourceData[i])
      else:
        outputJson[keyVal] = [list(sourceData[i])]

  with open(f'{outputPath}.pickle', 'wb') as handle:
    pkl.dump(outputJson, handle, protocol=pkl.HIGHEST_PROTOCOL)

def seq_align_identity(x,y,matrix = BLOSUM62_MAT):
    X = x.upper()
    Y = y.upper()
    alignments = Bio.pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
    max_iden = 0
    for i in alignments:
        same = 0
        for j in range(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        iden = float(same)/float(i[-1])
        if iden > max_iden:
            max_iden = iden
    return max_iden


def check_exp_af_seq_iden(
      set_list: List,
      set_file: str,
      exp_feature_dir: str,
      af_feature_dir: str):

  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
  
  
  for set_name in set_list:
    af_seq_list = []
    af_seq_id_list = []
    exp_seq_list = []
    exp_seq_id_list = []
    if int(os.path.getsize(f'{af_feature_dir}/{set_name}.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{af_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          af_seq_list.append(item['seq_primary'])
          af_seq_id_list.append(item['seq_id'])
      input_env.close()
    if len(af_seq_list) == 0:
      continue  
    if int(os.path.getsize(f'{exp_feature_dir}/{set_name}.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          exp_seq_list.append(item['seq_primary'])
          exp_seq_id_list.append(item['seq_id'])
      input_env.close()

    ## check seq identity
    iden_noalign_count = 0
    iden_align_count = 0
    iden_pair_list = []
    for af_seq_i in range(len(af_seq_list)):
      af_seq = af_seq_list[af_seq_i]
      af_seq_id = af_seq_id_list[af_seq_i]
      exit_flag = False
      for exp_seq_i in range(len(exp_seq_list)):
        exp_seq = exp_seq_list[exp_seq_i]
        exp_seq_id = exp_seq_id_list[exp_seq_i]
        if af_seq == exp_seq:
          iden_noalign_count += 1
          exit_flag = True
          iden_pair_list.append([af_seq_id,exp_seq_id])
        #iden_score = seq_align_identity(af_seq,exp_seq)
        #if iden_score == 1.:
        #  iden_align_count += 1
        #  exit_flag = True
        #if exit_flag:
        #  break
    print(f'{set_name}, {iden_noalign_count} exp seqs same to af seqs')
    np.savetxt(f'data_process/structure/shin2021/seq_identity_check/{set_name}.csv',iden_pair_list,delimiter=',',fmt='%s')

def combine_exp_af_sets(
      set_list: List,
      set_file: str,
      exp_feature_dir: str,
      af_feature_dir: str,
      exp_af_joint_dir: str,
      split: str = None,
      set_info_file: str = None):
  """

  Args:
    split: split mode 'random', 'iden'
  """

  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')
  
  map_size = (1024 * 100) * (2 ** 20) # 100G

  for set_name in set_list:
    max_seq_len = 0
    set_1_ids, set_2_ids = [], []
    if os.path.isdir(f'{exp_feature_dir}/{set_name}.lmdb') and int(os.path.getsize(f'{exp_feature_dir}/{set_name}.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        set_1_num = num_examples
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          #e.g: UniRef100_A0A960MG78/1-231
          unp_id_range_str = '/'.join(item['seq_id'].split('/')[1:3])
          set_1_ids.append([f'UniRef100_{unp_id_range_str}',idx])
      input_env.close()

    if os.path.isdir(f'{af_feature_dir}/{set_name}.lmdb') and int(os.path.getsize(f'{af_feature_dir}/{set_name}.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{af_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        set_2_num = num_examples
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          unp_id_range_str = '/'.join(item['seq_id'].split('/')[1:3])
          set_2_ids.append([f'UniRef100_{unp_id_range_str}',idx])
      input_env.close()

    if not os.path.isdir(exp_af_joint_dir):
      os.mkdir(exp_af_joint_dir)
    print('Begin spliting>')
    if split is not None:
      if split == 'random':
        np.random.seed(seed=42)
        ## sample jointly
        #valid_set_idx = np.random.choice(len(all_data),size=max(1,int(np.floor(len(all_data)*0.1))),replace=False)
        
        valid_set_1_idx = np.random.choice(set_1_num,size=max(1,int(np.floor(set_1_num*0.1))),replace=False)
        valid_set_2_idx = np.random.choice(set_2_num,size=max(1,int(np.floor(set_2_num*0.1))),replace=False)
        
        train_env = lmdb.open(f"{exp_af_joint_dir}/{set_name}_train.lmdb", map_size=map_size)
        valid_env = lmdb.open(f"{exp_af_joint_dir}/{set_name}_valid.lmdb", map_size=map_size)
        print(f'{set_name}, num: {set_1_num+set_2_num}; train {set_1_num+set_2_num-len(valid_set_1_idx)-len(valid_set_2_idx)}; valid: {len(valid_set_1_idx)+len(valid_set_2_idx)}',flush=True)
        counter_train, counter_valid = 0, 0
        input_env_1 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_2 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_list = [input_env_1,input_env_2]
        valid_set_idx_list = [valid_set_1_idx, valid_set_2_idx]
        for set_i in range(2):
          input_env = input_env_list[set_i]
          valid_set_idx = valid_set_idx_list[set_i]
          with input_env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
            for idx in range(num_examples):
              item = pkl.loads(txn.get(str(idx).encode()))
              if item['seq_len'] > max_seq_len:
                max_seq_len = item['seq_len']
              if idx in valid_set_idx:
                with valid_env.begin(write=True) as txn:
                  txn.put(str(counter_valid).encode(), pkl.dumps(item))
                  counter_valid += 1
              else:
                with train_env.begin(write=True) as txn:
                  txn.put(str(counter_train).encode(), pkl.dumps(item))
                  counter_train += 1
        with valid_env.begin(write=True) as txn:
          txn.put(b'num_examples', pkl.dumps(counter_valid))
        with train_env.begin(write=True) as txn:
          txn.put(b'num_examples', pkl.dumps(counter_train))
        train_env.close()
        valid_env.close()
      elif split == 'iden': # valid set: iden in [0,0.8]
        # load identity file
        path,file_nm = os.path.split(set_info_file)
        set_info_df = pd.read_csv(set_info_file,header=0,delimiter=',')
        iden_file = set_info_df.loc[set_info_df['prot_name']==set_name,'identity_file_name'].iloc[0]
        iden2target_df = pd.read_csv(f'{path}/{iden_file}',header=0,delimiter=',') # id  identity_to_query
        set_1_ids_df = pd.DataFrame(set_1_ids,columns=['id','idx'])
        set_2_ids_df = pd.DataFrame(set_2_ids,columns=['id','idx'])
        filter_iden_df = iden2target_df.loc[(iden2target_df['identity_to_query'].astype(float) < 0.8) & (iden2target_df['identity_to_query'].astype(float) > 0.0),:]
        valid_set_1_df = set_1_ids_df.merge(filter_iden_df,on=['id'],how='inner').sample(n=int(len(set_1_ids_df)*0.1),random_state=42,ignore_index=True)
        valid_set_2_df = set_2_ids_df.merge(filter_iden_df,on=['id'],how='inner').sample(n=int(len(set_1_ids_df)*0.1),random_state=42,ignore_index=True)

        counter_train, counter_valid = 0, 0
        train_env = lmdb.open(f"{exp_af_joint_dir}/{set_name}_train.lmdb", map_size=map_size)
        valid_env = lmdb.open(f"{exp_af_joint_dir}/{set_name}_valid.lmdb", map_size=map_size)
        input_env_1 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_2 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_list = [input_env_1,input_env_2]
        valid_set_idx_list = [valid_set_1_df['idx'].tolist(), valid_set_2_df['idx'].tolist()]
        for set_i in range(2):
          input_env = input_env_list[set_i]
          valid_set_idx = valid_set_idx_list[set_i]
          with input_env.begin(write=False) as input_txn:
            num_examples = pkl.loads(input_txn.get(b'num_examples'))
            for idx in range(num_examples):
              if idx % 5000 == 0:
                print(f'idx: {idx}')
              item = pkl.loads(input_txn.get(str(idx).encode()))
              if item['seq_len'] > max_seq_len:
                max_seq_len = item['seq_len']
              if idx in valid_set_idx:
                with valid_env.begin(write=True) as valid_txn:
                  valid_txn.put(str(counter_valid).encode(), pkl.dumps(item))
                  counter_valid += 1
              else:
                with train_env.begin(write=True) as train_txn:
                  train_txn.put(str(counter_train).encode(), pkl.dumps(item))
                  counter_train += 1
        with valid_env.begin(write=True) as valid_txn:
          valid_txn.put(b'num_examples', pkl.dumps(counter_valid))
        with train_env.begin(write=True) as train_txn:
          train_txn.put(b'num_examples', pkl.dumps(counter_train))
        train_env.close()
        valid_env.close()
        print(f'{set_name}, num: {len(set_1_ids_df)+len(set_2_ids_df)}; train {counter_train}; valid: {counter_valid}',flush=True)
      else:
        out_env = lmdb.open(f"{exp_af_joint_dir}/{set_name}.lmdb", map_size=map_size)
        input_env_1 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_2 = lmdb.open(f'{exp_feature_dir}/{set_name}.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        input_env_list = [input_env_1,input_env_2]
        counter_num = 0
        for set_i in range(2):
          input_env = input_env_list[set_i]
          with input_env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
            for idx in range(num_examples):
              item = pkl.loads(txn.get(str(idx).encode()))
              if item['seq_len'] > max_seq_len:
                max_seq_len = item['seq_len']
              with out_env.begin(write=True) as txn:
                txn.put(str(counter_num).encode(), pkl.dumps(item))
                counter_num += 1
        with out_env.begin(write=True) as txn:
          txn.put(b'num_examples', pkl.dumps(counter_num))
        out_env.close()
        print(f'{set_name}, num: {counter_num}',flush=True)
    
    print(f'max_seq_len: {max_seq_len}',flush=True)
  return

def check_max_seq_len(fam_list: List,
                      data_dir: str):
  """query max seq length for each family
  """
  for fam in fam_list:
    max_seq_len = 0
    if os.path.isdir(f'{data_dir}/{fam}_train.lmdb') and int(os.path.getsize(f'{data_dir}/{fam}_train.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{data_dir}/{fam}_train.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          if item['seq_len'] > max_seq_len:
            max_seq_len = item['seq_len']
      input_env.close()
    
    if os.path.isdir(f'{data_dir}/{fam}_valid.lmdb') and int(os.path.getsize(f'{data_dir}/{fam}_valid.lmdb/data.mdb')) > 8192:
      input_env = lmdb.open(f'{data_dir}/{fam}_valid.lmdb', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          if item['seq_len'] > max_seq_len:
            max_seq_len = item['seq_len']
      input_env.close()
    print(f"{fam} max_seq_len: {max_seq_len}")

def config_files(path: str = "/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark",
                 af_version: int = 3):
  config_folder = "model_configs/seq_struct_multi_task"
  mp_layer_name = ['gatv2'] #None,'gatv2','gine','transfomer'

  loss_weights = {
    'aa_loss_weight':     [1.0], #, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    'ss_rsa_loss_weight': [20.0], #0.0, 0.01, 0.1, 0.33, 0.5, 1.0, 2.0, 5.0, 10.0, 0.0, 1.0
    'dist_loss_weight':   [20.0], #0.0, 0.01, 0.1, 0.33, 0.5, 1.0, 2.0, 5.0, 10.0, 1.0, 0.0
    'struct_loss_weight': [1.0] #0.0, 0.01, 0.1, 0.33, 0.5, 1.0, 2.0, 5.0, 10.0
  }
  
  fam_list = ['AMIE_PSEAE', 'DLG4_RAT', 'PABP_YEAST', 'RASH_HUMAN', 'KKA2_KLEPN', 'PTEN_HUMAN', 'MTH3_HAEAESTABILIZED', 'HIS7_YEAST', 'BRCA1_HUMAN_BRCT', 'BRCA1_HUMAN_RING', 'YAP1_HUMAN']
  #'AMIE_PSEAE', 'DLG4_RAT', 'PABP_YEAST', 'RASH_HUMAN', 'KKA2_KLEPN', 'PTEN_HUMAN', 'MTH3_HAEAESTABILIZED', 'HIS7_YEAST', 'BRCA1_HUMAN_BRCT', 'BRCA1_HUMAN_RING', 'YAP1_HUMAN'

  #'AMIE_PSEAE' 'B3VI55_LIPST' 'BF520_env' 'BG505_env' 'BG_STRSQ' 'BLAT_ECOLX' 'BRCA1_HUMAN_BRCT' 'BRCA1_HUMAN_RING' 'CALM1_HUMAN' 'DLG4_RAT' 'GAL4_YEAST' 'HG_FLU' 'HIS7_YEAST' 'HSP82_YEAST' 'IF1_ECOLI' 'KKA2_KLEPN' 'MK01_HUMAN' 'MTH3_HAEAESTABILIZED' 'TIM_THETH' 'PABP_YEAST' 'PA_FLU' 'POLG_HCVJF' 'PTEN_HUMAN' 'RASH_HUMAN' 'RL401_YEAST' 'SUMO1_HUMAN' 'TPK1_HUMAN' 'TPMT_HUMAN' 'TIM_SULSO' 'TIM_THEMA' 'UBC9_HUMAN' 'UBE4B_MOUSE' 'YAP1_HUMAN'

  use_class_weights = False
  freeze_bert_firstN_list = [0] #[0,0,0,0,0],12,12,6,4,4
  vocab_size = 29
  pretrain_mdl_list = ['rp15_all_2'] #'rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'
  bert_hidden_layers_list = [6] #12,12,6,4,4
  bert_attention_heads_list = [12] #12,12,12,8,8
  intermediate_size_list = [3072] #3072,3072,3072,3072,1024
  seq_max_length = 512
  freeze_mode_list = ['all'] #'all' 'selfAtten'
  distMap_bin = 2
  model_var = '_lm_multitask_symm'
  dist_first_cutoff = 8.0
  dist_last_cutoff = 8.0

  with open(f'{path}/{config_folder}/vanilla{model_var}.json','r') as f:
    config_json = json.load(f)
  
  for m in range(len(pretrain_mdl_list)):
    config_json['num_hidden_layers'] = bert_hidden_layers_list[m]
    config_json['num_attention_heads'] = bert_attention_heads_list[m]
    config_json['intermediate_size'] = intermediate_size_list[m]
    config_json['use_class_weights'] = use_class_weights
    config_json['freeze_bert_firstN'] = freeze_bert_firstN_list[m]
    #config_json['vocab_size'] = vocab_size
    config_json['freeze_mode'] = freeze_mode_list[m]
    config_json['num_dist_classes'] = distMap_bin
    config_json['dist_first_cutoff'] = dist_first_cutoff
    config_json['dist_last_cutoff'] = dist_last_cutoff
    for fam_name in fam_list:
      if af_version == 2:
        #max_seq_len_dict[fam_name]+10
        config_json['seq_max_length'] = seq_max_length
      elif af_version == 3:
        #min(max_seq_len_dict_v3[fam_name],256)
        config_json['seq_max_length'] = seq_max_length+2
      else:
        Exception(f'invalid af version number {af_version}')
      for mp_l in mp_layer_name:
        #config_json['mp_layer_name'] = mp_l
        if mp_l is None:
          mp_l = 'noMP'
        # with open(f'{path}/{config_folder}/{fam_name}_{mp_l}_CW.json','w') as f:
        #           json.dump(config_json,f)
        #'''
        for w_i in range(len(loss_weights['aa_loss_weight'])):
          aa_w = loss_weights['aa_loss_weight'][w_i]
          ss_w = loss_weights['ss_rsa_loss_weight'][w_i]
          rsa_w = loss_weights['ss_rsa_loss_weight'][w_i]
          dist_w = loss_weights['dist_loss_weight'][w_i]
          config_json['aa_loss_weight'] = aa_w
          config_json['ss_loss_weight'] = ss_w
          config_json['rsa_loss_weight'] = rsa_w
          config_json['dist_loss_weight'] = dist_w
          if af_version == 2:
            #_{mp_l}
            with open(f'{path}/{config_folder}/{pretrain_mdl_list[m]}_aa{aa_w}_ss{ss_w}_rsa{rsa_w}_dist{dist_w}.json','w') as f:
              json.dump(config_json,f)
          elif af_version == 3:
            #_{mp_l}
            with open(f'{path}/{config_folder}/{pretrain_mdl_list[m]}_aa{aa_w}_ss{ss_w}_rsa{rsa_w}_dist{dist_w}_bertFRZF{freeze_bert_firstN_list[m]}_FRZMode{freeze_mode_list[m]}_AFDBv3_seqMaxL{seq_max_length}_distBin{distMap_bin}{model_var}.json','w') as f:
              json.dump(config_json,f)
          else:
            Exception(f'invalid af version number {af_version}')
        #'''
  return None

def wt_protein_structure_info(
      set_list: List,
      set_file: str = None,
      afDB_version: int = None,
      struct_format: str = 'mmCif',
      use_exp_struct: bool = False,
      use_AFDB_struct: bool = False,
      use_AF_pred_struct: bool = False,
      logger: logging.Logger = None):
  """Generate structure property information for WT protein of each family

  """
  root_path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  ## read family list to process
  if set_list is None:
    set_list = np.loadtxt(f'{set_file}',dtype='str',delimiter='\t')

  ## loop over family list
  for setI in range(len(set_list)):
    setNm = set_list[setI]
    ## load mapping and search targets
    if use_exp_struct:
      set_feature_list = []
      unp_expPdb_map_df = pd.read_csv(f'{root_path}/data_process/structure/shin2021/unp_pdb_map_afv2/{setNm}_unp_expPDB_map.tsv',sep='\t',header=None,names=['unp_name','identifier','pdb_entity','pdb_instance'])
      ## load seq fasta file (single wt seq)
      with open(f'{root_path}/data_process/mutagenesis/DeepSequenceMutaSet_priSeq/{setNm}.fasta') as handle:
        for record in SeqIO.parse(handle,"fasta"):
          seq_str = str(record.seq)
          seqId_split = record.id.split('|')
          target_unp_acc = seqId_split[1]
          target_seq_start,target_seq_end = seqId_split[-1].split('-')
          target_expPdb_rows = unp_expPdb_map_df.loc[unp_expPdb_map_df['identifier'].str.contains(target_unp_acc)]
          target_exp_pdbs = target_expPdb_rows['pdb_instance'].drop_duplicates()
          if len(target_expPdb_rows) == 0:
            print(f'{setNm}/{target_unp_acc}/{target_seq_start}-{target_seq_end}: no exp pdbs found')
          else:
            for idx, row in target_exp_pdbs.items():
              print(f"{setNm}/{target_unp_acc}/{target_seq_start}-{target_seq_end} exp:{row}")
    if use_AFDB_struct:
      ## Create a new LMDB environment
      map_size = (1024 * 20) * (2 ** 20) # 100G
      env = lmdb.open(f"{root_path}/data_process/mutagenesis/wt_seq_structure/set_structure_data/{set_name}_AFDB.lmdb", map_size=map_size)
      set_feature_list = []
      unp_afdbPdb_map_df = pd.read_csv(f'{root_path}/data_process/structure/shin2021/unp_pdb_map_afv3/{setNm}_unp_afPDB_map.tsv',sep='\t',header=None,names=['unp_name','identifier','unp_acc','afdb_id'])
      ## load seq fasta file (single wt seq)
      with open(f'{root_path}/data_process/mutagenesis/DeepSequenceMutaSet_priSeq/{setNm}.fasta') as handle:
        for record in SeqIO.parse(handle,"fasta"):
          seq_str = str(record.seq)
          seqId_split = record.id.split('|')
          target_unp_acc = seqId_split[1]
          target_seq_start,target_seq_end = seqId_split[-1].split('-')
          target_afdbPdb_rows = unp_afdbPdb_map_df.loc[unp_afdbPdb_map_df['unp_acc'].str.contains(target_unp_acc)]
          target_afdb_pdbs = target_afdbPdb_rows['afdb_id'].drop_duplicates()
          if len(target_afdbPdb_rows) == 0:
            print(f'{setNm}/{target_unp_acc}/{target_seq_start}-{target_seq_end}: no AFDB pdbs found')
          else:
            # for idx, row in target_exp_pdbs.items():
            #   print(f"{setNm}/{target_unp_acc}/{target_seq_start}-{target_seq_end} afdb:{row}")
            # only one pdb match in AFDB
            target_pdb_afdb_id = target_afdb_pdbs.iloc[0]
            try:
              af_id, af_range_str = target_pdb_afdb_id.split(':')
              af_start, af_end = af_range_str.split('-')
              range_start = min(int(target_seq_start),int(af_start))
              pdb_seq_target, contact_map_target, distance_map_target, \
              ss3_seq_target, ss8_seq_target, rsa_class_seq_target, \
              rsa_value_seq_target = alphafold_struct_feature(
                                      family_name=setNm,
                                      af_id=af_id,
                                      afDB_version=afDB_version,
                                      unp_id=target_unp_acc,
                                      struct_format='mmCif',
                                      start_idx=int(target_seq_start),
                                      end_idx=int(target_seq_end),
                                      af_start_idx=int(af_start),
                                      af_end_idx=int(af_end))
              ## use seq in fasta instead of pdb
              seq_len = len(seq_str)
              
              assert distance_map_target.shape[0] == seq_len
              if np.sum(~distance_map_target.mask) == 0:
                continue
              assert contact_map_target.shape[0] == seq_len
              assert np.sum(~contact_map_target.mask) > 0
              assert len(ss3_seq_target) == seq_len
              assert len (ss8_seq_target) == seq_len
              assert len(rsa_class_seq_target) == seq_len
              assert rsa_value_seq_target.shape[0] == seq_len
              assert np.sum(~rsa_value_seq_target.mask) > 0

              for repeat_i in range(16):
                set_feature_list.append({
                  'seq_id': f'{setNm}/{target_unp_acc}/{seqId_split[-1]}/{af_id}',
                  'seq_primary': seq_str,
                  'seq_len': seq_len,
                  'distance_map': distance_map_target.tobytes(fill_value=np.nan),
                  'ss3': ss3_seq_target,
                  'ss8': ss8_seq_target,
                  'rsa_class': rsa_class_seq_target,
                  'rsa_value': rsa_value_seq_target.tobytes(fill_value=np.nan)
                })
              #'seq_reweight': None
            except Exception as err:
              print_exception(err=err,level=enum_exception_level.ERROR,message=f'set:{setNm},seq_id:{setNm}/{target_unp_acc}/{seqId_split[-1]}/{af_id}')
        
      # start a new lmdb write transaction
      print(f'num: {len(set_feature_list)}')
      with env.begin(write=True) as txn:
        for i, entry in enumerate(set_feature_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      env.close()
    if use_AF_pred_struct:
      set_feature_list = []
      unp_afPredPdb_map_df = pd.read_csv(f'{root_path}/data_process/structure/shin2021/unp_pdb_map_afv3/{setNm}_unp_afPredPDB_map.tsv')
      ## load seq fasta file (single wt seq)
      with open(f'{root_path}/data_process/mutagenesis/DeepSequenceMutaSet_priSeq/{setNm}.fasta') as handle:
        for record in SeqIO.parse(handle,"fasta"):
          seq_str = str(record.seq)
          seqId_split = record.id.split('|')
          target_unp_acc = seqId_split[1]
          target_seq_start,target_seq_end = seqId_split[-1].split('-')

  return None

def structure_eval_config_files():
  """modification of config files for structure awareness evaluation task
  
  """
  path = "/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/model_configs/structure_eval"
  label_list = ['aa', 'ss', 'rsa', 'distMap']
  eval_set_list = ['wt', 'valid']
  for label_type in label_list:
    with open(f'{path}/{label_type}_head.json','r') as f:
      config_json = json.load(f)
    for eval_set in eval_set_list:
      eval_phase=True
      if eval_set == 'wt':
        multi_copy_num=32
      elif eval_set == 'valid':
        multi_copy_num=1
      
      config_json['multi_copy_num'] = multi_copy_num
      config_json['eval_phase'] = eval_phase
      
      with open(f'{path}/{label_type}_head_eval_{eval_set}.json','w') as f:
        json.dump(config_json,f)

  return None

if __name__ == "__main__":
  set_name = sys.argv[1]
  #set_name = 'test'
  ### setup logger ###
  log_file = f'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/dataProcess_scripts/exception/exception_{set_name}.out'
  if os.path.isfile(log_file):
    os.remove(log_file)
  logging.basicConfig(filename=log_file, level=logging.DEBUG, \
                    format="[%(asctime)s:%(levelname)s:%(funcName)10s] %(message)s")
  logger=logging.getLogger(__name__)
  logging.getLogger("requests").setLevel(logging.WARNING)
  logging.getLogger("urllib3").setLevel(logging.WARNING)

  ### setup requests session ###
  req_sess = requests.Session()
  retries = Retry(total=5,
                  backoff_factor=0,
                  status_forcelist=[104, 429, 500, 502, 503, 504])
  req_sess.mount('https://', HTTPAdapter(max_retries=retries))

  ### setup dirs ###
  root_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  pdb_download_dir = f'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/download_structures/{set_name}'
  task2run = sys.argv[2]
  if task2run == 'query_struct_ShinData_v2':
    query_struct_ShinData(
        set_list = [set_name],
        set_file = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
        unpId_batch_size=10000,
        afDB_version = 2,
        afAccessionPath = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/AF_StructureDB/accession_ids_v2.pickle',
        download_struct = True,
        struct_format = 'mmCif',
        check_exp_struct = True,
        check_AF_struct = True,
        unp_pdb_map_path ='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_afv2',
        logger=logger)
  elif task2run == 'query_struct_ShinData_v3':
    query_struct_ShinData(
        set_list = [set_name],
        set_file = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
        unpId_batch_size=10000,
        afDB_version = 3,
        afAccessionPath = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/AF_StructureDB/accession_ids_v3.csv',
        download_struct = True,
        struct_format = 'mmCif',
        check_exp_struct = False,
        check_AF_struct = True,
        unp_pdb_map_path ='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_afv3',
        logger=logger)
  elif task2run == 'generate_feature_ShinData':
    '''
    generate_expStruct_feature_ShinData(
        set_list=[set_name],
        set_file=None,#'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
        unp_pdb_map_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_afv2',
        feature_save_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_structure_features',
        min_len=10)
    '''
    generate_afStruct_feature_ShinData(
        set_list=[set_name],
        set_file=None,#'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
        unp_pdb_map_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_afv3',
        feature_save_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/af_structure_features_v3',
        min_len=10,
        afDB_version=3)
    combine_exp_af_sets(
      set_list=[set_name],
      set_file=None, #'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
      exp_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_structure_features',
      af_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/af_structure_features_v3',
      exp_af_joint_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_afv3_structure_features',
      split='random')
  elif task2run == 'generate_afStruct_feature_ShinData_multithread':
    generate_afStruct_feature_ShinData_multithread(
        set_list=[set_name],
        set_file=None,#'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
        unp_pdb_map_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_afv3',
        feature_save_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/af_structure_features_v3',
        min_len=10,
        afDB_version=3,
        num_workers=24)
    combine_exp_af_sets(
      set_list=[set_name],
      set_file=None, #'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
      exp_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_structure_features',
      af_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/af_structure_features_v3',
      exp_af_joint_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_afv3_structure_features',
      split=None)
  elif task2run == 'generate_structure_feature':
    root_path = '/scratch/user/sunyuanfei/Projects/CAGI6'
    combine_exp_af_sets(
      set_list=[set_name],
      set_file=None, #'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
      exp_feature_dir=f'{root_path}/structure_features/exp_afv3_structure_features',
      af_feature_dir=f'{root_path}/structure_features/seq_wo_structure_features',
      exp_af_joint_dir=f'{root_path}/structure_features/all_seq_structure_features',
      split='iden',
      set_info_file=f'{root_path}/ev_align/sequence_list.csv')
  elif task2run == 'check_max_seq_len':
    check_max_seq_len(fam_list = ['AMIE_PSEAE','HIS7_YEAST','KKA2_KLEPN','MTH3_HAEAESTABILIZED','PTEN_HUMAN','RASH_HUMAN'],
                      data_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_afv3_structure_features')
  elif task2run == 'config_files':
    config_files(path=root_dir,af_version=3)
  elif task2run == 'structure_eval_config_files':
    structure_eval_config_files()
  elif task2run == 'wt_protein_structure_info':
    wt_protein_structure_info(
      set_list = [set_name],
      afDB_version = 3,
      struct_format = 'mmCif',
      use_exp_struct = False,
      use_AFDB_struct = True,
      use_AF_pred_struct = False,
      logger = logger)
  else:
    print(f'invalid task {task2run}')
  '''
  a,b,c,d,e,f,g = exp_struct_feature(
                family_name='UBC9_HUMAN',
                pdb_id='5KHR',
                unp_id='O00762',
                struct_format='mmCif',
                pdbChain_id='P',
                start_idx=33,
                end_idx=172)
  '''
  '''
  alphafold_struct_feature(
      family_name='AMIE_PSEAE',
      af_id='AF-P11436-F1',
      unp_id='P11436',
      struct_format='mmCif',
      start_idx=1,
      end_idx=346,
      af_start_idx=1,
      af_end_idx=346)
  '''
  '''
  XToJson(inputPath='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/AF_StructureDB/accession_ids_v3.csv',
          delimiter=',',
          keyCol=0,
          outputPath='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/AF_StructureDB/accession_ids_v3',
          scanLine=True)
  '''
  '''
  check_exp_af_seq_iden(
      set_list=[sys.argv[1]],
      set_file=None,#'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames',
      exp_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/exp_structure_features',
      af_feature_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/af_structure_features')
  '''
  

  #seq_align_identity('aasderfgh','asderf')
  #dist_map, cont_map = test_singleProtein_distMat()
  #print(dist_map.count(), cont_map.count())
  
  # ss_dict = read_ss_from_file(
  #               struct_file = f'{root_dir}data_process/structure/shin2021/download_structures/AMIE_PSEAE/7ogu.cif',
  #               file_format = 'mmCif')

  #threeIdxSetMap, unp_seq, pdb_seq, auth_chain_id, entity_id = get_index_mapping(pdb_id='6P65',chain_id='D',unpAcc='A1EAI1')
