# Table of **Contents**
- [Installation](#installation)
- [Data and model Availability](#data-and-model-availability)
    - [Data](#data)
    - [Model](#model-weights)
- [Tasks](#tasks)
    - [Sequence embedding](#sequence-embedding)
    - [Variant fitness](#variant-fitness)
    - [Structure-informed FT](#structure-informed-finetuning)

# Installation
Use `conda` command to set up the environment with necessary dependencies.

```shell
conda env create -f environment.yml
```
We support cross-machine parallel training and mixed percision training for our models. You need to install [Apex](https://github.com/NVIDIA/apex/tree/master?tab=readme-ov-file#from-source) in order to use these features. Please install Apex with CUDA and C++ extensions. GCC library is required for compiling Apex. We recommend version 9.3.0 and 10.3.0 which have been tested working.

*Updates:* The latest [master](https://github.com/NVIDIA/apex/tree/master?tab=readme-ov-file) branch may give failures in setting up cpp extensions and cuda extensions. Check this [issue](https://github.com/NVIDIA/apex/issues/1737#issuecomment-1762662648) from Apex for a potential solution.

# Data and model Availability
Processed data in LMDB format and model weights can be downlaoded from [Zenodo](https://zenodo.org/records/8197882)
#### Data:
* domain sequences for pretraining
* family homologous sequences for sequence-only finetuning
* family sequences+structures for structure-informed finetuning
* fitness labels from DMS assays
#### Model weights:
* pretrained meta models
* sequence-only finetuned models
* structure-informed finetuned models

### Meta model architecture hyperparams
|model id|training set|# layers|# heads|hidden size| FF  |
|--------- | --------- | -------| ----  | -------   |--- |
|RP75_B1   | pfam_rp75 | 12  | 12    | 768       |3072|
|RP15_B1   | pfam_rp15 | 12  | 12    | 768       |3072|
|RP15_B2   | pfam_rp15 | 6   | 12    | 768       |3072|
|RP15_B3   | pfam_rp15 | 4   | 8     | 768       |3072|
|RP15_B4   | pfam_rp15 | 4   | 8     | 768       |1024|

# Tasks

## Sequence embedding
1. Prepare sequence data

For each sequence, save the information of identifier and amino acids in the dictionary format, follow the example below. And append all such dictionaries to a List (e.g. named data_list).
```text
{'seq_id': 'protein_seq_1' # change to your identifier, unique for each sequence
 'seq_primary': 'VQLVQSGAAVKKPGESLRISCKGSGYIFTNYWINWVRQMPGRGLEWMGRIDPSDSYTNYSSSFQGHVTISADKSISTVYLQWRSLKDTDTAMYYCARLGSTA' # string, upper cases
 }
```

Then, use the following code to save into lmdb format.
```python
import lmdb
import pickle as pkl

map_size = (1024 * 15) * (2 ** 20) # 15G, change accordingly
wrtEnv = lmdb.open('/path/to/dataset/data_file_name.lmdb',map_size=map_size)
with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data_list): # data_list contains all dictionaries in the above format
        txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
wrtEnv.close()
```

2. Run model

Run the following command to generate embeddings. If you use any relative path inputs for some parameters, make sure it is visible under the 'ProteinEncoder-LM/' folder. Please change parameters in '{}' accordingly (delete '{}' and comments afterwards)

```python
python scripts/main.py \
    run_eval \
    transformer \
    embed_seq \
    {'/path/to/saved_model'}  (e.g. 'trained_models/rp15_pretrain_1_models') \
    --batch_size {4} (change accordingly) \
    --data_dir {'/path/to/dataset'} \
    --metrics save_embedding \
    --split {'data_file_name'} \
    --embed_modelNm {'customized identifier for the model'} (e.g. 'rp15_pretrain_1')
```
For a sequence of length L, the final embeddings have size L * 768. After successful running of the python command, a json file 'embedding_{data_file_name}_{embed_modelNm}.json' should appear under the folder '/path/to/dataset'. Sequence identifiers and embeddings are organized in the dictionary format below.
```text
{seq_id : embedding_matrix} // seq_id is the given identifier for the sequence, embedding_matrix is a list with size L * 768.
```

## Variant fitness
0. Information
* Log ratio of likelihood: $\textup{log}_{e} \frac{p(\textup{mut})}{p(\textup{wt})}$ is used as fitness prediction. More positive value means better than WT and more negative value means worse than WT.

1. Prepare mutation data

For each mutation, follow the example below to save information in dictionary, then append all such dictionaries to a List (e.g. named data_list).
```text
{'set_nm': 'mutation_set_1', # mutation set name
 'wt_seq': 'VQLVQSGAAVKKPGESLRIS', # WT seq (upper cases)
 'mut_seq':'SQLVQSGADVKKPGESLRIS', # Mut seq (upper cases),
 'mutants': ['V10S','A18D'], # list of mutant names (missense mutations only)
 'mut_relative_idxs' [0,8], # list of mutant indices, relative to the wt_seq (0-based)
 'fitness': 1.5 # fitness score. If given, will calcualte Spearman's r and Pearson's r; use 0 if unknown
}
```

Then, use the following code to save into lmdb format.
```python
import lmdb
import pickle as pkl

map_size = (1024 * 15) * (2 ** 20) # 15G, change accordingly
wrtEnv = lmdb.open('/path/to/dataset/data_file_name.lmdb',map_size=map_size)
with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(data_list): # data_list contains all dictionaries in the above format
        txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
wrtEnv.close()
```

2. Run models
Run the following command to predict mutation fitness. If you use any relative path inputs for some parameters, make sure it is visible under the 'ProteinEncoder-LM/' folder. Please change parameters in '{}' accordingly (delete '{}' and comments afterwards)

```python
python scripts/main.py \
    run_eval \
    transformer \
    mutation_fitness_UNsupervise_mutagenesis \
    {'/path/to/saved_model'}  (e.g. 'trained_models/rp15_pretrain_1_models') \
    --batch_size {4} (change accordingly) \
    --data_dir {'/path/to/dataset'} \
    --metrics fitness_unsupervise_mutagenesis \
    --mutgsis_set {'data_file_name'} (e.g. 'mutation_set_1') \
    --embed_modelNm {'customized identifier for the model'} (e.g. 'rp15_pretrain_1')
```

If run command successfully, a csv file named '{data_file_name}_{embed_modelNm}_predictions.csv' should appear under the folder '/path/to/dataset' which contains predicted fitness scores for each mutation. If groundtruth fitness scores are given, a json file named '{data_file_name}_{embed_modelNm}_metrics.json' will be generated which contains metric values.


## Structure-informed finetuning
Required packages:
* [DSSP](https://ssbio.readthedocs.io/en/latest/instructions/dssp.html#dssp), run `mkdssp -h` to verify.
* [MMseq2](https://github.com/soedinglab/MMseqs2), run `mmseqs easy-linclust -h` to verify.
* [HMMER](http://hmmer.org/documentation.html), run `esl-reformat -h` to verify.

Required database files:
* `accession_ids.csv` from [ftp](https://ftp.ebi.ac.uk/pub/databases/alphafold/) of Alphafold Database
* `batch_download.sh` from [RCSB](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script)

### Training data generation
Given a target protein sequence, the first step is to query its homologous sequences using [EVcouplings](https://github.com/debbiemarkslab/EVcouplings?tab=readme-ov-file#evcouplings), please follow instructions on their Github repo. It is recommended to explore multiple bit score thresholds and select the best one as ellaborated in Appendix.C of [Tranception](https://arxiv.org/abs/2205.13760).

With the acquired MSA, generate training data of sequences and structures with the following command

```python
python
```
