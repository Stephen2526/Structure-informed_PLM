# Setup Environment
Please check steps in __setup_env.sh__ to set up the environment. Some variables need to changed according to user's preference.

# Protein Sequence-based Language Modeling

### Download pretrained models
Before running specific tasks, download protein pretrained language models from [google drive](https://drive.google.com/file/d/1FZewUpVQ2jJL_Hg5NyFM6qGb4exJ2SRr/view?usp=share_link), decompress and save under the main folder ProteinEncoder-LM/

### Model architecture hyperparameters
|model name|training data|# layers|# heads|hidden size|FF  |
|--------- | ----------- | -------| ----  | -------   |--- |
|rp75_pretrain_1| pfam_rp75 | 12  | 12    | 768       |3072|
|rp15_pretrain_1| pfam_rp15 | 12  | 12    | 768       |3072|
|rp15_pretrain_2| pfam_rp15 | 6   | 12    | 768       |3072|
|rp15_pretrain_3| pfam_rp15 | 4   | 8     | 768       |3072|
|rp15_pretrain_4| pfam_rp15 | 4   | 8     | 768       |1024|


## Compute Residue Embeddings for Protein Sequences
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

## Predict mutation fitness scores
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


## Sequence-only fine-tuning

## Sequence+structure joint fine-tuning