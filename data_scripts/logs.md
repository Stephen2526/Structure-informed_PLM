## Process Pfam-A.rp** files
* make a folder to store individual msa files
```console
mkdir Pfam-A.rp**_seqs
```
* split the whole Pfam-A.** MSA file into msa file of each family in stockholm format (NEED '^' in pattern)
```console
mkdir -p Pfam-A.full.uniprot
gzip -dc Pfam-A.full.uniprot.gz | csplit -z --digits=5 --prefix=Pfam-A.full.uniprot/msa_fam_ - /^\/\//+1 '{*}'
for flNm in Pfam-A.full.uniprot/msa_fam_*; 
do
  acc=$(grep \#\=GF\ AC ${flNm} | cut -d' ' -f3);
  acc_noVersion=$(echo ${acc} | cut -d'.' -f1) 
  mv $flNm Pfam-A.full.uniprot/${acc_noVersion}.sth
done
```

* (check file encoding scheme)
```console
file -i [file_name]
```
* (convert every msa file to utf8 encoding)
```console
for f in $(ls Pfam-A.rp**_seqs);
do
  iconv -f iso88591 -t utf8 [Pfam-A.rp**_seqs/${f}] > [Pfam-A.rp**_seqs_utf8/${f}]
done
```
* extract protein sequences from each msa file (STOCKHOLM format) and concatenate them into one single fasta file
```console
python dataProcess_scripts/pfam_rp_proc.py
```
* first fasta to json, then json to tfRecord
```console
python dataProcess_scripts/pfam_pretrain_memory_wise.py
```
* compress folder containing seq files
```console
tar -zcvf [Pfam-A.rp**_seqs.tar.gz] [Pfam-A.rp**_seqs]
```
* extract back files to a dir
```console
tar -zxvf [Pfam-A.rp**_seqs.tar.gz] [-C [Pfam-A.rp**_seqs]]
```
* generate an index file with msa-file name and its family acc number
```console
for flNm in pfam_rp15_seqs_utf8/*; 
do
  echo $flNm;
  acc=$(grep \#\=GF\ AC ${flNm} | cut -d' ' -f3);
  acc_noVersion=$(echo ${acc} | cut -d'.' -f1) 
  clan=$(grep ${acc_noVersion} Pfam-A.clans.tsv | cut -d$'\t' -f2)
  echo "${flNm} ${acc} ${clan}" >> pfam_rp15_seqs_files_famAcc
done
tr -s " " !squeeze multiple continuous spaces to one space
``` 

## Process Pfam-A.fasta file
* split the file to individual fasta files with one family per file
```console
python dataProcess_scripts/pfam_splitFasta.py
```
