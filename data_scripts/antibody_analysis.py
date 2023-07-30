from sklearn.manifold import TSNE
from typing import List
import json,os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

def draw_tSNE(pretrain_set_list: List = None,
              set2draw: List = ['sabdab/sabdabABData','exp_data/exp_batch1','exp_data/exp_batch2'],
              seq_straty_list: List = ['seqConcate','seqIndiv'],
              mlm_straty_list: List = ['vanilla','cdr_vanilla','cdr_margin','cdr_pair'],
              loss_straty_list: List = ['mlm_only','mlm_gene'],
              loadDf_bool: bool = False):
  #large_subCPair = [87,86,93,85,40,45,77,38,24,69,109,41,46,101,42,122,79,103,98,74]  #sabdab
  #large_subCPair = []
  #large_subCPair = [24,38,40,45,46,47,69,73,74,77,85,86,87,93,96,101]
  
  ## load inhibition score of 25 pairs in exp_batch1
  b1_data = np.loadtxt('/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/antibody/exp_data/exp_batch1/idx_barcode_bindInhibit.csv',dtype='str',delimiter=',')
  b1BarInhibi_dict = {}
  for i in range(b1_data.shape[0]):
    barCode = b1_data[i,1]
    inhibiScore = float(b1_data[i,2])
    b1BarInhibi_dict[barCode] = inhibiScore

  for pretrain_set in pretrain_set_list:
    pretrain_set_split = re.split(r'_',pretrain_set)
    rpNm = f'{pretrain_set_split[0]}_{pretrain_set_split[1]}'
    mdl_subIdx = f'{pretrain_set_split[2]}'
    ini_epoch = f'{pretrain_set_split[3]}'
    for seq_straty in seq_straty_list:
      for mlm_straty in mlm_straty_list:
        for loss_straty in loss_straty_list:
          if not loadDf_bool:
            hidden_embeds_lastLayer = [] # both VH and VL
            extraInfo_list = []
            ## load embedding
            for setNm in set2draw:
              print(f'>>loading embedding data from {setNm}')
              with open(f'{setNm}/dt4tSNE/{pretrain_set}/{seq_straty}_{mlm_straty}_{loss_straty}.json') as jfl:
                dt_load = json.load(jfl)
              for i in range(len(dt_load)):
                hidden_states_lastLayer_token_VH = np.array(dt_load[i]['hidden_states_lastLayer_token_VH']) # [l_h,768]
                hidden_states_lastLayer_token_VL = np.array(dt_load[i]['hidden_states_lastLayer_token_VL']) # [l_l,768]
                if hidden_states_lastLayer_token_VH.shape[1] != 768:
                  print(hidden_states_lastLayer_token_VH.shape)
                  input()
                hidden_embeds_lastLayer.append(np.mean(np.concatenate((hidden_states_lastLayer_token_VH,hidden_states_lastLayer_token_VL), axis=0), axis=0)) # [768,]
                subCPair_ori = int(dt_load[i]['subClass_pair'])
                ## query binding inhibition score
                if setNm == 'exp_data/exp_batch1':
                  barCode = re.split(r'_',dt_load[i]['entityH'])[2]
                  if b1BarInhibi_dict.get(barCode) is not None:
                    bindInScore = b1BarInhibi_dict[barCode]
                    hasBindIn = 1
                  else:
                    bindInScore = 0.
                    hasBindIn = 0
                else:
                  bindInScore = 0.
                  hasBindIn = 0

                '''
                if subCPair_ori in large_subCPair:
                  subCPair_new = subCPair_ori
                else:
                  subCPair_new = -1
                '''
                #extraInfo_list.append([subCPair_new,setNm])
                extraInfo_list.append([dt_load[i]['entityH'],dt_load[i]['entityL'],subCPair_ori,setNm,bindInScore,hasBindIn])
            hidden_embeds_lastLayer = np.array(hidden_embeds_lastLayer) # [n_example(three sets),768]
            extraInfo_list = np.array(extraInfo_list) # [n_example(three sets),6]
            #print(hidden_embeds_lastLayer.shape)
            #print(extraInfo_list.shape)
            ## run tSNE
            print('>>running tSNE')
            tsne = TSNE(n_components = 2)
            hidden_hat = tsne.fit_transform(hidden_embeds_lastLayer) # [n_example(three sets),2]
            #print(hidden_hat.shape)
            data_list = np.concatenate((hidden_hat,extraInfo_list),axis=1)
            print(data_list.shape)
            # save to dataframe
            df = pd.DataFrame(data_list,columns=['tSNE1_lastLayer','tSNE2_lastLayer','entityH','entityL','subClassPair','setNm','bindInhibiScore','hasBindIn'])
            df = df.astype(dtype={'tSNE1_lastLayer':'float','tSNE2_lastLayer':'float','entityH':'str','entityL':'str','subClassPair':'int','setNm':'str','bindInhibiScore':'float','hasBindIn':'int'})
            print(df.dtypes)
            ## save dataframe to file
            df.to_pickle(f'savedDataFrame/dataframe4tSNE_{rpNm}_{mdl_subIdx}_{ini_epoch}_{seq_straty}_{mlm_straty}_{loss_straty}.pkl')
          else:
            df=pd.read_pickle(f'savedDataFrame/dataframe4tSNE_{rpNm}_{mdl_subIdx}_{ini_epoch}_{seq_straty}_{mlm_straty}_{loss_straty}.pkl')
            #df_filter = df[(df['hasBindIn']==1)]
          ## plot 
          '''
          fig1,ax1 = plt.subplots()
          markers = ['o','^','s']
          for i in range(len(set2draw)):
            setNm = set2draw[i]
            df_filter = df.loc[(df["setNm"]==setNm)]
            xx=df['tSNE1'].to_numpy().astype(float)
            yy=df['tSNE2'].to_numpy().astype(float)
            subC = df['subClassPair'].to_numpy().astype(int)
            sc=plt.scatter(x=xx,y=yy,marker=markers[i],label=setNm,s=10,alpha=0.5)
          plt.legend()
          #plt.colorbar(sc)
          '''
          ## sabdab, batch1, batch2
          sns.scatterplot(data=df,x='tSNE1_lastLayer',y='tSNE2_lastLayer',hue='setNm',style='hasBindIn',legend="full")
          # Put the legend out of the figure
          lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
          plt.savefig(f'tSNE/tSNE_SB12_setNm_hasBindIn_{rpNm}_{mdl_subIdx}_{ini_epoch}_{seq_straty}_{mlm_straty}_{loss_straty}.png',bbox_inches='tight',dpi=600)
          '''
          ## batch1, batch2
          ## * samples without score vs samples with scores
          df_fil_noBI = df[((df['setNm']=='exp_data/exp_batch1') | (df['setNm']=='exp_data/exp_batch2')) & (df['hasBindIn'] == 0)]
          df_fil_hasBI = df[(df['hasBindIn'] == 1)]
          sns.set_style("whitegrid") 
          #g = sns.FacetGrid(df_fil,hue='hasBindIn',height=4, aspect=1.25)
          #g.map_dataframe(sns.scatterplot, x="tSNE1_lastLayer",y="tSNE2_lastLayer",alpha=.5,style='setNm',legend="full")
          ax=sns.scatterplot(data=df_fil_noBI,x='tSNE1_lastLayer',y='tSNE2_lastLayer',style='setNm',hue='hasBindIn',palette=['gray'])
          sns.scatterplot(data=df_fil_hasBI,x='tSNE1_lastLayer',y='tSNE2_lastLayer',hue='bindInhibiScore',style='hasBindIn',markers=['X'],palette=sns.color_palette("coolwarm", as_cmap=True),legend="full",ax=ax)
          
          # Put the legend out of the figure
          lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
          #g.add_legend()
          
          ## add colorbar
          #norm = plt.Normalize(df_fil_hasBI['hasBindIn'].min(), df_fil_hasBI['hasBindIn'].max())
          #sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
          #sm.set_array([])

          # Remove the legend and add a colorbar
          #ax.get_legend().remove()
          #ax.figure.colorbar(sm)
          plt.savefig(f'tSNE/tSNE_B12_setNm_hasBindIn_colorPoint_{rpNm}_{mdl_subIdx}_{ini_epoch}_{seq_straty}_{mlm_straty}_{loss_straty}.png',bbox_inches='tight',dpi=600)
          '''
  return None

def metrics_analysis(rp_set: str = 'rp15_all',
                     mdl_subIdx: str = '1',
                     ini_epoch: str = '700',
                     set2analysis: List = ['sabdabABData','exp_batch1','exp_batch2'],
                     seq_straty_list: List = ['seqConcate','seqIndiv'],
                     mlm_straty_list: List = ['vanilla','cdr_vanilla','cdr_margin','cdr_pair'],
                     loss_straty_list: List = ['mlm_only','mlm_gene'],
                     seed_list: List = None):
  """
  * report mean/std across training tracks
  """
  for loss_straty in loss_straty_list:
    for seq_straty in seq_straty_list:
      for mlm_straty in mlm_straty_list:
        token_ppl_list = []
        gene_ppl_list = []
        token_acc_list = []
        gene_acc_list = []
        if seed_list is not None:
          for seed in seed_list:
            if len(seed) > 0:
              seed = '_{}'.format(seed)
            ## acquire model dir
            mdl_dir=os.popen(f"grep 'Saving model checkpoint' /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/job_logs/archive_antibody_bert/antibody_bert_{rp_set}_{mdl_subIdx}_{ini_epoch}_torch_distriTrain_fp16_mulNode_{mlm_straty}_{seq_straty}_{loss_straty}{seed}.0.out| cut -d'/' -f2 | sort | uniq").read().strip('\n')
            set_token_ppl_list = []
            set_gene_ppl_list = []
            set_token_acc_list = []
            set_gene_acc_list = []
            for setNm in set2analysis:
              ## load metrics from json
              mdl_path = f"/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/results_to_keep/{rp_set}/{rp_set}_{mdl_subIdx}_{ini_epoch}_antibody_models/{seq_straty}_{mlm_straty}_{loss_straty}/{mdl_dir}/results_metrics_{setNm}_all.json"
              with open(mdl_path) as handle:
                metric_json = json.load(handle)
              set_token_ppl_list.append(metric_json['lm_ece'])
              set_token_acc_list.append(metric_json['accuracy'])
              set_gene_ppl_list.append(metric_json['AB_subClass_ece'])
              set_gene_acc_list.append(metric_json['accuracy_subClass_AB'])
            ## append to seed-level list
            token_ppl_list.append(set_token_ppl_list)
            gene_ppl_list.append(set_gene_ppl_list)
            token_acc_list.append(set_token_acc_list)
            gene_acc_list.append(set_gene_acc_list)
        ## report mean/std
        token_ppl_mean, token_ppl_std = np.mean(token_ppl_list,axis=0), np.std(token_ppl_list,axis=0)
        token_acc_mean, token_acc_std = np.mean(token_acc_list,axis=0), np.std(token_acc_list,axis=0)
        gene_ppl_mean, gene_ppl_std = np.mean(gene_ppl_list,axis=0), np.std(gene_ppl_list,axis=0)
        gene_acc_mean, gene_acc_std = np.mean(gene_acc_list,axis=0), np.std(gene_acc_list,axis=0)
        print(f"{loss_straty},{seq_straty},{mlm_straty}:{token_ppl_mean[0]:.3f}/{token_ppl_std[0]:.3f};{token_ppl_mean[1]:.3f}/{token_ppl_std[1]:.3f};{token_ppl_mean[2]:.3f}/{token_ppl_std[2]:.3f},{token_acc_mean[0]:.3f}/{token_acc_std[0]:.3f};{token_acc_mean[1]:.3f}/{token_acc_std[1]:.3f};{token_acc_mean[2]:.3f}/{token_acc_std[2]:.3f},{gene_ppl_mean[0]:.3f}/{gene_ppl_std[0]:.3f};{gene_ppl_mean[1]:.3f}/{gene_ppl_std[1]:.3f};{gene_ppl_mean[2]:.3f}/{gene_ppl_std[2]:.3f},{gene_acc_mean[0]:.3f}/{gene_acc_std[0]:.3f};{gene_acc_mean[1]:.3f}/{gene_acc_std[1]:.3f};{gene_acc_mean[2]:.3f}/{gene_acc_std[2]:.3f}")
              
def lm_comparison(working_dir: str = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                  rp_set: str = 'rp15_all',
                  mdl_subIdx: str = '1',
                  ini_epoch: str = '700',
                  set2analysis: List = ['sabdabABData','exp_batch1','exp_batch2'],
                  seq_straty_list: List = ['seqConcate','seqIndiv'],
                  mlm_straty_list: List = ['vanilla','cdr_vanilla','cdr_margin','cdr_pair'],
                  loss_straty_list: List = ['mlm_only','mlm_gene'],
                  seed_list: List = None,
                  save_df: bool = True):
  """ compare language modeling performance (ppl) of models
      across four masking strategy: ``vanilla" ``cdr_vanilla" ``cdr_margin" ``cdr_pair"
  """
  if save_df:
    whole_dtList = []
    ## loop over loss, seq, mlm strategies
    for loss_straty in loss_straty_list:
      for seq_straty in seq_straty_list:
        for mlm_straty in mlm_straty_list:
          if seed_list is not None:
            for seed in seed_list:
              print(f"{seq_straty};{loss_straty};{mlm_straty};{seed}")
              if len(seed) > 0:
                seed = '_{}'.format(seed)
              ## acquire model dir
              mdl_dir=os.popen(f"grep 'Saving model checkpoint' {working_dir}/job_logs/archive_antibody_bert/antibody_bert_{rp_set}_{mdl_subIdx}_{ini_epoch}_torch_distriTrain_fp16_mulNode_{mlm_straty}_{seq_straty}_{loss_straty}{seed}.0.out| awk -F 'results/' '{{print $2}}' | sort | uniq").read().strip('\n')
              #print(f"{working_dir}/job_logs/archive_antibody_bert/antibody_bert_{rp_set}_{mdl_subIdx}_{ini_epoch}_torch_distriTrain_fp16_mulNode_{mlm_straty}_{seq_straty}_{loss_straty}{seed}.0.out")
              ## loop over eval mlm strategy
              for eval_mlm in mlm_straty_list:
                # one fig for each set
                for setNm in set2analysis:
                  ## load metrics from json
                  json_path = f"{working_dir}/results_to_keep/{rp_set}/{rp_set}_{mdl_subIdx}_{ini_epoch}_antibody_models/{seq_straty}_{mlm_straty}_{loss_straty}/{mdl_dir}/results_metrics_{setNm}_all_{eval_mlm}.json"
                  with open(json_path) as handle:
                    metric_json = json.load(handle)
                  whole_dtList.append([metric_json[f'lm_ece_{eval_mlm}'],metric_json[f'accuracy_{eval_mlm}'],metric_json[f'AB_subClass_ece_{eval_mlm}'],metric_json[f'accuracy_subClass_AB_{eval_mlm}'],seq_straty, loss_straty, f"{seq_straty}/{loss_straty}", mlm_straty,eval_mlm, seed, setNm])
    print(f"total records: {len(whole_dtList)}")
    whole_dtFrame = pd.DataFrame(whole_dtList,columns=['lm_ece','lm_acc','subc_ece','subc_acc','seq_straty','loss_straty', 'seq_loss_straty', 'mlm_straty', 'eval_mlm', 'seed', 'setNm'])
    whole_dtFrame.to_pickle(f"{working_dir}/data_process/antibody/savedDataFrame/lm_com.pkl")
  else:
    whole_dtFrame=pd.read_pickle(f"{working_dir}/data_process/antibody/savedDataFrame/lm_com.pkl")

  ## draw figures
  #filter_df = whole_dtFrame.loc[whole_dtFrame(["setNm"]==setNm)]
  sns.set(style="whitegrid", font_scale=1.5) #rc={"lines.linewidth": 1.0, 'figure.figsize':(120,80)}, font_scale=6
  #gax = sns.catplot(x="eval_mlm",y="lm_ece",hue="mlm_straty",data=whole_dtFrame,row="setNm",col="seq_loss_straty", ci='sd',order=mlm_straty_list,hue_order=mlm_straty_list, row_order=set2analysis, col_order=['seqConcate/mlm_only','seqConcate/mlm_gene','seqIndiv/mlm_only','seqIndiv/mlm_gene'],kind='bar',height=5, aspect=1.8)
  gax = sns.catplot(x="eval_mlm",y="lm_ece",hue="seq_loss_straty",data=whole_dtFrame,row="setNm",col="mlm_straty", ci='sd',order=mlm_straty_list,col_order=mlm_straty_list, row_order=set2analysis, hue_order=['seqConcate/mlm_only','seqConcate/mlm_gene','seqIndiv/mlm_only','seqIndiv/mlm_gene'],kind='bar',height=5, aspect=1.8)
  plt.savefig(f"{working_dir}/data_process/antibody/figures/lm_com_eval_seqLoss.png",dpi=100)
            


if __name__ == '__main__':
  #rp15_all_1_700_antibody_models
  '''
  draw_tSNE(pretrain_set_list = ['rp15_all_1_700_antibody_models'],
            set2draw=['sabdab/sabdabABData','exp_data/exp_batch2','exp_data/exp_batch1'],
            seq_straty_list=['seqIndiv'],
            mlm_straty_list=['cdr_margin'],
            loss_straty_list=['mlm_only'],
            loadDf_bool=True)
  #[0,42,128,646,1548]
  '''
  '''
  metrics_analysis(rp_set='rp15_all',
                   mdl_subIdx = '1',
                   ini_epoch = '700',
                   set2analysis = ['sabdabABData','exp_batch1','exp_batch2'],
                   seq_straty_list = ['seqConcate','seqIndiv'],
                   mlm_straty_list = ['vanilla','cdr_vanilla','cdr_margin','cdr_pair'],
                   loss_straty_list = ['mlm_only','mlm_gene'],
                   seed_list = ['0','42','128','646','1548'])
  '''
  lm_comparison(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                rp_set = 'rp15_all',
                mdl_subIdx = '1',
                ini_epoch = '700',
                set2analysis = ['sabdabABData','exp_batch1','exp_batch2'],
                seq_straty_list = ['seqConcate','seqIndiv'],
                mlm_straty_list = ['vanilla','cdr_vanilla','cdr_margin','cdr_pair'],
                loss_straty_list = ['mlm_only','mlm_gene'],
                seed_list = ['0','42','128','646','1548'],
                save_df = False)