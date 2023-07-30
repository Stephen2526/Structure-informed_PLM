import re, collections, os, sys, json, lmdb, random
from string import Template
import numpy as np
import pandas as pd
import pickle as pkl
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import spearmanr
import seaborn as sns
from functools import reduce


def seq_struct_fun_analysis(
        report_ave_score: bool = False,
        draw_figure: bool = False,
        draw_figure_family: bool = False,
        draw_figure_pre2fine: bool = False
    ):
    """Analyze performance of AA, SS, RSA, DistMap, Fitness predictions for:
    pretrained seq models, finetuned seq models
    """
    path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
    fitness_fam_info = 'data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv'
    fitness_fam_info_df = pd.read_csv(f'{path}/{fitness_fam_info}',delimiter=',',header=0,comment='#')
    setNm_list = ['AMIE_PSEAE','DLG4_RAT','PABP_YEAST','RASH_HUMAN','KKA2_KLEPN','PTEN_HUMAN','MTH3_HAEAESTABILIZED','HIS7_YEAST','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','YAP1_HUMAN']
    label_type_list = ['aa', 'ss', 'rsa', 'fitness', 'distMap'] 
    label_task_mapping = {
        'aa':'masked_language_modeling',
        'ss':'structure_awareness_1d',
        'rsa':'structure_awareness_1d',
        'distMap':'structure_awareness_2d',
        'fitness':'mutation_fitness_UNsupervise_mutagenesis'
    }
    json_key_mapping = {
        'aa':'aa_ppl_20renor', #aa_ppl
        'ss':'ss_ppl',
        'rsa':'rsa_ppl',
        'distMap':'distMap_ce_ave_exp', #distMap_ce_mvave_exp,distMap_ppl_ave,distMap_ppl_mvave
        'fitness':'spearmanR'
    }
    background_ppl = {
        'aa':20,
        'ss':3,
        'rsa':2,
        'distMap':32
    }
    model_name_list = ['rp75_all_1_224','rp15_all_1_729','rp15_all_2_729','rp15_all_3_729','rp15_all_4_729']
    model_name_mapping = {'rp75_all_1_224':'rp75_bert_1',
                          'rp15_all_1_729':'rp15_bert_1',
                          'rp15_all_2_729':'rp15_bert_2',
                          'rp15_all_3_729':'rp15_bert_3',
                          'rp15_all_4_729':'rp15_bert_4'}
    model_epoch_mapping = {'rp75_all_1_224':'224',
                          'rp15_all_1_729':'729',
                          'rp15_all_2_729':'729',
                          'rp15_all_3_729':'729',
                          'rp15_all_4_729':'729'}
    train_mode_list = ['pretrain','finetune']
    eval_set_list = ['wt','mutation'] #'valid',
    eval_set_mapping = {
        'valid':'valid',
        'wt':'AFDB',
        'mutation':'mutation'}
    log_file_tmpls = {
        'pretrain_aa': Template('baseline_bert_${task_name}_${model_name}.PreTrain.${set_name}.${eval_set}.${label_type}.eval.out'),
        'finetune_aa': Template('baseline_bert_${task_name}_${model_name}.seq_finetune.reweighted.best.${set_name}.${eval_set}.${label_type}.eval.out'),
        'pretrain_ss': Template('baseline_bert_${task_name}_${model_name}.${set_name}.${eval_set}.${label_type}.eval.out'),
        'finetune_ss': Template('baseline_bert_${task_name}_${model_name}.seq_finetune.reweighted.best.${set_name}.${eval_set}.${label_type}.eval.out'),
        'pretrain_rsa': Template('baseline_bert_${task_name}_${model_name}.${set_name}.${eval_set}.${label_type}.eval.out'),
        'finetune_rsa': Template('baseline_bert_${task_name}_${model_name}.seq_finetune.reweighted.best.${set_name}.${eval_set}.${label_type}.eval.out'),
        'pretrain_distMap': Template('baseline_bert_${task_name}_${model_name}.${set_name}.${eval_set}.${label_type}.eval.distMap32.out'),
        'finetune_distMap': Template('baseline_bert_${task_name}_${model_name}.seq_finetune.reweighted.best.${set_name}.${eval_set}.${label_type}.eval.distMap32.out'),
        'pretrain_fitness': Template('baseline_bert_${task_name}_${model_name}.PreTrain.${set_name}.${variant_set}.out'),
        'finetune_fitness': Template('baseline_bert_${task_name}_${model_name}.seq_finetune.reweighted.best.${set_name}.${variant_set}.out'),
        }

    ## dataframe: 'model_name','train_mode','family_name','eval_set','label_type','variant_set_name', 'score'
    data_records = []
    ## loop models
    for mdl_i in range(len(model_name_list)):
        model_name = model_name_list[mdl_i]
        model_name_2 = model_name_mapping[model_name]
        for train_mode in train_mode_list:
            ## loop families
            for fam in setNm_list:
                # if fam in ['HIS7_YEAST']:
                #     continue
                fam_muta_sets = fitness_fam_info_df.loc[fitness_fam_info_df['Shin2021_set'] == fam]['setNM'].tolist()
                ## loop mutagenesis set for fam
                for var_set in fam_muta_sets:
                    ## loop eval setseval_set
                    for eval_set in eval_set_list:
                        if eval_set in ['valid','wt']:
                            tmp_label_type = ['aa', 'ss', 'rsa', 'distMap']
                        elif eval_set in ['mutation']:
                            tmp_label_type = ['fitness']
                        ## loop labels
                        for label_type in tmp_label_type:
                            log_file_path = log_file_tmpls[f'{train_mode}_{label_type}'].substitute(task_name=label_task_mapping[label_type],model_name=model_name,set_name=fam,eval_set=eval_set,label_type=label_type,variant_set=var_set)
                            if len(log_file_path) == 0:
                                print(log_file_path)
                            model_path = os.popen(f"grep -a 'loading weights file' job_logs/archive_baseline_bert_eval/{log_file_path} | awk -F '/pytorch_model' '{{print $1}}' | rev | cut -d'/' -f1 | rev | sort | uniq | tr -d '\r'").read().strip('\n')
                            if len(model_path) == 0:
                                print(model_path)
                            ## load score files
                            if label_type in ['aa', 'ss', 'rsa', 'distMap']:
                                if train_mode == 'pretrain' and label_type == 'aa':
                                    epoch = model_epoch_mapping[model_name]
                                else:
                                    epoch = 'best'
                                with open(f'{path}/eval_results/{label_task_mapping[label_type]}/predictions/{model_path}/{fam}_{eval_set_mapping[eval_set]}_{epoch}_metrics.json') as fl:
                                    metric_json = json.load(fl)
                                score = metric_json[json_key_mapping[label_type]]
                                data_records.append([model_name_2,train_mode,fam,var_set,eval_set,label_type, score])
                            elif label_type in ['fitness']:
                                if train_mode == 'pretrain':
                                    epoch = model_epoch_mapping[model_name]
                                else:
                                    epoch = 'best'
                                with open(f'{path}/eval_results/{label_task_mapping[label_type]}/predictions/{model_path}/{var_set}_{epoch}_metrics.json') as fl:
                                    metric_json = json.load(fl)
                                score = metric_json[json_key_mapping[label_type]]
                                data_records.append([model_name_2,train_mode,fam,var_set,eval_set,label_type,score])
    whole_df = pd.DataFrame(data_records,columns=['model_name','train_mode','family_name','variant_set_name','eval_set','label_type','score'])
    print(f'total num: {len(whole_df)}')

    if report_ave_score:
        record_scores = {}
        ## report average score
        for train_mode in ['pretrain','finetune']:
            for model_name in ['rp75_bert_1','rp15_bert_1','rp15_bert_2','rp15_bert_3','rp15_bert_4']:
                struct_merge_df = pd.DataFrame()
                for label_type in ['fitness', 'aa', 'ss', 'rsa', 'distMap']:
                    if label_type in ['aa', 'ss', 'rsa', 'distMap']:
                        eval_sets = ['wt'] #'valid',
                        for eval_set in eval_sets:
                            print(f'>>{model_name}-{train_mode}-{label_type}-{eval_set}')
                            target_df = whole_df.loc[(whole_df['model_name'] == model_name) & (whole_df['train_mode'] == train_mode) & (whole_df['label_type'] == label_type) & (whole_df['eval_set'] == eval_set)].drop_duplicates()
                            trim_df = target_df[['model_name','train_mode','family_name','eval_set','label_type','score']].drop_duplicates().reset_index()
                            struct_merge_df[f'{label_type}_{eval_set}'] = background_ppl[label_type] - trim_df['score']
                            struct_merge_df[f'{label_type}_{eval_set}_gainPPLNor'] = (background_ppl[label_type] - trim_df['score']) / background_ppl[label_type]
                            delta_nor_score = struct_merge_df[f'{label_type}_{eval_set}_gainPPLNor']
                            #ave_score = trim_df['score'].mean()
                            #sd_score = trim_df['score'].std(ddof=0)
                            ave_score = delta_nor_score.mean()
                            sd_score = delta_nor_score.std(ddof=0)
                            print(f'{len(trim_df)}')
                            record_scores[f'{model_name}-{train_mode}-{label_type}-{eval_set}'] = [ave_score,sd_score]
                    elif label_type in ['fitness']:
                        eval_sets = ['mutation']
                        for eval_set in eval_sets:
                            print(f'>>{model_name}-{train_mode}-{label_type}-{eval_set}')
                            target_df = whole_df.loc[(whole_df['model_name'] == model_name) & (whole_df['train_mode'] == train_mode) & (whole_df['label_type'] == label_type) & (whole_df['eval_set'] == eval_set)].drop_duplicates().reset_index()
                            ave_score = target_df['score'].mean()
                            sd_score = target_df['score'].std(ddof=0)
                            print(f'{len(target_df)}')
                            record_scores[f'{model_name}-{train_mode}-{label_type}-{eval_set}'] = [ave_score,sd_score]
                ## ppl gain averaged over three structure labels
                #struct_merge_df['srd_gain_ppl_valid'] = (struct_merge_df['ss_valid']+struct_merge_df['rsa_valid']+struct_merge_df['distMap_valid']) / (background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                struct_merge_df['srd_gain_ppl_wt'] = (struct_merge_df['ss_wt']+struct_merge_df['rsa_wt']+struct_merge_df['distMap_wt']) / (background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                #struct_merge_df['asrd_gain_ppl_valid'] = (struct_merge_df['aa_valid']+struct_merge_df['ss_valid']+struct_merge_df['rsa_valid']+struct_merge_df['distMap_valid']) / (background_ppl['aa']+background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                struct_merge_df['asrd_gain_ppl_wt'] = (struct_merge_df['aa_wt']+struct_merge_df['ss_wt']+struct_merge_df['rsa_wt']+struct_merge_df['distMap_wt']) / (background_ppl['aa']+background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])

                #record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-valid'] = [struct_merge_df['srd_gain_ppl_valid'].mean(),struct_merge_df['srd_gain_ppl_valid'].std(ddof=0)]
                record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-wt'] = [struct_merge_df['srd_gain_ppl_wt'].mean(),struct_merge_df['srd_gain_ppl_wt'].std(ddof=0)]
                #record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-valid'] = [struct_merge_df['asrd_gain_ppl_valid'].mean(),struct_merge_df['srd_gain_ppl_valid'].std(ddof=0)]
                record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-wt'] = [struct_merge_df['asrd_gain_ppl_wt'].mean(),struct_merge_df['asrd_gain_ppl_wt'].std(ddof=0)]
        
                
                ## geometric mean of ppl gain per token
                #struct_merge_df['srd_gain_ppl_GM_valid'] = struct_merge_df[['ss_valid_gainPPLNor','rsa_valid_gainPPLNor','distMap_valid_gainPPLNor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                struct_merge_df['srd_gain_ppl_GM_wt'] = struct_merge_df[['ss_wt_gainPPLNor','rsa_wt_gainPPLNor','distMap_wt_gainPPLNor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                #struct_merge_df['asrd_gain_ppl_GM_valid'] = struct_merge_df[['aa_valid_gainPPLNor','ss_valid_gainPPLNor','rsa_valid_gainPPLNor','distMap_valid_gainPPLNor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                struct_merge_df['asrd_gain_ppl_GM_wt'] = struct_merge_df[['aa_wt_gainPPLNor','ss_wt_gainPPLNor','rsa_wt_gainPPLNor','distMap_wt_gainPPLNor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)

                #record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-GM-valid'] = [struct_merge_df['srd_gain_ppl_GM_valid'].mean(),struct_merge_df['srd_gain_ppl_GM_valid'].std(ddof=0)]
                record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-GM-wt'] = [struct_merge_df['srd_gain_ppl_GM_wt'].mean(),struct_merge_df['srd_gain_ppl_GM_wt'].std(ddof=0)]
                #record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-GM-valid'] = [struct_merge_df['asrd_gain_ppl_GM_valid'].mean(),struct_merge_df['srd_gain_ppl_GM_valid'].std(ddof=0)]
                record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-GM-wt'] = [struct_merge_df['asrd_gain_ppl_GM_wt'].mean(),struct_merge_df['asrd_gain_ppl_GM_wt'].std(ddof=0)]
        
        ## report in format
        mdl_name_report_map = {'rp75_bert_1':'RP75\_B1','rp15_bert_1':'RP15\_B1','rp15_bert_2':'RP15\_B2','rp15_bert_3':'RP15\_B3','rp15_bert_4':'RP15\_B4'}
        for eval_set in ['wt']:#,'valid'
            for train_mode in ['pretrain','finetune']:
                print(f'>>>{train_mode}, {eval_set}<<<')
                for model_name in ['rp75_bert_1','rp15_bert_1','rp15_bert_2','rp15_bert_3','rp15_bert_4']:
                    print(f"{mdl_name_report_map[model_name]} & {record_scores[f'{model_name}-{train_mode}-fitness-mutation'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-fitness-mutation'][1]} & {record_scores[f'{model_name}-{train_mode}-aa-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-aa-{eval_set}'][1]} & {record_scores[f'{model_name}-{train_mode}-ss-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-ss-{eval_set}'][1]} & {record_scores[f'{model_name}-{train_mode}-rsa-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-rsa-{eval_set}'][1]} & {record_scores[f'{model_name}-{train_mode}-distMap-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-distMap-{eval_set}'][1]} & {record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-GM-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-srd-gain-ppl-GM-{eval_set}'][1]} & {record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-GM-{eval_set}'][0]} & \Ft {record_scores[f'{model_name}-{train_mode}-asrd-gain-ppl-GM-{eval_set}'][1]} \\\\")

    if draw_figure:
        ## draw figures
        # fitness - {aa,ss,rsa} - scatter
        # pretrain min-fitness 0; max-fitness 0.57
        sizes_value = (20,200)
        size_norm_value = (0,0.8)
        alpha_value = 0.6
        palette_name = 'coolwarm'
        size_norm_range_map = {'pretrain':(0,0.6),'finetune': (0.4,0.8)}
        xlimit_map = {'pretrain':[0,0.9],'finetune':[0.5,0.95]}
        ylimit_map = {'pretrain':[0.30,0.7],'finetune':[0.40,0.7]}
        for train_mode in ['pretrain','finetune']:
            for model_name in ['rp75_bert_1','rp15_bert_1','rp15_bert_2','rp15_bert_3','rp15_bert_4']:
                for eval_set in ['wt']: #'valid'
                    target_df = whole_df.loc[(whole_df['model_name'] == model_name) & (whole_df['train_mode'] == train_mode)].drop_duplicates().reset_index()
                    fitness_df = target_df.loc[(target_df['label_type'] == 'fitness')].drop_duplicates().reset_index()[['family_name','variant_set_name','score']]
                    fitness_df = fitness_df.rename(columns={"score": "fitness_R"})

                    aa_df = target_df.loc[(target_df['label_type'] == 'aa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score']]
                    aa_df['aa_ppl_delta_nor'] = (background_ppl['aa'] - aa_df['score']) / background_ppl['aa']
                    aa_df = aa_df.rename(columns={"score": "aa_ppl"})
                    
                    fit_aa_df = pd.merge(fitness_df,aa_df,on=['family_name','variant_set_name'])

                    ## fitness-aa + ss
                    ss_df = target_df.loc[(target_df['label_type'] == 'ss') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score']]
                    ss_df['ss_ppl_delta_nor'] = (background_ppl['ss'] - ss_df['score']) / background_ppl['ss']
                    ss_df = ss_df.rename(columns={"score": "ss_ppl"})
                    #ss_df['struct_score'] = ss_df['ss_ppl']
                    #ss_df['struct_type'] = 'ss'
                    fit_aa_ss_df = pd.merge(fit_aa_df,ss_df,on=['family_name','variant_set_name'])
                    plt.figure()
                    g = sns.scatterplot(data=fit_aa_ss_df,x="aa_ppl_delta_nor",y="ss_ppl_delta_nor",hue="fitness_R",size="fitness_R",alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range_map[train_mode], hue_norm=size_norm_range_map[train_mode],legend=False, palette=palette_name)
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                    plt.xlim(xlimit_map[train_mode][0], xlimit_map[train_mode][1])
                    plt.ylim(ylimit_map[train_mode][0], ylimit_map[train_mode][1])
                    plt.savefig(f'eval_results/structure_awareness_eval/figures/fit_aa_ss_{model_name}_{train_mode}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    plt.close()

                    rsa_df = target_df.loc[(target_df['label_type'] == 'rsa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score']]
                    rsa_df['rsa_ppl_delta_nor'] = (background_ppl['rsa'] - rsa_df['score']) / background_ppl['rsa']
                    rsa_df = rsa_df.rename(columns={"score": "rsa_ppl"})
                    #rsa_df['struct_score'] = rsa_df['rsa_ppl']
                    #rsa_df['struct_type'] = 'rsa'
                    fit_aa_rsa_df = pd.merge(fit_aa_df,rsa_df,on=['family_name','variant_set_name'])
                    plt.figure()
                    g = sns.scatterplot(data=fit_aa_rsa_df, x="aa_ppl_delta_nor", y="rsa_ppl_delta_nor", hue="fitness_R", size="fitness_R", alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range_map[train_mode],hue_norm=size_norm_range_map[train_mode],legend=False,palette=palette_name)
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                    plt.xlim(xlimit_map[train_mode][0], xlimit_map[train_mode][1])
                    plt.ylim(ylimit_map[train_mode][0], ylimit_map[train_mode][1])
                    plt.savefig(f'eval_results/structure_awareness_eval/figures/fit_aa_rsa_{model_name}_{train_mode}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    plt.close()

                    dist_df = target_df.loc[(target_df['label_type'] == 'distMap') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score']]
                    dist_df['dist_ppl_delta_nor'] = (background_ppl['distMap'] - dist_df['score']) / background_ppl['distMap']
                    dist_df = dist_df.rename(columns={"score": "distMap_ppl"})
                    #dist_df['struct_score'] = dist_df['distMap_ppl']
                    #dist_df['struct_type'] = 'distMap'
                    fit_aa_dist_df = pd.merge(fit_aa_df,dist_df,on=['family_name','variant_set_name'])
                    plt.figure()
                    g = sns.scatterplot(data=fit_aa_dist_df, x="aa_ppl_delta_nor", y="dist_ppl_delta_nor", hue="fitness_R", size="fitness_R", alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range_map[train_mode],hue_norm=size_norm_range_map[train_mode],legend=False,palette=palette_name)
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                    plt.xlim(xlimit_map[train_mode][0], xlimit_map[train_mode][1])
                    plt.ylim(ylimit_map[train_mode][0], ylimit_map[train_mode][1])
                    plt.savefig(f'eval_results/structure_awareness_eval/figures/fit_aa_distMap_{model_name}_{train_mode}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    plt.close()

                    #all_comb_df = pd.merge([fit_aa_ss_df,fit_aa_rsa_df,fit_aa_dist_df], ignore_index=True)
                    all_data_frames = [fit_aa_ss_df,fit_aa_rsa_df,fit_aa_dist_df]
                    all_comb_df = reduce(lambda left,right: pd.merge(left,right,on=['family_name','variant_set_name','fitness_R','aa_ppl','aa_ppl_delta_nor']), all_data_frames)
                    all_comb_df['ss_rsa_dist_ppl_delta_nor'] = (background_ppl['ss']-all_comb_df['ss_ppl']+background_ppl['rsa']-all_comb_df['rsa_ppl']+background_ppl['distMap']-all_comb_df['distMap_ppl']) / (background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                    all_comb_df['ss_rsa_dist_ppl_delta_nor_GM'] = all_comb_df[['ss_ppl_delta_nor','rsa_ppl_delta_nor','dist_ppl_delta_nor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                    plt.figure()
                    g = sns.scatterplot(data=all_comb_df, x="aa_ppl_delta_nor", y="ss_rsa_dist_ppl_delta_nor_GM", hue="fitness_R", size="fitness_R", sizes=sizes_value, size_norm=size_norm_range_map[train_mode],hue_norm=size_norm_range_map[train_mode], alpha=alpha_value, palette=palette_name)
                    plt.legend(title='fitness_R',bbox_to_anchor=(1.03, .63), loc='upper left', borderaxespad=0)
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    plt.xlim(xlimit_map[train_mode][0], xlimit_map[train_mode][1])
                    plt.ylim(ylimit_map[train_mode][0], ylimit_map[train_mode][1])
                    plt.savefig(f'eval_results/structure_awareness_eval/figures/fit_aa_all_{model_name}_{train_mode}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    plt.close()
    if draw_figure_pre2fine:
        ## draw figures
        # fitness - {aa,ss,rsa} - scatter
        # pretrain min-fitness 0; max-fitness 0.57
        sizes_value = (20,200)
        alpha_value = 0.6
        palette_name = 'coolwarm'
        size_norm_range = (0,0.8)
        xlimit_range = [0,0.95]
        ylimit_range = [0.3,0.7]
        for model_name in ['rp75_bert_1','rp15_bert_1','rp15_bert_2','rp15_bert_3','rp15_bert_4']:
            for eval_set in ['wt']: #'valid'
                target_df = whole_df.loc[(whole_df['model_name'] == model_name)].drop_duplicates().reset_index()
                fitness_df = target_df.loc[(target_df['label_type'] == 'fitness')].drop_duplicates().reset_index()[['family_name','variant_set_name','score','train_mode']]
                fitness_df = fitness_df.rename(columns={"score": "fitness_R"})

                aa_df = target_df.loc[(target_df['label_type'] == 'aa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score','train_mode']]
                aa_df['aa_ppl_delta_nor'] = (background_ppl['aa'] - aa_df['score']) / background_ppl['aa']
                aa_df = aa_df.rename(columns={"score": "aa_ppl"})
                
                fit_aa_df = pd.merge(fitness_df,aa_df,on=['family_name','variant_set_name','train_mode'])

                ## fitness-aa + ss
                ss_df = target_df.loc[(target_df['label_type'] == 'ss') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score','train_mode']]
                ss_df['ss_ppl_delta_nor'] = (background_ppl['ss'] - ss_df['score']) / background_ppl['ss']
                ss_df = ss_df.rename(columns={"score": "ss_ppl"})
                #ss_df['struct_score'] = ss_df['ss_ppl']
                #ss_df['struct_type'] = 'ss'
                fit_aa_ss_df = pd.merge(fit_aa_df,ss_df,on=['family_name','variant_set_name','train_mode'])
                plt.figure()
                g = sns.scatterplot(data=fit_aa_ss_df,x="aa_ppl_delta_nor",y="ss_ppl_delta_nor",hue="fitness_R",size="fitness_R",alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range, hue_norm=size_norm_range,legend=False, palette=palette_name)
                g.set(ylabel=None)
                g.set(xlabel=None)
                #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                plt.xlim(xlimit_range[0], xlimit_range[1])
                plt.ylim(ylimit_range[0], ylimit_range[1])
                plt.savefig(f'eval_results/structure_awareness_eval/figures/pre2fine_fit_aa_ss_{model_name}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                plt.close()

                rsa_df = target_df.loc[(target_df['label_type'] == 'rsa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score','train_mode']]
                rsa_df['rsa_ppl_delta_nor'] = (background_ppl['rsa'] - rsa_df['score']) / background_ppl['rsa']
                rsa_df = rsa_df.rename(columns={"score": "rsa_ppl"})
                #rsa_df['struct_score'] = rsa_df['rsa_ppl']
                #rsa_df['struct_type'] = 'rsa'
                fit_aa_rsa_df = pd.merge(fit_aa_df,rsa_df,on=['family_name','variant_set_name','train_mode'])
                plt.figure()
                g = sns.scatterplot(data=fit_aa_rsa_df, x="aa_ppl_delta_nor", y="rsa_ppl_delta_nor", hue="fitness_R", size="fitness_R", alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range,hue_norm=size_norm_range,legend=False,palette=palette_name)
                g.set(ylabel=None)
                g.set(xlabel=None)
                #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                plt.xlim(xlimit_range[0], xlimit_range[1])
                plt.ylim(ylimit_range[0], ylimit_range[1])
                plt.savefig(f'eval_results/structure_awareness_eval/figures/pre2fine_fit_aa_rsa_{model_name}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                plt.close()

                dist_df = target_df.loc[(target_df['label_type'] == 'distMap') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['family_name','variant_set_name','score','train_mode']]
                dist_df['dist_ppl_delta_nor'] = (background_ppl['distMap'] - dist_df['score']) / background_ppl['distMap']
                dist_df = dist_df.rename(columns={"score": "distMap_ppl"})
                #dist_df['struct_score'] = dist_df['distMap_ppl']
                #dist_df['struct_type'] = 'distMap'
                fit_aa_dist_df = pd.merge(fit_aa_df,dist_df,on=['family_name','variant_set_name','train_mode'])
                plt.figure()
                g = sns.scatterplot(data=fit_aa_dist_df, x="aa_ppl_delta_nor", y="dist_ppl_delta_nor", hue="fitness_R", size="fitness_R", alpha=alpha_value, sizes=sizes_value, size_norm=size_norm_range,hue_norm=size_norm_range,legend=False,palette=palette_name)
                g.set(ylabel=None)
                g.set(xlabel=None)
                #plt.legend(title='fitness_R', bbox_to_anchor=(1.01, .63), loc='upper left', borderaxespad=0)
                plt.xlim(xlimit_range[0], xlimit_range[1])
                plt.ylim(ylimit_range[0], ylimit_range[1])
                plt.savefig(f'eval_results/structure_awareness_eval/figures/pre2fine_fit_aa_distMap_{model_name}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                plt.close()

                #all_comb_df = pd.merge([fit_aa_ss_df,fit_aa_rsa_df,fit_aa_dist_df], ignore_index=True)
                all_data_frames = [fit_aa_ss_df,fit_aa_rsa_df,fit_aa_dist_df]
                all_comb_df = reduce(lambda left,right: pd.merge(left,right,on=['family_name','variant_set_name','fitness_R','aa_ppl','aa_ppl_delta_nor','train_mode']), all_data_frames)
                all_comb_df['ss_rsa_dist_ppl_delta_nor'] = (background_ppl['ss']-all_comb_df['ss_ppl']+background_ppl['rsa']-all_comb_df['rsa_ppl']+background_ppl['distMap']-all_comb_df['distMap_ppl']) / (background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                all_comb_df['ss_rsa_dist_ppl_delta_nor_GM'] = all_comb_df[['ss_ppl_delta_nor','rsa_ppl_delta_nor','dist_ppl_delta_nor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                plt.figure()
                g = sns.scatterplot(data=all_comb_df, x="aa_ppl_delta_nor", y="ss_rsa_dist_ppl_delta_nor_GM", hue="fitness_R", size="fitness_R", sizes=sizes_value, size_norm=size_norm_range,hue_norm=size_norm_range, alpha=alpha_value, palette=palette_name, zorder=2)
                for set_name in setNm_list:
                #   g.plot(all_comb_df.loc[all_comb_df['family_name']==set_name]['aa_ppl_delta_nor'], all_comb_df.loc[all_comb_df['family_name']==set_name]['ss_rsa_dist_ppl_delta_nor_GM'], '-', alpha=0.6, zorder=1,color='gray',linewidth=1.2)
                    arrow_x = all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='pretrain')]['aa_ppl_delta_nor'].to_list()
                    arrow_y = all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='pretrain')]['ss_rsa_dist_ppl_delta_nor_GM'].to_list()
                    arrow_x_direct = all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='finetune')].reset_index()['aa_ppl_delta_nor'].subtract(all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='pretrain')].reset_index()['aa_ppl_delta_nor']).to_list()
                    arrow_y_direct = all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='finetune')].reset_index()['ss_rsa_dist_ppl_delta_nor_GM'].subtract(all_comb_df.loc[(all_comb_df['family_name']==set_name) & (all_comb_df['train_mode']=='pretrain')].reset_index()['ss_rsa_dist_ppl_delta_nor_GM']).to_list()
                    g.quiver(arrow_x,arrow_y,arrow_x_direct,arrow_y_direct,angles='xy',scale_units='xy',scale=1, width=0.003,alpha=0.5)

                plt.legend(title='fitness_R',bbox_to_anchor=(1.03, .63), loc='upper left', borderaxespad=0)
                g.set(ylabel=None)
                g.set(xlabel=None)
                plt.xlim(xlimit_range[0], xlimit_range[1])
                plt.ylim(ylimit_range[0], ylimit_range[1])
                plt.savefig(f'eval_results/structure_awareness_eval/figures/pre2fine_fit_aa_all_{model_name}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                plt.close()
    if draw_figure_family:
        sizes_value = (100,500)
        alpha_value = 0.8
        palette_name = 'coolwarm'
        size_norm_range = (0,0.8)
        xlimit_range = [0,0.95] ## aa
        ylimit_range = [0.3,0.7] ## srd
        for set_name in setNm_list:
            var_set_names = whole_df.loc[(whole_df['family_name'] == set_name)]['variant_set_name'].drop_duplicates().to_list()
            for var_set in var_set_names:
                target_df = whole_df.loc[(whole_df['family_name'] == set_name) & (whole_df['variant_set_name'] == var_set)].drop_duplicates().reset_index()
                fitness_df = target_df.loc[(target_df['label_type'] == 'fitness')].drop_duplicates().reset_index()[['model_name','variant_set_name','score','train_mode']]
                fitness_df = fitness_df.rename(columns={"score": "fitness_R"})
                for eval_set in ['wt']: #'valid'
                    aa_df = target_df.loc[(target_df['label_type'] == 'aa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['model_name','variant_set_name','score','train_mode']]
                    aa_df['aa_ppl_delta_nor'] = (background_ppl['aa'] - aa_df['score']) / background_ppl['aa']
                    aa_df = aa_df.rename(columns={"score": "aa_ppl"})
                    
                    fit_aa_df = pd.merge(fitness_df,aa_df,on=['model_name','variant_set_name','train_mode'])

                    ## fitness-aa + ss
                    ss_df = target_df.loc[(target_df['label_type'] == 'ss') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['model_name','variant_set_name','score','train_mode']]
                    ss_df['ss_ppl_delta_nor'] = (background_ppl['ss'] - ss_df['score']) / background_ppl['ss']
                    ss_df = ss_df.rename(columns={"score": "ss_ppl"})
                    #ss_df['struct_score'] = ss_df['ss_ppl']
                    #ss_df['struct_type'] = 'ss'
                    fit_aa_ss_df = pd.merge(fit_aa_df,ss_df,on=['model_name','variant_set_name','train_mode'])
                    
                    rsa_df = target_df.loc[(target_df['label_type'] == 'rsa') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['model_name','variant_set_name','score','train_mode']]
                    rsa_df['rsa_ppl_delta_nor'] = (background_ppl['rsa'] - rsa_df['score']) / background_ppl['rsa']
                    rsa_df = rsa_df.rename(columns={"score": "rsa_ppl"})
                    #rsa_df['struct_score'] = rsa_df['rsa_ppl']
                    #rsa_df['struct_type'] = 'rsa'
                    fit_aa_rsa_df = pd.merge(fit_aa_df,rsa_df,on=['model_name','variant_set_name','train_mode'])
                    
                    dist_df = target_df.loc[(target_df['label_type'] == 'distMap') & (target_df['eval_set'] == eval_set)].drop_duplicates().reset_index()[['model_name','variant_set_name','score','train_mode']]
                    dist_df['dist_ppl_delta_nor'] = (background_ppl['distMap'] - dist_df['score']) / background_ppl['distMap']
                    dist_df = dist_df.rename(columns={"score": "distMap_ppl"})
                    #dist_df['struct_score'] = dist_df['distMap_ppl']
                    #dist_df['struct_type'] = 'distMap'
                    fit_aa_dist_df = pd.merge(fit_aa_df,dist_df,on=['model_name','variant_set_name','train_mode'])
                    
                    all_data_frames = [fit_aa_ss_df,fit_aa_rsa_df,fit_aa_dist_df]
                    all_comb_df = reduce(lambda left,right: pd.merge(left,right,on=['model_name','variant_set_name','fitness_R','aa_ppl','aa_ppl_delta_nor','train_mode']), all_data_frames)
                    all_comb_df['ss_rsa_dist_ppl_delta_nor'] = (background_ppl['ss']-all_comb_df['ss_ppl']+background_ppl['rsa']-all_comb_df['rsa_ppl']+background_ppl['distMap']-all_comb_df['distMap_ppl']) / (background_ppl['ss']+background_ppl['rsa']+background_ppl['distMap'])
                    all_comb_df['ss_rsa_dist_ppl_delta_nor_GM'] = all_comb_df[['ss_ppl_delta_nor','rsa_ppl_delta_nor','dist_ppl_delta_nor']].apply(lambda x: np.exp(np.mean(np.log(x))), axis = 1)
                    plt.figure()
                    g = sns.scatterplot(data=all_comb_df, x="aa_ppl_delta_nor", y="ss_rsa_dist_ppl_delta_nor_GM", hue="fitness_R", size="fitness_R", sizes=sizes_value, size_norm=size_norm_range,hue_norm=size_norm_range, alpha=alpha_value, palette=palette_name,zorder=2,legend=False)
                    for model_name in ['rp75_bert_1','rp15_bert_1','rp15_bert_2','rp15_bert_3','rp15_bert_4']:
                        arrow_x = all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='pretrain')]['aa_ppl_delta_nor'].to_list()
                        arrow_y = all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='pretrain')]['ss_rsa_dist_ppl_delta_nor_GM'].to_list()
                        arrow_x_direct = all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='finetune')].reset_index()['aa_ppl_delta_nor'].subtract(all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='pretrain')].reset_index()['aa_ppl_delta_nor']).to_list()
                        arrow_y_direct = all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='finetune')].reset_index()['ss_rsa_dist_ppl_delta_nor_GM'].subtract(all_comb_df.loc[(all_comb_df['model_name']==model_name) & (all_comb_df['train_mode']=='pretrain')].reset_index()['ss_rsa_dist_ppl_delta_nor_GM']).to_list()
                        g.quiver(arrow_x,arrow_y,arrow_x_direct,arrow_y_direct,angles='xy',scale_units='xy',scale=1, width=0.003,alpha=0.5)
                    #plt.legend(title='fitness_R',bbox_to_anchor=(1.03, .63), loc='upper left', borderaxespad=0)
                    g.set(ylabel=None)
                    g.set(xlabel=None)
                    plt.xlim(xlimit_range[0], xlimit_range[1])
                    plt.ylim(ylimit_range[0], ylimit_range[1])
                    plt.savefig(f'eval_results/structure_awareness_eval/figures/family_fit_aa_all_{var_set}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    plt.close()
                    
                    ## family-wise pretrain
                    # for train_mode in ['pretrain', 'finetune']:
                    #     train_mode_all_comb_df = all_comb_df.loc[all_comb_df['train_mode'] == train_mode]
                    #     plt.figure()
                    #     g = sns.scatterplot(data=train_mode_all_comb_df, x="aa_ppl_delta_nor", y="ss_rsa_dist_ppl_delta_nor_GM", hue="fitness_R", size="fitness_R",sizes=(100,500),alpha=0.8,palette=palette_name,zorder=2,legend=False)
                    #     #plt.legend(title='fitness_R',bbox_to_anchor=(1.03, .63), loc='upper left', borderaxespad=0)
                    #     g.set(ylabel=None)
                    #     g.set(xlabel=None)
                    #     #plt.xlim(xlimit_range[0], xlimit_range[1])
                    #     #plt.ylim(ylimit_range[0], ylimit_range[1])
                    #     plt.savefig(f'eval_results/structure_awareness_eval/figures/family_{train_mode}_fit_aa_all_{var_set}_{eval_set}_combined.png', format='png', dpi=800, bbox_inches='tight')
                    #     plt.close()
                    
    return None

def joint_model_fitness_figure():
    """Figures for fitness evaluation, joint modeling
    """
    path = "/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark"
    raw_score_df = pd.read_csv(f"{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/raw_scores/fitness_round1.csv",header=0,delimiter=',')
    col_names = raw_score_df.columns
    reform_raw_score_list = []
    for i, row in raw_score_df.iterrows():
        for col_nm in col_names[1:]:
            reform_raw_score_list.append([row['model_name'],col_nm,row[col_nm]])
    reform_raw_score_df = pd.DataFrame(reform_raw_score_list,columns=['model_name','var_set','fitness_R'])
    fig, ax = plt.subplots()
    #colors = ["#FCF014","#F0BF1F","#F09411","#29F0E9","#37A0F0","#38FF81","#44F041","#96F043"]
    colors = ['peachpuff','sandybrown','darkorange','cyan','blue','lightgreen','lime','green']
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.boxplot(x="model_name",y="fitness_R",data=reform_raw_score_df,palette=customPalette,saturation=0.5)
    g=sns.stripplot(x="model_name",y="fitness_R",data=reform_raw_score_df,color='gray',alpha=.5)
    g.set(ylabel=None)
    g.set(xlabel=None)
    plt.xticks(rotation=45)
    plt.savefig(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/fitness_round1.png', format='png', dpi=800, bbox_inches='tight')
    plt.close()

    return None

if __name__ == '__main__':
  working_dir   = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  task2run = sys.argv[1]
  
  if task2run == 'seq_struct_fun_analysis':
    seq_struct_fun_analysis(report_ave_score=False,draw_figure=False,draw_figure_pre2fine=False,draw_figure_family=True)
  elif task2run == 'joint_model_fitness_figure':
    joint_model_fitness_figure()