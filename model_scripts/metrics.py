import os,re,json,joblib,umap,math
from pathlib import Path
from typing import Sequence, Union, Tuple, Dict
from string import Template
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.stats
from scipy.special import softmax
from mapping import registry
from tokenizers import aaCodes, PFAM_VOCAB, PFAM_VOCAB_20AA_IDX, PFAM_VOCAB_20AA_IDX_MAP
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pylab as pylab
mpl.use('Agg')
import matplotlib.pyplot as plt

@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))

@registry.register_metric('fitness_assess_supervise')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return None


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy_subClass_AB')
@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]],
             normalize: bool = False,
             **kwargs) -> Union[float, Tuple[float, float]]:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top3_subClass_AB')
@registry.register_metric('accuracy_top3')
def accuracy_top3(target: Union[Sequence[int], Sequence[Sequence[int]]],
                  prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                  normalize: bool = False,
                  **kwargs) -> Union[float, Tuple[float, float]]:
    topK = 3
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top5_subClass_AB')
@registry.register_metric('accuracy_top5')
def accuracy_top5(target: Union[Sequence[int], Sequence[Sequence[int]]],
                  prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                  normalize: bool = False,
                  **kwargs) -> Union[float, Tuple[float, float]]:
    topK = 5
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top10_subClass_AB')
@registry.register_metric('accuracy_top10')
def accuracy_top10(target: Union[Sequence[int], Sequence[Sequence[int]]],
                   prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                   normalize: bool = False,
                   **kwargs) -> Union[float, Tuple[float,float]]:
    topK = 10
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('perplexity_subClass_AB')
@registry.register_metric('perplexity')
def perplexity(target: Union[Sequence[int], Sequence[Sequence[int]]],
               prediction: Union[Sequence[float], Sequence[Sequence[float]]],
               normalize: bool = False,
               **kwargs) -> Union[float,Tuple[float,float]]:
  '''
  ECE and perplexity evaluated as token level

  * ECE: exp(mean(per_maskedToken_ce list))
  * perplexity: mean(exp(per_maskedToken_ce list))
  '''
  maskedToken_count = 0.0
  #ce_total = 0
  ece_ce_sum_nn = 0.0  # accumulating sum of ce
  ppl_expCE_sum_nn = 0.0 # accumulating sum of exp(ce)
  for label, score in zip(target, prediction):
    label_array = np.asarray(label)
    #pred_array = np.asarray(score)
    mask = label_array != -1 #[L_max,]
    label_mask_arr = label_array[mask] #[k_maskedTokens,]
    #pred_mask_arr = pred_array[mask] # logit score
    #pred_mask_arr_prob = softmax(pred_mask_arr,axis=-1) # convert to probability
    
    if len(label_mask_arr) == 0:
      #print('no masked pos. label_array: {}'.format(label_array))
      continue
    else:
      # scipy version cross entropy
      #ce_sum = log_loss(label_mask_arr,pred_mask_arr_prob,normalize=False,labels=np.arange(pred_mask_arr.shape[-1]))
      #ce_total += ce_sum
      
      maskedToken_count += label_mask_arr.shape[0]
      # pytorch version cross entropy
      ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
      if len(label_array.shape) == 0:
        score_tensor = torch.from_numpy(np.array(score).reshape((1,-1)))
        label_tensor = torch.from_numpy(np.array(label).reshape(1))
      else:
        score_tensor = torch.from_numpy(score)
        label_tensor = torch.from_numpy(label)
      ce_values_nn = ce_loss(score_tensor,label_tensor).numpy() #[L_max,]
      ece_ce_sum_nn += np.sum(ce_values_nn[mask])
      ppl_expCE_sum_nn += np.sum(np.exp(ce_values_nn[mask]))

  if normalize:
    return (np.exp(ece_ce_sum_nn / maskedToken_count), ppl_expCE_sum_nn / maskedToken_count)
  else:
    return (ece_ce_sum_nn,ppl_expCE_sum_nn,maskedToken_count)

def symm_apc(attenMap):
  """
  symmetrize and apply apc to attention matrix
  attenMap size: [bs,num_head,L_max,L_max]
  reutrn apc_mat size: same
  """
  symm_mat = attenMap + np.transpose(attenMap,axes=(0,1,3,2))
  rowSum_mat = np.sum(symm_mat,axis=-1,keepdims=True)
  colSum_mat = np.sum(symm_mat,axis=-2,keepdims=True)
  allSum_mat = np.sum(symm_mat,axis=(-1,-2),keepdims=True)
  apc_mat = symm_mat - rowSum_mat * colSum_mat / allSum_mat
  return apc_mat


@registry.register_metric('contact_background_prec')
def contact_background_precision(contactMap: np.ndarray,
                                 normalize: bool = False,
                                 **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  cal_range = kwargs['cal_range'] # all, short, medium, long
  valid_mask = kwargs['valid_mask'] # size [bs,L]
  # contactMap size [bs,L,L]
  seq_length = kwargs['seq_length'] # size [bs,]
  valid_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L,L]
  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  if cal_range == 'all':
    #valid_mask_mat &= ((y_ind - x_ind) >= 6)
    valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 6)
    range_mask_mat = ((y_ind - x_ind) >= 6)
  elif cal_range == 'short':
    #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12)
    valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
    range_mask_mat = np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12)
  elif cal_range == 'medium':
    #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24)
    valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
    range_mask_mat = np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24)
  elif cal_range == 'long':
    #valid_mask_mat &= ((y_ind - x_ind) >= 24)
    valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 24)
    range_mask_mat = ((y_ind - x_ind) >= 24)
  else:
    raise Exception("Unexpected range for precision calculation: {}".format(cal_range))
  correct = 0.
  total = 0.
  indiv_prec_list = []
  for length, contM, mask in zip(seq_length, contactMap, valid_mask_mat):
    val_count = np.sum(mask)
    if val_count == 0:
      #total += np.sum(range_mask_mat)
      continue
    else:
      val_contM = contM[mask] # size [val_count,]
      correct += np.sum(val_contM)
      total += val_contM.shape[0]
      indiv_prec_list.append(np.sum(val_contM)/val_contM.shape[0])
  if normalize:
    return correct / total
  else:
    return correct, total, indiv_prec_list

@registry.register_metric('all_pred_distribution')
def all_pred_distribution(contactMap: np.ndarray,
                          attentionMat: np.ndarray,
                          **kwargs) -> Tuple:
  top_cut = kwargs['top_cut']
  symm_way = kwargs['symm_way']
  valid_mask = kwargs['valid_mask']
  seq_length = kwargs['seq_length']
  # attentionMat: [bs,num_layer,num_head,L_max,L_max]
  # contactMap: [bs,L_max,L_max]
  # valid_mask: [bs,L_max]
  # !reshape attentionMat to [num_layer,bs,num_head,L_max,L_max]!
  attentionMat = np.transpose(attentionMat, (1,0,2,3,4))
  correct_list = [] # [num_layer, num_head]
  total_list = [] # [num_layer, num_head]
  correct_range_distr = [] # [num_layer, num_head, 3] (3-short,medium,long)
  
  # loop along 'num_layer'
  for atten_layer_np in attentionMat:
    correct = np.zeros(atten_layer_np.shape[1]) # [num_head,]
    total = np.zeros(atten_layer_np.shape[1])
    # symmetrize attention matrix - max(a_ij,a_ji) or mean(a_ij,a_ji)
    if symm_way == 'max':
      attenMat_symm = np.maximum(atten_layer_np, np.transpose(atten_layer_np, (0,1,3,2)))
    elif symm_way == 'mean':
      attenMat_symm = 0.5 * (atten_layer_np + np.transpose(atten_layer_np, (0,1,3,2)))
    elif symm_way == 'apc':
      attenMat_symm = symm_apc(atten_layer_np)
    else:
      raise Exception("Unexpected symmetrize way for precision calculation: {}".format(symm_way))

    valid_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
    seqpos = np.arange(valid_mask.shape[1])
    x_ind, y_ind = np.meshgrid(seqpos, seqpos) # [L_max,L_max]
    # mask for all range (i-j > 6, each pair is counted once)
    valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 6)
    
    correct_range_distr_layer = np.zeros((attenMat_symm.shape[1],3)) # [n_head, 3]
    # loop through 'bs'    
    for length, atten, contM, mask in zip(seq_length, attenMat_symm, contactMap, valid_mask_mat):
      '''
      masked_atten = (atten * mask).reshape(atten.shape[0],-1) # size [n_head,l_max*l_max]
      most_likely_idx = np.argpartition(-masked_atten,kth=length // top_cut,axis=-1)[:,:(length // top_cut) + 1]
      '''
      val_count = np.sum(mask)
      if val_count == 0:
        #total += np.asarray([length//top_cut]*atten.shape[0]).reshape(-1) # [n_head,]
        continue
      else:
        val_atten = atten[:,mask] # [n_head,val_count]
        val_contM = contM[mask] # [val_count,]
        val_x_ind = x_ind[mask] # [val_count,]
        val_y_ind = y_ind[mask] # [val_count,]

        top_len = min(length//top_cut,val_count)
        most_likely_idx = np.argpartition(-val_atten,kth=top_len-1,axis=-1)[:,:top_len] # [n_head,top_len]

        selected = np.take_along_axis(val_contM.reshape(1,-1), most_likely_idx, axis=-1) # [n_head,top_len]
        sele_x_ind = np.take_along_axis(val_x_ind.reshape(1,-1), most_likely_idx, axis=-1) # [n_head,top_len]
        sele_y_ind = np.take_along_axis(val_y_ind.reshape(1,-1), most_likely_idx, axis=-1) # [n_head,top_len]
        # loop n_head
        for n_h in range(selected.shape[0]):
          selected_oneH = selected[n_h,:].astype(bool) # [top_len,]
          sele_x_ind_con = sele_x_ind[n_h,:][selected_oneH] # [contact_in_top_len,]
          sele_y_ind_con = sele_y_ind[n_h,:][selected_oneH] # [contact_in_top_len,]
          correct_short = np.sum(np.logical_and((sele_y_ind_con - sele_x_ind_con) >= 6,
                                                (sele_y_ind_con - sele_x_ind_con) < 12)) # scaler
          correct_medium = np.sum(np.logical_and((sele_y_ind_con - sele_x_ind_con) >= 12,
                                                 (sele_y_ind_con - sele_x_ind_con) < 24)) # scaler
          correct_long = np.sum((sele_y_ind_con - sele_x_ind_con) >= 24) # scaler
          correct_range_distr_layer[n_h,:] += np.array([correct_short,correct_medium,correct_long])
        
        correct += np.sum(selected,axis=-1).reshape(-1) # [n_head,]
        total += np.asarray([length//top_cut]*selected.shape[0]).reshape(-1) # [n_head,]

    correct_range_distr.append(correct_range_distr_layer) # [n_layer, n_head, 3]
    correct_list.append(correct) # [n_layer, n_head]
    total_list.append(total) #[n_layer, n_head]
  
  return (correct_range_distr, correct_list, total_list)

@registry.register_metric('contact_precision')
def contact_precision(contactMap: np.ndarray,
                     attentionMat: np.ndarray,
                     normalize: bool = False,
                     **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  top_cut = kwargs['top_cut'] # 1,2,5
  symm_way = kwargs['symm_way'] # max, mean
  cal_range = kwargs['cal_range'] # all, short, medium, long
  valid_mask = kwargs['valid_mask']
  seq_length = kwargs['seq_length'] 
  # attentionMat: [bs,num_layer,num_head,L_max,L_max]
  # contactMap: [bs,L_max,L_max]
  # valid_mask: [bs,L_max]
  # reshape attentionMat to [num_layer,bs,num_head,L_max,L_max]
  attentionMat = np.transpose(attentionMat, (1,0,2,3,4))
  correct_list = []
  total_list = []
  indiv_prec_list = [] # [n_layer,bs,n_head]
  for atten_layer_np in attentionMat:
    correct = np.zeros(atten_layer_np.shape[1])
    total = np.zeros(atten_layer_np.shape[1])
    indiv_prec = [] # [bs,n_head]
    # symmetrize attention matrix - max(a_ij,a_ji) or mean(a_ij,a_ji)
    if symm_way == 'max':
      attenMat_symm = np.maximum(atten_layer_np, np.transpose(atten_layer_np, (0,1,3,2)))
    elif symm_way == 'mean':
      attenMat_symm = 0.5 * (atten_layer_np + np.transpose(atten_layer_np, (0,1,3,2)))
    elif symm_way == 'apc':
      attenMat_symm = symm_apc(atten_layer_np)
    else:
      raise Exception("Unexpected symmetrize way for precision calculation: {}".format(symm_way))

    valid_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
    seqpos = np.arange(valid_mask.shape[1])
    x_ind, y_ind = np.meshgrid(seqpos, seqpos)
    if cal_range == 'all':
      #valid_mask_mat &= ((y_ind - x_ind) >= 6)
      valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 6)
    elif cal_range == 'short':
      #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12)
      valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
    elif cal_range == 'medium':
      #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24)
      valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
    elif cal_range == 'long':
      #valid_mask_mat &= ((y_ind - x_ind) >= 24)
      valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 24)
    else:
      raise Exception("Unexpected range for precision calculation: {}".format(cal_range))
    #loop throught batch size
    for length, atten, contM, mask in zip(seq_length, attenMat_symm, contactMap, valid_mask_mat):
      '''
      masked_atten = (atten * mask).reshape(atten.shape[0],-1) # size [n_head,l_max*l_max]
      most_likely_idx = np.argpartition(-masked_atten,kth=length // top_cut,axis=-1)[:,:(length // top_cut) + 1]
      '''
      val_count = np.sum(mask)
      if val_count == 0:
        #total += np.asarray([length//top_cut]*atten.shape[0]).reshape(-1) # still use L/k as demoninator
        continue
      else:
        val_atten = atten[:,mask] # size [n_head,val_count]
        val_contM = contM[mask] # size [val_count,]
        top_len = length//top_cut
        top_len = min(top_len,val_count)
        most_likely_idx = np.argpartition(-val_atten,kth=top_len-1,axis=-1)[:,:top_len] # size [n_head,top_len]
        selected = np.take_along_axis(val_contM.reshape(1,-1), most_likely_idx, axis=-1) #size [n_head,top_len]
        correct += np.sum(selected,axis=-1).reshape(-1) # size [n_head,]
        total += np.asarray([length//top_cut]*selected.shape[0]).reshape(-1) # still use L/k as demoninator,[n_head,]
        #indiv_prec [bs,n_head]
        indiv_prec.append(np.divide(np.sum(selected,axis=-1).reshape(-1),np.asarray([length//top_cut]*selected.shape[0]).reshape(-1)))
    
    correct_list.append(correct)
    total_list.append(total)
    if len(indiv_prec) > 0:
      indiv_prec_list.append(indiv_prec)
  
  # reshape indiv_prec_list from [n_layer,bs,n_head] to [bs,n_layer,n_head]
  #print('indiv_prec_list',np.array(indiv_prec_list).shape)
  if len(indiv_prec_list) > 0:
    indiv_prec_arr = np.transpose(indiv_prec_list,(1,0,2))
  else:
    indiv_prec_arr = None
  if normalize:
    return np.array(correct_list / total_list)
  else:
    return (correct_list, total_list, indiv_prec_arr)

@registry.register_metric('train_logisticRegression_layerwise')
def logisticRegTrainLayerwise(contactMap: np.ndarray,
                              attentionMat: np.ndarray,
                              type_flag: np.ndarray,
                              **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  def sele_pair(atteM, valM):
    # atteM: [n_layer, n_head, L_max, L_max]
    # valM: [L_max,L_max]
    #print('atteM:',atteM.shape)
    #print('valM:',valM.shape)
    #print('valM:',valM)
    fil_atte = atteM[:,:,valM] #[n_layer,n_head,n_valid_pairs]
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
    indiv_prec_list = []
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        #total += length//topk
        continue
      else:
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count,length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) # [top_len,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected)/(length//topk))
    return np.mean(indiv_prec_list) #correct / total

  #np.set_printoptions(threshold=np.inf) 
  valid_mask = kwargs.get('valid_mask')
  #print('valid_mask sum:',np.sum(valid_mask,axis=-1))
  seq_length = kwargs.get('seq_length')
  #print('seq_length:',seq_length)
  data_dir = kwargs.get('data_dir')
  task = kwargs.get('task')
  pretrain_model = kwargs.get('pretrain_model')
  pretrained_epoch = kwargs.get('pretrained_epoch')
  
  if task == 'esm_eval':
    mdl_save_dir = 'logistic_models_esm'
  else:
    if pretrained_epoch is not None:
      epoch_dir = '_{}'.format(pretrained_epoch)
    else:
      epoch_dir = ''
    if not os.path.isdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model,epoch_dir)):
      os.mkdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model))
    if not os.path.isdir('{}/logistic_models/{}{}/layerwise'.format(data_dir,pretrain_model,epoch_dir)):
      os.mkdir('{}/logistic_models/{}{}/layerwise'.format(data_dir,pretrain_model,epoch_dir))
    mdl_save_dir = 'logistic_models/{}{}/layerwise'.format(pretrain_model,epoch_dir)

  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

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
  
  valid_prec_best_layers = [] # [n_layer,n_range]
  # contactMap size: [bs, L_max, L_max]
  # attentionMat size: [bs, n_layer, n_head, L_max, L_max]
  for ly in range(attentionMat.shape[1]):
    print('>layer: {}'.format(ly+1))
    attentionMat_layer = attentionMat[:,ly,:,:,:][:,None,:,:,:] #[bs,1,n_head,L_max,L_max]
    # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
    attenMat_symm = attentionMat_layer + np.transpose(attentionMat_layer, (0,1,2,4,3))
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
    #print('train len:',len(train_idx))
    #print('val len:',len(valid_idx))
    
    # prepare validation set (all, short, medium, long)
    valid_contMap = contactMap[valid_idx] # [n_valid,L_max,L_max]
    valid_atteMat = attenMat_symm_apc[valid_idx] # [n_valid,n_layer,n_head,L_max,L_max]
    valid_all_valMask = val_all_mask_mat[valid_idx]  # [n_valid,L_max,L_max]
    valid_short_valMask = val_short_mask_mat[valid_idx]
    valid_medium_valMask = val_medium_mask_mat[valid_idx]
    valid_long_valMask = val_long_mask_mat[valid_idx]

    #valid pair numbers for each example
    valid_all_valMask_count = np.sum(valid_all_valMask,axis=(-1,-2)) # [n_valid,]
    valid_short_valMask_count = np.sum(valid_short_valMask,axis=(-1,-2))
    valid_medium_valMask_count = np.sum(valid_medium_valMask,axis=(-1,-2))
    valid_long_valMask_count = np.sum(valid_long_valMask,axis=(-1,-2))

    # select working pairs
    valid_all_valContMap = valid_contMap[valid_all_valMask] # [n_valid_pairs,]
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

    #l1_weight_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
    l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10]
    train_num_list = [16,17,18,19,20] 
    for train_num in train_num_list:
    #for train_num in range(1,3):
      print('>>train_num',train_num)
      train_contMap = contactMap[train_idx[0:train_num]] # [n_train,L_max,L_max]
      train_atteMat = attenMat_symm_apc[train_idx[0:train_num]] # [n_train/valid,n_layer,n_head,L_max,L_max]
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
        print('>>>l1_weight:',l1w)
        mdl = LogisticRegression(penalty='l1',C=l1w,random_state=0,tol=0.001,solver='saga',max_iter=1000).fit(train_valAtteMat,train_valContMap)
        joblib.dump(mdl,'{}/{}/model_{}_{}_{}'.format(data_dir,mdl_save_dir,ly+1,train_num,l1w))
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

    # find best model for each range
    valid_mode = ['all','short','medium','long']
    valid_prec_idxs = [valid_prec_all_best_idx,valid_prec_short_best_idx,valid_prec_medium_best_idx,valid_prec_long_best_idx]
    valid_prec_mat = [valid_prec_all,valid_prec_short,valid_prec_medium,valid_prec_long]
    valid_prec_max, valid_prec_min = np.amax(valid_prec_mat), np.amin(valid_prec_mat)
    valid_prec_best = []
    for i in range(len(valid_mode)):
      idx_tuple = valid_prec_idxs[i]
      valid_prec_best.append(np.amax(valid_prec_mat[i]))
      num_best = train_num_list[idx_tuple[0]]
      l1w_best = l1_weight_list[idx_tuple[1]]
      mode = valid_mode[i]
      print('>_mode: {}; num_best: {}; l1w_best: {}'.format(mode,num_best,l1w_best))
   
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
      plt.savefig('{}/{}/gridSear_model_{}_{}.png'.format(data_dir,mdl_save_dir,ly+1,mode))
      plt.close(fig)

      # save validation set precision 
      np.savetxt('{}/{}/gridSear_model_{}_{}.csv'.format(data_dir,mdl_save_dir,ly+1,mode),valid_prec_mat[i],fmt='%.3f',delimiter=',')
    valid_prec_best_layers.append(valid_prec_best)
  return np.array(valid_prec_best_layers)

@registry.register_metric('train_logisticRegression_layersupervise')
def logisticRegTrainLayersupervise(contactMap: np.ndarray,
                                   attentionMat: np.ndarray,
                                   type_flag: np.ndarray,
                                   **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  def sele_pair(atteM, valM):
    # atteM: [n_layer, n_head, L_max, L_max]
    # valM: [L_max,L_max]
    fil_atte = atteM[:,:,valM] #[n_layer,n_head,n_valid_pairs]
    fil_atte_shape = fil_atte.shape
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
    indiv_prec_list = []
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        #total += length//topk
        continue
      else:
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count,length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) # [top_len,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected)/(length//topk))
    return np.mean(indiv_prec_list) #correct / total
  
  #np.set_printoptions(threshold=np.inf) 
  valid_mask = kwargs.get('valid_mask')
  #print('valid_mask sum:',np.sum(valid_mask,axis=-1))
  seq_length = kwargs.get('seq_length')
  #print('seq_length:',seq_length)
  data_dir = kwargs.get('data_dir')
  task = kwargs.get('task')
  pretrain_model = kwargs.get('pretrain_model')
  pretrained_epoch = kwargs.get('pretrained_epoch')
  head_selector = np.array(kwargs.get('head_selector')).astype(bool) # [n_layer,n_head]
  
  if task == 'esm_eval':
    mdl_save_dir = 'logistic_models_esm'
  else:
    if pretrained_epoch is not None:
      epoch_dir = '_{}'.format(pretrained_epoch)
    else:
      epoch_dir = ''
    if not os.path.isdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model,epoch_dir)):
      os.mkdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model,epoch_dir))
    if not os.path.isdir('{}/logistic_models/{}{}/layersupervise'.format(data_dir,pretrain_model,epoch_dir)):
      os.mkdir('{}/logistic_models/{}{}/layersupervise'.format(data_dir,pretrain_model,epoch_dir))
    mdl_save_dir = 'logistic_models/{}{}/layersupervise'.format(pretrain_model,epoch_dir)

  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

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

  # filter out supervised heads
  attentionMat_layer = attentionMat[:,head_selector,:,:][:,None,:,:,:] #[bs,1,n_head_SV,L_max,L_max]
  # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
  attenMat_symm = attentionMat_layer + np.transpose(attentionMat_layer, (0,1,2,4,3))
  # rowSum/colSum size: [bs,1,n_head_SV,L_max]; allSum [bs,1,n_head_SV];
  # all broadcast to [bs,1,n_head_SV,L_max,L_max]
  attenMat_symm_rowSum = np.repeat(np.sum(attenMat_symm, axis=-1)[:,:,:,:,None],attenMat_symm.shape[-1],axis=-1)
  attenMat_symm_colSum = np.repeat(np.sum(attenMat_symm, axis=-2)[:,:,:,None,:],attenMat_symm.shape[-1],axis=-2)
  attenMat_symm_allSum = np.repeat(np.repeat(np.sum(attenMat_symm,axis=(-2,-1))[:,:,:,None,None],attenMat_symm.shape[-1],axis=-1),attenMat_symm.shape[-1],axis=-2)
  # attenMat_symm_apc [bs, 1, n_head_SV, L_max, L_max]
  #print('attenMat_symm:',attenMat_symm.shape)
  #print('attenMat_symm_rowSum:',attenMat_symm_rowSum.shape)
  #print('attenMat_symm_colSum:',attenMat_symm_colSum.shape)
  #print('attenMat_symm_allSum:',attenMat_symm_allSum.shape)
  attenMat_symm_apc = attenMat_symm-attenMat_symm_rowSum*attenMat_symm_colSum/attenMat_symm_allSum
  
  # split train / valid set
  train_idx = np.squeeze(np.argwhere(type_flag == 1))
  valid_idx = np.squeeze(np.argwhere(type_flag == 0))
  #print('train len:',len(train_idx))
  #print('val len:',len(valid_idx))
  
  # prepare validation set (all, short, medium, long)
  valid_contMap = contactMap[valid_idx] # [n_valid,L_max,L_max]
  valid_atteMat = attenMat_symm_apc[valid_idx] # [n_valid,1,n_head_SV,L_max,L_max]
  valid_all_valMask = val_all_mask_mat[valid_idx]  # [n_valid,L_max,L_max]
  valid_short_valMask = val_short_mask_mat[valid_idx]
  valid_medium_valMask = val_medium_mask_mat[valid_idx]
  valid_long_valMask = val_long_mask_mat[valid_idx]

  #valid pair numbers for each example
  valid_all_valMask_count = np.sum(valid_all_valMask,axis=(-1,-2)) # [n_valid,]
  valid_short_valMask_count = np.sum(valid_short_valMask,axis=(-1,-2))
  valid_medium_valMask_count = np.sum(valid_medium_valMask,axis=(-1,-2))
  valid_long_valMask_count = np.sum(valid_long_valMask,axis=(-1,-2))

  # select working pairs
  valid_all_valContMap = valid_contMap[valid_all_valMask] # [n_valid_pairs,]
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

  #l1_weight_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10]
  train_num_list = [16,17,18,19,20] 
  for train_num in train_num_list:
  #for train_num in range(1,3):
    print('>>train_num',train_num)
    train_contMap = contactMap[train_idx[0:train_num]] # [n_train,L_max,L_max]
    train_atteMat = attenMat_symm_apc[train_idx[0:train_num]] # [n_train/valid,1,n_head_SV,L_max,L_max]
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
      print('>>>l1_weight:',l1w)
      mdl = LogisticRegression(penalty='l1',C=l1w,random_state=0,tol=0.001,solver='saga',max_iter=1000).fit(train_valAtteMat,train_valContMap)
      joblib.dump(mdl,'{}/{}/model_{}_{}'.format(data_dir,mdl_save_dir,train_num,l1w))
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

  # find best model for each range
  valid_mode = ['all','short','medium','long']
  valid_prec_idxs = [valid_prec_all_best_idx,valid_prec_short_best_idx,valid_prec_medium_best_idx,valid_prec_long_best_idx]
  valid_prec_mat = [valid_prec_all,valid_prec_short,valid_prec_medium,valid_prec_long]
  valid_prec_max, valid_prec_min = np.amax(valid_prec_mat), np.amin(valid_prec_mat)
  valid_prec_best = [] # [4(range),]
  for i in range(len(valid_mode)):
    idx_tuple = valid_prec_idxs[i]
    valid_prec_best.append(np.amax(valid_prec_mat[i]))
    num_best = train_num_list[idx_tuple[0]]
    l1w_best = l1_weight_list[idx_tuple[1]]
    mode = valid_mode[i]
    print('>_mode: {}; num_best: {}; l1w_best: {}'.format(mode,num_best,l1w_best))
 
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
    plt.savefig('{}/{}/gridSear_model_{}.png'.format(data_dir,mdl_save_dir,mode))
    plt.close(fig)

    # save validation set precision 
    np.savetxt('{}/{}/gridSear_model_{}.csv'.format(data_dir,mdl_save_dir,mode),valid_prec_mat[i],fmt='%.3f',delimiter=',')
  return np.array(valid_prec_best)


@registry.register_metric('train_logisticRegression')
def logisticRegTrain(contactMap: np.ndarray,
                     attentionMat: np.ndarray,
                     type_flag: np.ndarray,
                     **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  #np.set_printoptions(threshold=np.inf) 
  valid_mask = kwargs.get('valid_mask')
  #print('valid_mask sum:',np.sum(valid_mask,axis=-1))
  seq_length = kwargs.get('seq_length')
  #print('seq_length:',seq_length)
  data_dir = kwargs.get('data_dir')
  task = kwargs.get('task')
  pretrained_epoch = kwargs.get('pretrained_epoch')
  pretrain_model = kwargs.get('pretrain_model')
  
  if task == 'esm_eval':
    mdl_save_dir = 'logistic_models_esm'
  else:
    if pretrained_epoch is not None:
      epoch_dir = '_{}'.format(pretrained_epoch)
    else:
      epoch_dir = ''
    if not os.path.isdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model,epoch_dir)):
      os.mkdir('{}/logistic_models/{}{}'.format(data_dir,pretrain_model,epoch_dir))  
    mdl_save_dir = 'logistic_models/{}{}'.format(pretrain_model,epoch_dir)

  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

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
  attenMat_symm = attentionMat + np.transpose(attentionMat, (0,1,2,4,3))
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
  #print('train len:',len(train_idx))
  #print('val len:',len(valid_idx))
  
  # prepare validation set (all, short, medium, long)
  valid_contMap = contactMap[valid_idx]
  valid_atteMat = attenMat_symm_apc[valid_idx]
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
    indiv_prec_list = []
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        #total += length//topk
        continue
      else:
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count,length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) # [top_len,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected)/(length//topk))
    return np.mean(indiv_prec_list) #correct / total

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

  #l1_weight_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10]
  train_num_list = [16,17,18,19,20] 
  for train_num in train_num_list:
  #for train_num in range(1,3):
    print('train_num',train_num)
    train_contMap = contactMap[train_idx[0:train_num]] # [n_train,L_max,L_max]
    train_atteMat = attenMat_symm_apc[train_idx[0:train_num]] # [n_train/valid,n_layer,n_head,L_max,L_max]
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
      mdl = LogisticRegression(penalty='l1',C=l1w,random_state=0,tol=0.001,solver='saga',max_iter=1000).fit(train_valAtteMat,train_valContMap)
      joblib.dump(mdl,'{}/{}/model_{}_{}'.format(data_dir,mdl_save_dir,train_num,l1w))
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
    num_best = train_num_list[idx_tuple[0]]
    l1w_best = l1_weight_list[idx_tuple[1]]
    mode = valid_mode[i]
    print('>_mode: {}; num_best: {}; l1w_best: {}'.format(mode,num_best,l1w_best))
 
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
    plt.savefig('{}/{}/gridSear_model_{}.png'.format(data_dir,mdl_save_dir,mode))
    plt.close(fig)

    # save validation set precision 
    np.savetxt('{}/{}/gridSear_model_{}.csv'.format(data_dir,mdl_save_dir,mode),valid_prec_mat[i],fmt='%.3f',delimiter=',')
  return valid_prec_best    
  

@registry.register_metric('test_logisticRegression')
def logisticRegTest(contactMap: np.ndarray,
                    attentionMat: np.ndarray,
                    best_mdl_set: Sequence,
                    **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  data_dir = kwargs.get('data_dir')
  valid_mask = kwargs.get('valid_mask')
  seq_length = kwargs.get('seq_length')
  mdl_save_dir = kwargs.get('mdl_save_dir')
  pretrain_model = kwargs.get('pretrain_model')
  pretrained_epoch = kwargs.get('pretrained_epoch')

  if pretrained_epoch is not None:
    epoch_dir = '_{}'.format(pretrained_epoch)
  else:
    epoch_dir = ''

  l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10] #[0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  train_num_list = [16,17,18,19,20]
  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  #print('(y_ind - x_ind) >= 6',(y_ind - x_ind) >= 6)
  all_mask_mat = np.logical_and(val_all_mask_mat, (y_ind - x_ind)>=6)
  short_mask_mat = np.logical_and(val_short_mask_mat, np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
  medium_mask_mat = np.logical_and(val_medium_mask_mat, np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
  long_mask_mat = np.logical_and(val_long_mask_mat, (y_ind - x_ind) >= 24)

  #valid pair numbers for each example
  all_valMask_count = np.sum(all_mask_mat,axis=(-1,-2)) # [bs,]
  short_valMask_count = np.sum(short_mask_mat,axis=(-1,-2))
  medium_valMask_count = np.sum(medium_mask_mat,axis=(-1,-2))
  long_valMask_count = np.sum(long_mask_mat,axis=(-1,-2))
  
  # get indices of seqs with non-zero valid pairs
  all_valMask_non0_idx = np.squeeze(np.argwhere(all_valMask_count > 0))
  short_valMask_non0_idx = np.squeeze(np.argwhere(short_valMask_count > 0))
  medium_valMask_non0_idx = np.squeeze(np.argwhere(medium_valMask_count > 0))
  long_valMask_non0_idx = np.squeeze(np.argwhere(long_valMask_count > 0))
  

  # contactMap size: [bs, L_max, L_max]
  # attentionMat size: [bs, n_layer, n_head, L_max, L_max]
  # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
  attenMat_symm = attentionMat + np.transpose(attentionMat, (0,1,2,4,3))
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

  def sele_pair(atteM, valM):
    #print('atteM:',atteM.shape)
    #print('valM:',valM.shape)
    #print('valM sum:',np.sum(valM))
    fil_atte = atteM[:,:,valM]
    fil_atte_shape = fil_atte.shape
    #print('fil_atte_shape:',fil_atte_shape)
    if np.sum(valM) == 0:
      return None
    else:
      return fil_atte.reshape(-1,fil_atte_shape[-1]).transpose(1,0).tolist()

  def calc_prec(targs,preds,mask_count,lengths,topk):
    """
    targ: true contactMap with value as 0/1, [total num of valid pairs,]
    pred: predicted prob for class 1, [total num of valid pairs,]
    mask_count: num of valid pairs for each example, [n_examples,] 
    topk: L/topk
    """
    correct = 0.
    total = 0.
    indiv_prec_list = []
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        total += 0.
        indiv_prec_list.append(np.nan)
      else:
        #print('count:{},length:{}'.format(count,length))
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count, length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) #size[seq_length,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected) / (length//topk))
    return correct, total, indiv_prec_list

  # only handle sequences with >0 working_pairs
  # select working pairs
  all_valContMap = contactMap[all_mask_mat]
  short_valContMap = contactMap[short_mask_mat]
  medium_valContMap = contactMap[medium_mask_mat]
  long_valContMap = contactMap[long_mask_mat]

  #print(list(map(sele_pair,attentionMat,all_mask_mat)))
 
  all_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,all_mask_mat)))) if np.sum(all_valMask_count) >0 else None
  short_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,short_mask_mat)))) if np.sum(short_valMask_count) >0 else None
  medium_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,medium_mask_mat)))) if np.sum(medium_valMask_count) >0 else None
  long_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,long_mask_mat)))) if np.sum(long_valMask_count) >0 else None


  prec_set = [] #[k_range(4),k_top(3),2]
  indiv_prec_set = [] #[k_range(4),k_top(3),bs]
  # load best model
  valContMap = [all_valContMap,short_valContMap,medium_valContMap,long_valContMap]
  valAtteMat = [all_valAtteMat,short_valAtteMat,medium_valAtteMat,long_valAtteMat]
  valMask_count = [all_valMask_count,short_valMask_count,medium_valMask_count,long_valMask_count]
  mode_list = ['all','short','medium','long']
  for para_i in range(len(mode_list)):
    if np.sum(valMask_count[para_i]) == 0:
      prec_set.append([[0,0],[0,0],[0,0]])
      indiv_prec_set.append(np.full([3,attenMat_symm_apc.shape[0]],np.nan))
      continue
    mode = mode_list[para_i]
    if len(best_mdl_set) > 0:
      para = best_mdl_set[para_i]
      best_num = para[0]
      best_l1w = para[1]
      mdl=joblib.load('{}/{}/{}{}/model_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,best_num,best_l1w))
    else:
      gridSearch = np.loadtxt('{}/{}/{}{}/gridSear_model_{}.csv'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,mode),dtype='float',delimiter=',')  
      valid_prec_best_idx = np.unravel_index(gridSearch.argmax(), gridSearch.shape)
      best_num = train_num_list[valid_prec_best_idx[0]]
      best_l1w = l1_weight_list[valid_prec_best_idx[1]]
      mdl=joblib.load('{}/{}/{}{}/model_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,best_num,best_l1w))
           
    posi_label_idx = np.argwhere(mdl.classes_ == 1)[0][0]
    # predict prob
    contM_pred = mdl.predict_proba(valAtteMat[para_i])
    # calculate precision
    prec_topk_set = [] # size [3,2]
    indiv_prec_topk_set = [] # [3,bs]
    for k in [1,2,5]:
      corr, total, indiv_prec = calc_prec(valContMap[para_i],contM_pred[:,posi_label_idx],valMask_count[para_i],seq_length,k)
      prec_topk_set.append([corr, total])
      indiv_prec_topk_set.append(indiv_prec) # indiv_prec: [bs,]
    prec_set.append(prec_topk_set)
    indiv_prec_set.append(np.reshape(indiv_prec_topk_set,(3,-1)))
  #print("indiv_prec_set:{},{},{},{},{}".format(len(indiv_prec_set),indiv_prec_set[0].shape,indiv_prec_set[1].shape,indiv_prec_set[2].shape,indiv_prec_set[3].shape))
  prec_set = np.asarray(prec_set)
  indiv_prec_set = np.asarray(indiv_prec_set)
  return prec_set, indiv_prec_set

@registry.register_metric('test_logisticRegression_layerwise')
def logisticRegTestLayerwise(contactMap: np.ndarray,
                             attentionMat: np.ndarray,
                             best_mdl_set: Sequence,
                             **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:

  def sele_pair(atteM, valM):
    #print('atteM:',atteM.shape)
    #print('valM:',valM.shape)
    #print('valM sum:',np.sum(valM))
    fil_atte = atteM[:,:,valM]
    fil_atte_shape = fil_atte.shape
    #print('fil_atte_shape:',fil_atte_shape)
    if np.sum(valM) == 0:
      return None
    else:
      return fil_atte.reshape(-1,fil_atte_shape[-1]).transpose(1,0).tolist()

  def calc_prec(targs,preds,mask_count,lengths,topk):
    """
    targ: true contactMap with value as 0/1, [total num of valid pairs,]
    pred: predicted prob for class 1, [total num of valid pairs,]
    mask_count: num of valid pairs for each example, [n_examples,] 
    topk: L/topk
    """
    correct = 0.
    total = 0.
    indiv_prec_list = [] #[bs,]
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        total += 0.
        indiv_prec_list.append(np.nan)
      else:
        #print('count:{},length:{}'.format(count,length))
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count, length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) #size[seq_length,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected) / (length//topk))
    return correct, total, indiv_prec_list

  
  data_dir = kwargs.get('data_dir')
  valid_mask = kwargs.get('valid_mask')
  seq_length = kwargs.get('seq_length')
  mdl_save_dir = kwargs.get('mdl_save_dir')
  pretrain_model = kwargs.get('pretrain_model')
  pretrained_epoch = kwargs.get('pretrained_epoch')
  
  if pretrained_epoch is not None:
      epoch_dir = '_{}'.format(pretrained_epoch)
  else:
      epoch_dir = ''
  
  l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10] #[0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  train_num_list = [16,17,18,19,20]
  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  #print('(y_ind - x_ind) >= 6',(y_ind - x_ind) >= 6)
  all_mask_mat = np.logical_and(val_all_mask_mat, (y_ind - x_ind)>=6)
  short_mask_mat = np.logical_and(val_short_mask_mat, np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
  medium_mask_mat = np.logical_and(val_medium_mask_mat, np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
  long_mask_mat = np.logical_and(val_long_mask_mat, (y_ind - x_ind) >= 24)

  #valid pair numbers for each example
  all_valMask_count = np.sum(all_mask_mat,axis=(-1,-2)) # [bs,]
  short_valMask_count = np.sum(short_mask_mat,axis=(-1,-2))
  medium_valMask_count = np.sum(medium_mask_mat,axis=(-1,-2))
  long_valMask_count = np.sum(long_mask_mat,axis=(-1,-2))
  
  # get indices of seqs with non-zero valid pairs
  all_valMask_non0_idx = np.squeeze(np.argwhere(all_valMask_count > 0))
  short_valMask_non0_idx = np.squeeze(np.argwhere(short_valMask_count > 0))
  medium_valMask_non0_idx = np.squeeze(np.argwhere(medium_valMask_count > 0))
  long_valMask_non0_idx = np.squeeze(np.argwhere(long_valMask_count > 0))
  

  # contactMap size: [bs, L_max, L_max]
  # attentionMat size: [bs, n_layer, n_head, L_max, L_max]
  prec_set_layer = [] #[n_layer(4),k_range(4),k_top(3),2(corr,total)]
  indiv_prec_set_layer = [] #[n_layer(4),k_range(4),k_top(3),bs]
  for ly in range(attentionMat.shape[1]):
    print('layer: {}'.format(ly+1))
    attentionMat_layer = attentionMat[:,ly,:,:,:][:,None,:,:,:]
    # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
    attenMat_symm = attentionMat_layer + np.transpose(attentionMat_layer, (0,1,2,4,3))
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


    # only handle sequences with >0 working_pairs
    # select working pairs
    all_valContMap = contactMap[all_mask_mat]
    short_valContMap = contactMap[short_mask_mat]
    medium_valContMap = contactMap[medium_mask_mat]
    long_valContMap = contactMap[long_mask_mat]

    #print(list(map(sele_pair,attentionMat,all_mask_mat)))
   
    all_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,all_mask_mat)))) if np.sum(all_valMask_count) >0 else None
    short_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,short_mask_mat)))) if np.sum(short_valMask_count) >0 else None
    medium_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,medium_mask_mat)))) if np.sum(medium_valMask_count) >0 else None
    long_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,long_mask_mat)))) if np.sum(long_valMask_count) >0 else None


    prec_set = [] # [k_range(4),k_top(3),2(corr,total)]
    indiv_prec_set = [] # [k_range(4),k_top(3),bs]
    # load best model
    valContMap = [all_valContMap,short_valContMap,medium_valContMap,long_valContMap]
    valAtteMat = [all_valAtteMat,short_valAtteMat,medium_valAtteMat,long_valAtteMat]
    valMask_count = [all_valMask_count,short_valMask_count,medium_valMask_count,long_valMask_count]
    mode_list = ['all','short','medium','long']
    for para_i in range(len(mode_list)):
      if np.sum(valMask_count[para_i]) == 0:
        prec_set.append([[0,0],[0,0],[0,0]]) #topK
        indiv_prec_set.append(np.full([3,attenMat_symm_apc.shape[0]],np.nan))
        continue
      mode = mode_list[para_i]
      if len(best_mdl_set) > 0:
        para = best_mdl_set[ly][para_i]
        best_num = para[0]
        best_l1w = para[1]
        mdl=joblib.load('{}/{}/{}{}/layerwise/model_{}_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,ly+1,best_num,best_l1w))
      else:
        gridSearch = np.loadtxt('{}/{}/{}{}/layerwise/gridSear_model_{}_{}.csv'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,ly+1,mode),dtype='float',delimiter=',')  
        valid_prec_best_idx = np.unravel_index(gridSearch.argmax(), gridSearch.shape)
        best_num = train_num_list[valid_prec_best_idx[0]]
        best_l1w = l1_weight_list[valid_prec_best_idx[1]]
        mdl=joblib.load('{}/{}/{}{}/layerwise/model_{}_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,ly+1,best_num,best_l1w))
             
      posi_label_idx = np.argwhere(mdl.classes_ == 1)[0][0]
      # predict prob
      contM_pred = mdl.predict_proba(valAtteMat[para_i])
      # calculate precision
      prec_topk_set = [] # size [3,2]
      indiv_prec_topk_set = [] # [3,bs]
      for k in [1,2,5]:
        corr, total, indiv_prec = calc_prec(valContMap[para_i],contM_pred[:,posi_label_idx],valMask_count[para_i],seq_length,k)
        prec_topk_set.append([corr, total])
        indiv_prec_topk_set.append(indiv_prec) # indiv_prec: [bs,]
      # append
      prec_set.append(prec_topk_set)
      indiv_prec_set.append(np.reshape(indiv_prec_topk_set,(3,-1)))
    # append
    prec_set_layer.append(prec_set)
    indiv_prec_set_layer.append(indiv_prec_set)

  prec_set_layer = np.asarray(prec_set_layer) #[n_layer(4),k_range(4),k_top(3),2(corr,total)]
  indiv_prec_set_layer = np.asarray(indiv_prec_set_layer) #[n_layer(4),k_range(4),k_top(3),bs]
  return prec_set_layer,indiv_prec_set_layer


@registry.register_metric('test_logisticRegression_layersupervise')
def logisticRegTestLayersupervise(contactMap: np.ndarray,
                                  attentionMat: np.ndarray,
                                  best_mdl_set: Sequence,
                                  **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:

  def sele_pair(atteM, valM):
    #print('atteM:',atteM.shape)
    #print('valM:',valM.shape)
    #print('valM sum:',np.sum(valM))
    fil_atte = atteM[:,:,valM]
    fil_atte_shape = fil_atte.shape
    #print('fil_atte_shape:',fil_atte_shape)
    if np.sum(valM) == 0:
      return None
    else:
      return fil_atte.reshape(-1,fil_atte_shape[-1]).transpose(1,0).tolist()

  def calc_prec(targs,preds,mask_count,lengths,topk):
    """
    targ: true contactMap with value as 0/1, [total num of valid pairs,]
    pred: predicted prob for class 1, [total num of valid pairs,]
    mask_count: num of valid pairs for each example, [n_examples,] 
    topk: L/topk
    """
    correct = 0.
    total = 0.
    indiv_prec_list = [] #[bs,]
    cunt_sum = 0
    for i in range(len(mask_count)):
      count = mask_count[i]
      length = lengths[i]
      if count == 0:
        total += 0.
        indiv_prec_list.append(np.nan)
      else:
        #print('count:{},length:{}'.format(count,length))
        targ_set = targs[cunt_sum:cunt_sum+count]
        pred_set = preds[cunt_sum:cunt_sum+count]
        cunt_sum += count
        top_len = min(count, length//topk)
        most_likely_idx = np.argpartition(-pred_set,kth=top_len-1)[:top_len]
        selected = np.take_along_axis(targ_set, most_likely_idx, axis=0) #size[seq_length,]
        correct += np.sum(selected)
        total += length//topk
        indiv_prec_list.append(np.sum(selected) / (length//topk))
    return correct, total, indiv_prec_list

  
  data_dir = kwargs.get('data_dir')
  valid_mask = kwargs.get('valid_mask')
  seq_length = kwargs.get('seq_length')
  mdl_save_dir = kwargs.get('mdl_save_dir')
  pretrain_model = kwargs.get('pretrain_model')
  pretrained_epoch = kwargs.get('pretrained_epoch')
  head_selector = np.array(kwargs.get('head_selector')).astype(bool) # [n_layer,n_head]

  if pretrained_epoch is not None:
    epoch_dir = '_{}'.format(pretrained_epoch)
  else:
    epoch_dir = ''

  l1_weight_list = [0.001,0.01,0.05,0.08,0.1,0.12,0.15,0.17,0.2,0.5,1,5,10] #[0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,50,100]
  train_num_list = [16,17,18,19,20]
  # valid_mask size [bs,L_max] 
  val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  val_all_mask_mat = val_mask_mat
  val_short_mask_mat = val_mask_mat
  val_medium_mask_mat = val_mask_mat
  val_long_mask_mat = val_mask_mat

  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  #print('(y_ind - x_ind) >= 6',(y_ind - x_ind) >= 6)
  all_mask_mat = np.logical_and(val_all_mask_mat, (y_ind - x_ind)>=6)
  short_mask_mat = np.logical_and(val_short_mask_mat, np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
  medium_mask_mat = np.logical_and(val_medium_mask_mat, np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
  long_mask_mat = np.logical_and(val_long_mask_mat, (y_ind - x_ind) >= 24)

  #valid pair numbers for each example
  all_valMask_count = np.sum(all_mask_mat,axis=(-1,-2)) # [bs,]
  short_valMask_count = np.sum(short_mask_mat,axis=(-1,-2))
  medium_valMask_count = np.sum(medium_mask_mat,axis=(-1,-2))
  long_valMask_count = np.sum(long_mask_mat,axis=(-1,-2))
  
  # get indices of seqs with non-zero valid pairs
  all_valMask_non0_idx = np.squeeze(np.argwhere(all_valMask_count > 0))
  short_valMask_non0_idx = np.squeeze(np.argwhere(short_valMask_count > 0))
  medium_valMask_non0_idx = np.squeeze(np.argwhere(medium_valMask_count > 0))
  long_valMask_non0_idx = np.squeeze(np.argwhere(long_valMask_count > 0))
  

  # contactMap size: [bs, L_max, L_max]
  # attentionMat size: [bs, n_layer, n_head, L_max, L_max]
    
  attentionMat_layer = attentionMat[:,head_selector,:,:][:,None,:,:,:] #[bs,1,n_head_SV,L_max,L_max]
  # symmetrize attentionMat, then APC F_ij_apc = F_ij - F_i*F_j/F
  attenMat_symm = attentionMat_layer + np.transpose(attentionMat_layer, (0,1,2,4,3))
  # rowSum/colSum size: [bs,1,n_head_SV,L_max]; allSum [bs,1,n_head_SV];
  # all broadcast to [bs,1,n_head_SV,L_max,L_max]
  attenMat_symm_rowSum = np.repeat(np.sum(attenMat_symm, axis=-1)[:,:,:,:,None],attenMat_symm.shape[-1],axis=-1)
  attenMat_symm_colSum = np.repeat(np.sum(attenMat_symm, axis=-2)[:,:,:,None,:],attenMat_symm.shape[-1],axis=-2)
  attenMat_symm_allSum = np.repeat(np.repeat(np.sum(attenMat_symm,axis=(-2,-1))[:,:,:,None,None],attenMat_symm.shape[-1],axis=-1),attenMat_symm.shape[-1],axis=-2)
  # attenMat_symm_apc [bs, 1, n_head_SV, L_max, L_max]
  #print('attenMat_symm:',attenMat_symm.shape)
  #print('attenMat_symm_rowSum:',attenMat_symm_rowSum.shape)
  #print('attenMat_symm_colSum:',attenMat_symm_colSum.shape)
  #print('attenMat_symm_allSum:',attenMat_symm_allSum.shape)
  attenMat_symm_apc = attenMat_symm-attenMat_symm_rowSum*attenMat_symm_colSum/attenMat_symm_allSum


  # only handle sequences with >0 working_pairs
  # select working pairs
  all_valContMap = contactMap[all_mask_mat]
  short_valContMap = contactMap[short_mask_mat]
  medium_valContMap = contactMap[medium_mask_mat]
  long_valContMap = contactMap[long_mask_mat]

  #print(list(map(sele_pair,attentionMat,all_mask_mat)))
 
  all_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,all_mask_mat)))) if np.sum(all_valMask_count) >0 else None
  short_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,short_mask_mat)))) if np.sum(short_valMask_count) >0 else None
  medium_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,medium_mask_mat)))) if np.sum(medium_valMask_count) >0 else None
  long_valAtteMat = np.vstack(list(filter(None,map(sele_pair,attenMat_symm_apc,long_mask_mat)))) if np.sum(long_valMask_count) >0 else None
  
  # return vars
  prec_set = [] #[k_range(4),k_top(3),2(corr,total)]
  indiv_prec_set= [] #[k_range(4),k_top(3),bs]

  # load best model
  valContMap = [all_valContMap,short_valContMap,medium_valContMap,long_valContMap]
  valAtteMat = [all_valAtteMat,short_valAtteMat,medium_valAtteMat,long_valAtteMat]
  valMask_count = [all_valMask_count,short_valMask_count,medium_valMask_count,long_valMask_count]
  mode_list = ['all','short','medium','long']
  for para_i in range(len(mode_list)):
    if np.sum(valMask_count[para_i]) == 0:
      prec_set.append([[0,0],[0,0],[0,0]]) #topK
      indiv_prec_set.append(np.full([3,attenMat_symm_apc.shape[0]],np.nan))
      continue
    mode = mode_list[para_i]
    if len(best_mdl_set) > 0:
      para = best_mdl_set[para_i]
      best_num = para[0]
      best_l1w = para[1]
      mdl=joblib.load('{}/{}/{}{}/layersupervise/model_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,best_num,best_l1w))
    else:
      gridSearch = np.loadtxt('{}/{}/{}{}/layersupervise/gridSear_model_{}.csv'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,mode),dtype='float',delimiter=',')  
      valid_prec_best_idx = np.unravel_index(gridSearch.argmax(), gridSearch.shape)
      best_num = train_num_list[valid_prec_best_idx[0]]
      best_l1w = l1_weight_list[valid_prec_best_idx[1]]
      mdl=joblib.load('{}/{}/{}{}/layersupervise/model_{}_{}'.format(data_dir,mdl_save_dir,pretrain_model,epoch_dir,best_num,best_l1w))
           
    posi_label_idx = np.argwhere(mdl.classes_ == 1)[0][0]
    # predict prob
    contM_pred = mdl.predict_proba(valAtteMat[para_i])
    # calculate precision
    prec_topk_set = [] # size [3,2]
    indiv_prec_topk_set = [] # [3,bs]
    for k in [1,2,5]:
      corr, total, indiv_prec = calc_prec(valContMap[para_i],contM_pred[:,posi_label_idx],valMask_count[para_i],seq_length,k)
      prec_topk_set.append([corr, total])
      indiv_prec_topk_set.append(indiv_prec) # indiv_prec: [bs,]
    # append
    prec_set.append(prec_topk_set)
    indiv_prec_set.append(np.reshape(indiv_prec_topk_set,(3,-1)))

  prec_set = np.asarray(prec_set) #[k_range(4),k_top(3),2(corr,total)]
  indiv_prec_set = np.asarray(indiv_prec_set) #[k_range(4),k_top(3),bs]
  return prec_set,indiv_prec_set

@registry.register_metric('logisContact_esm')
def logisticContact_precision_esm(tar_contactMap: np.ndarray,
                                  pred_logits: np.ndarray,
                                  normalize: bool = False,
                                  **kwargs) -> Union[np.ndarray, Sequence[Sequence[float]]]:
  top_cut = kwargs['top_cut'] # 1,2,5
  cal_range = kwargs['cal_range'] # all, short, medium, long
  valid_mask = kwargs['valid_mask']
  seq_length = kwargs['seq_length']
  # contactMap: [bs,L_max,L_max]
  # valid_mask: [bs,L_max]
    
  valid_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
  seqpos = np.arange(valid_mask.shape[1])
  x_ind, y_ind = np.meshgrid(seqpos, seqpos)
  if cal_range == 'all':
    #valid_mask_mat &= ((y_ind - x_ind) >= 6)
    valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 6)
  elif cal_range == 'short':
    #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12)
    valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 6, (y_ind - x_ind) < 12))
  elif cal_range == 'medium':
    #valid_mask_mat &= np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24)
    valid_mask_mat = np.logical_and(valid_mask_mat,np.logical_and((y_ind - x_ind) >= 12, (y_ind - x_ind) < 24))
  elif cal_range == 'long':
    #valid_mask_mat &= ((y_ind - x_ind) >= 24)
    valid_mask_mat = np.logical_and(valid_mask_mat,(y_ind - x_ind) >= 24)
  else:
    raise Exception("Unexpected range for precision calculation: {}".format(cal_range))
  correct =0.
  total = 0.
  indiv_prec_list = [] #[bs,]
  # loop through bs
  for length, pred_logit, tar_contM, mask in zip(seq_length, pred_logits, tar_contactMap, valid_mask_mat):
    '''
    masked_atten = (atten * mask).reshape(atten.shape[0],-1) # size [n_head,l_max*l_max]
    most_likely_idx = np.argpartition(-masked_atten,kth=length // top_cut,axis=-1)[:,:(length // top_cut) + 1]
    '''
    val_count = np.sum(mask)
    if val_count == 0:
      #total += length//top_cut
      continue
    else:
      val_predLogit = pred_logit[mask] # size [val_count,]
      val_tarContM = tar_contM[mask] # size [val_count,]
      top_len = min(length//top_cut,val_count)
      most_likely_idx = np.argpartition(-val_predLogit,kth=top_len-1)[:top_len] # size [top_len,]
      selected = np.take_along_axis(val_tarContM, most_likely_idx,axis=0) #size [top_len,]
      correct += np.sum(selected) # scaler
      total += length//top_cut
      indiv_prec_list.append(np.sum(selected) / (length//top_cut))
  
  if normalize:
    return correct / (total + 1e-5)
  else:
    return correct, total, np.array(indiv_prec_list)

# fitness supervise prediction for CAGI
@registry.register_metric('fitness_supervise_CAGI')
def SVFitess_cagi(value: Sequence[Sequence],
                    data_dir: str,
                    split: str,
                    from_pretrained: str,
                    pretrained_epoch: int,
                    set_nm: str) -> float:
  pred_fit_list = value['pred_fit'] # [bs,]
  mut_str_list = value['mutants'] # [bs,]
  mdl_path = re.split(r'/',from_pretrained)[-1]
  pred_fit_pair = {}
  for bs_i in range(len(pred_fit_list)):
    pred_fit_pair[mut_str_list[bs_i]] = pred_fit_list[bs_i][0]
  if pretrained_epoch is not None:
    epoch = '_{}'.format(pretrained_epoch)
  else:
    epoch = '_best'
  ## mkdir folder to store data
  if not os.path.isdir(f'{data_dir}/{set_nm}/result/{mdl_path}{epoch}'):
    os.makedirs(f'{data_dir}/{set_nm}/result/{mdl_path}{epoch}')

  write_fl = open(f'{data_dir}/{set_nm}/result/{mdl_path}{epoch}/{split}.tsv','w')
  
  ## loop mutations in template
  with open(f'{data_dir}/{set_nm}/template/{split}_template.tsv')as fl:
    next(fl)  # skip head line
    for line in fl:
      line = line.replace('\n','')
      mutStr = re.split(r'\t',line)[0]
      ## wt_aa, mut_aa, idx_aa
      mut_list = re.findall(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)',mutStr)
      if len(mut_list) == 1: # single-site mut
        aaRef, aaIdx, aaMut = mut_list[0]
        aaR = aaCodes[aaRef] if len(aaRef) == 3 else aaRef.upper()
        aaI = int(aaIdx)
        if aaMut.lower() == 'del': # deletion
          aaM = '<pad>'
        elif aaMut == '=': # synomonous
          continue
        elif aaMut.lower() == 'ter': #stopcodon
          continue
        else: # missense
          aaM = aaCodes[aaMut] if len(aaMut) == 3 else aaMut.upper()
        try:
          pred_fit = pred_fit_pair[f'{aaR}{aaIdx}{aaM}']
          write_fl.write(f'{mutStr}\t{np.exp(pred_fit)}\n')
        except:
          Exception(f'invalid variant: {aaR}{aaIdx}{aaM}')

  return None


#fitness_unsupervise
@registry.register_metric('fitness_unsupervise_CAGI')
def unSVFitess_cagi(value: Sequence[Sequence],
                     data_dir: str,
                     split: str,
                     from_pretrained: str,
                     pretrained_epoch: int) -> float:
  mask_pred_logits = value[0] #list,[bs,n_mutPos,n_tokens]
  mask_labels = value[1] #list, [bs,n_mutPos]
  mask_mutPoses = value[2] #list, [bs,n_mutPos]
  pos_logits_dict = {}
  for i in range(len(mask_pred_logits)):
    pred_logit = mask_pred_logits[i] # ndarray,[n_mutPos,n_tokens]
    label = mask_labels[i] # ndarray, [n_mutPos,]
    mut_pos = np.sort(mask_mutPoses[i]) # list, [n_mutPos,]
    mut_pos_str = [str(p) for p in mut_pos]
    pos_logits_dict['_'.join(mut_pos_str)] = [pred_logit,label]
  
  mdl_path = re.split(r'/',from_pretrained)[-1]
  setNm = re.split(r'/', data_dir)[-1]
  if pretrained_epoch is not None:
    epoch = '_{}'.format(pretrained_epoch)
  else:
    epoch = '_best'
  ## mkdir folder to store data
  if not os.path.isdir('{}/result/{}{}'.format(data_dir,mdl_path,epoch)):
    os.makedirs('{}/result/{}{}'.format(data_dir,mdl_path,epoch))

  write_fl = open('{}/result/{}{}/{}.tsv'.format(data_dir,mdl_path,epoch,split),'w')
  
  # elif len(mut_list) > 1: # multi-site mut ##TODO
  #   Exception('Under development!')
  #   aaRef, aaIdx, aaMut = mut_list[0]
  #   aaR = aaCodes[aaRef] if len(aaRef) == 3 else aaRef.upper()
  #   aaM = aaCodes[aaMut] if len(aaMut) == 3 else aaMut.upper()
  #   aaI = int(aaIdx)

  #   mut_pos = np.sort([aaI,222])
  #   mut_pos_str = [str(p) for p in mut_pos]
  #   idx_key = '_'.join(mut_pos_str)
  #   pred_logit, label = pos_logits_dict[idx_key]
  #   pred_softmax = softmax(pred_logit,axis=-1)
  #   try:
  #     assert aaR == label[0]
  #   except:
  #     Exception('aaR,label[0]'.format(aaR,label[0]))
  #   # compare aaI and 222
  #   if aaI < 222:
  #     mut_prob = pred_softmax[0][PFAM_VOCAB[aaM]]*pred_softmax[1][PFAM_VOCAB['V']]
  #     wt_prob = pred_softmax[0][PFAM_VOCAB[aaR]]*pred_softmax[1][PFAM_VOCAB['A']]
  #   else:
  #     mut_prob = pred_softmax[0][PFAM_VOCAB['V']]*pred_softmax[1][PFAM_VOCAB[aaM]]
  #     wt_prob = pred_softmax[0][PFAM_VOCAB['A']]*pred_softmax[1][PFAM_VOCAB[aaR]]
  #   ratio = mut_prob / wt_prob
  #   write_fl.write('{}\t{}\n'.format(mutStr,ratio))

  ## loop mutations in template
  with open(f'{data_dir}/template/{split}_template.tsv')as fl:
    next(fl)  # skip head line
    for line in fl:
      line = line.replace('\n','')
      mutStr = re.split(r'\t',line)[0]
      ## wt_aa, mut_aa, idx_aa
      mut_list = re.findall(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)',mutStr)
      if len(mut_list) == 1: # single-site mut
        aaRef, aaIdx, aaMut = mut_list[0]
        aaR = aaCodes[aaRef] if len(aaRef) == 3 else aaRef.upper()
        aaI = int(aaIdx)
        if aaMut.lower() == 'del': # deletion
          aaM = '<pad>'
        elif aaMut == '=': # synomonous
          continue
        elif aaMut.lower() == 'ter': #stopcodon
          continue
        else: # missense
          aaM = aaCodes[aaMut] if len(aaMut) == 3 else aaMut.upper()
        pred_logit, label = pos_logits_dict['{}'.format(aaI)]
        pred_softmax = softmax(pred_logit,axis=-1)
        try:
          assert aaR == label[0]
        except:
          Exception('aaR,label[0]'.format(aaR,label[0]))
        
        mut_prob = pred_softmax[0][PFAM_VOCAB[aaM]]
        wt_prob = pred_softmax[0][PFAM_VOCAB[aaR]]
        ratio = mut_prob / wt_prob
        write_fl.write('{}\t{}\n'.format(mutStr,ratio))
        
  return None

@registry.register_metric('fitness_unsupervise_scanning')
def unSVFitness_muta(value: Sequence[Sequence],
                     data_dir: str,
                     file_name: str,
                     **kwargs) -> float:
  pretrained_epoch = kwargs.get('pretrained_epoch')
  pretrain_setId = kwargs.get('pretrain_setId')
  if pretrained_epoch is not None:
    pre_epoch_nm = '{}'.format(pretrained_epoch)
  else:
    pre_epoch_nm = 'best'

  ## all amino acid id list
  ##("K", 12),("R", 19),("H", 10),("E", 7),("D", 6),("N", 15),("Q", 18),("T", 21),("S", 20),("C", 5),("G", 9),("A", 4),("V", 23),("L", 13),("I", 11),("M", 14),("P", 17),("Y", 25),("F", 8),("W", 24)
  allAA_ids = [12,19,10,7,6,15,18,21,20,5,9,4,23,13,11,14,17,25,8,24]
  allAA_char = ["K","R","H","E","D","N","Q","T","S","C","G","A","V","L","I","M","P","Y","F","W"]
  
  mask_pred_logits = value[0] #list,[bs,1,n_token]
  label_wt = value[1] #list, [bs,1]
  pos_idx = value[2] #list, [bs,1]
  fitness_out = []
  # loop position
  for bs_i in range(len(label_wt)):
    fitness_onePos = []
    abso_pos = pos_idx[bs_i][0] # scalar
    fitness_onePos.append(abso_pos)

    pred_logit = mask_pred_logits[bs_i] #[1,n_token]
    pred_softmax = softmax(pred_logit,axis=-1) #[1,n_token]
    wt_resId = label_wt[bs_i] #[1,]
    # loop number of muts (1 here)
    for p in range(len(pred_softmax)):
      wtAA_id = wt_resId[p]
      prob_wt = pred_softmax[p][wtAA_id].item()
      # loop all possible muts (include wt AA, order defined in allAA_ids)
      for mut_id in allAA_ids:
        prob_mut = pred_softmax[p][mut_id].item()
        fitness_onePos.append(np.log(prob_mut/prob_wt))
    fitness_out.append(fitness_onePos)
    ## process mutation name
  np.savetxt(f'{data_dir}/{file_name}_predFit_{pretrain_setId}_{pre_epoch_nm}.csv',fitness_out,fmt='%s',delimiter=',')

@registry.register_metric('fitness_unsupervise_mutagenesis')
def unSVFitness_muta(value: Sequence[Sequence],
                     mutaSet_nm: str,
                     eval_save_dir: str,
                     model_dir: str,
                     task: str,
                     **kwargs) -> float:
  mask_pred_logits = value[0] #list,[bs,n_mut,n_token]
  label_wt = value[1] #list, [bs,n_mut]
  label_mut = value[2]
  fitness_gt = value[3] #list, [bs,]
  mutantNm_list = value[4] #list, [bs,]
  fitness_unSV_list,fitness_unSV_all_list = [],[] #[bs,]
  fitness_gt_list = []
  mutantStr_list = [] #[bs, ]
  pretrained_epoch = kwargs.get('pretrained_epoch')
  save_raw_score = kwargs.get('save_raw_score')

  if pretrained_epoch is not None:
    epoch_dir = '{}'.format(pretrained_epoch)
  else:
    epoch_dir = 'best'
  
  for bs_i in range(len(fitness_gt)):
    pred_logit = mask_pred_logits[bs_i] #[n_mut,n_token]
    pred_softmax = softmax(pred_logit,axis=-1)
    wt_resId = label_wt[bs_i][0] #[n_mut,]
    mut_resId = label_mut[bs_i][0]
    prob_wt, prob_mut = 1.0,1.0
    for p in range(len(pred_softmax)):
      prob_wt *= pred_softmax[p][wt_resId[p]]
      prob_mut *= pred_softmax[p][mut_resId[p]]
    fitness_unSV_all_list.append(np.log((prob_mut / prob_wt).item()))
    if math.isfinite(fitness_gt[bs_i]):
      fitness_unSV_list.append(np.log((prob_mut / prob_wt).item()))
      fitness_gt_list.append(fitness_gt[bs_i])
    ## process mutation name
    mutant_list = mutantNm_list[bs_i]
    if isinstance(mutant_list, str):
      mutantStr_list.append(mutant_list)
    elif isinstance(mutant_list, list):
      mutant_list.sort(key=lambda x:int(x[1:-1]))
      mutantStr_list.append(':'.join(mutant_list))
  
  ## save metrics
  eval_path = f'{eval_save_dir}/{task}/predictions/{model_dir}'
  Path(eval_path).mkdir(parents=True, exist_ok=True)
  if save_raw_score: # save prediected ratio score for each mutation
    raw_scores = np.stack((mutantStr_list, fitness_unSV_all_list), axis=-1)
    np.savetxt(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_rawScores.csv',raw_scores,fmt='%s',delimiter=',')

  ## calculate mse and spearman, pearson
  spearR, spearPvalue = scipy.stats.spearmanr(fitness_gt_list, fitness_unSV_list)
  pearR, pearPvalue = scipy.stats.pearsonr(fitness_gt_list, fitness_unSV_list)
  out_json = {'mut_num':len(fitness_gt),
              'spearmanR': spearR,
              'spearmanPvalue': spearPvalue,
              'pearsonR': pearR,
              'pearsonPvalue': pearPvalue}
  ## save metrics
  with open(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_metrics.json','w') as oj:
    json.dump(out_json,oj)

  report_str = 'eval_report*> '
  for k,v in out_json.items():
    report_str += f'{k}: {v};'
  print(report_str,flush=True)

  return None

def get_class_weight(target_inputs: torch.Tensor, class_num: int):
  """generate class weights for crossEntropy

  This function only accept torch tensor for now  
  """
  total_num = target_inputs.reshape(-1).ge(0).sum().to('cpu').to(torch.float)
  bin_count = torch.bincount(torch.masked_select(target_inputs,target_inputs.ge(0))).to('cpu').to(torch.float)
  bin_miss = class_num - bin_count.size(0)
  if bin_miss > 0:
      bin_count = torch.cat((bin_count,torch.zeros(bin_miss).to('cpu')))
  bin_count = torch.where(bin_count == 0.,total_num,bin_count)
  class_weights = torch.reciprocal(bin_count)
  return class_weights

@registry.register_metric('seqModel_seq_struct_eval')
def seqModel_seq_struct_eval(value: dict,
                            set_nm: str=None,
                            eval_save_dir: str='./eval_results',
                            model_dir: str=None,
                            task: str=None,
                            **kwargs):
  """ Evaluation pipeline for sequence and structure tasks over multitask models
  * AA ppl; SS ppl, RSA ppl, Dist ppl

  Args:
    value: (dict)
    set_nm: e.g. AMIE_PSEAE/I2DK40/6-271/AF-I2DK40-F1
  """
  pretrained_epoch = kwargs.get('pretrained_epoch')
  label_type = kwargs.get('label_type')
  batch_ce_values = kwargs.get('batch_ce_values')
  split = kwargs.get('split')
  accumulator = kwargs.get('accumulator')

  set_nm = re.split('/',set_nm)[0]
  if pretrained_epoch is not None:
    epoch_dir = pretrained_epoch
  else:
    epoch_dir = 'best'
  ## report json
  out_json = {}

  if label_type == 'aa':
    ## aa_ppl
    aa_logits_tensor = torch.from_numpy(np.concatenate(value['pred_logits'],axis=0))
    aa_targets_tensor = torch.from_numpy(np.concatenate(value['targets_label'],axis=0))
    aa_class_weight = get_class_weight(aa_targets_tensor,aa_logits_tensor.size(1)).to(aa_logits_tensor)
    aa_loss_fct = nn.CrossEntropyLoss(weight=aa_class_weight,ignore_index=-1)
    ce_loss = aa_loss_fct(aa_logits_tensor.view(-1, aa_logits_tensor.size(1)), aa_targets_tensor.view(-1))
    out_json[f'{label_type}_ppl'] = torch.exp(ce_loss).item()

    ## aa_ppl renormolized
    aa_logits_tensor = aa_logits_tensor[:,PFAM_VOCAB_20AA_IDX]
    aa_targets_tensor.apply_(lambda x: PFAM_VOCAB_20AA_IDX_MAP[x])
    aa_class_weight = get_class_weight(aa_targets_tensor,aa_logits_tensor.size(1)).to(aa_logits_tensor)
    aa_loss_fct = nn.CrossEntropyLoss(weight=aa_class_weight,ignore_index=-1)
    ce_loss = aa_loss_fct(aa_logits_tensor.view(-1, aa_logits_tensor.size(1)), aa_targets_tensor.view(-1))
    out_json[f'{label_type}_ppl_20renor'] = torch.exp(ce_loss).item()
  elif label_type == 'ss':
    ## ss_ppl
    ss_logits_tensor = torch.from_numpy(np.concatenate(value['pred_logits'],axis=0))
    ss_targets_tensor = torch.from_numpy(np.concatenate(value['targets_label'],axis=0))
    ss_class_weight = get_class_weight(ss_targets_tensor,ss_logits_tensor.size(1)).to(ss_logits_tensor)
    ss_loss_fct = nn.CrossEntropyLoss(weight=ss_class_weight,ignore_index=-1)
    ce_loss = ss_loss_fct(ss_logits_tensor.view(-1, ss_logits_tensor.size(1)), ss_targets_tensor.view(-1))
    out_json[f'{label_type}_ppl'] = torch.exp(ce_loss).item()
  elif label_type == 'rsa':
    ## rsa_ppl
    rsa_logits_tensor = torch.from_numpy(np.concatenate(value['pred_logits'],axis=0))
    rsa_targets_tensor = torch.from_numpy(np.concatenate(value['targets_label'],axis=0))
    rsa_class_weight = get_class_weight(rsa_targets_tensor,rsa_logits_tensor.size(1)).to(rsa_logits_tensor)
    rsa_loss_fct = nn.CrossEntropyLoss(weight=rsa_class_weight,ignore_index=-1)
    ce_loss = rsa_loss_fct(rsa_logits_tensor.view(-1, rsa_logits_tensor.size(1)), rsa_targets_tensor.view(-1))
    out_json[f'{label_type}_ppl'] = torch.exp(ce_loss).item()
  elif label_type == 'distMap':
    ## dist_ppl ##
    # batch (moving) average #
    if value is not None:
      dist_logits_tensor = torch.from_numpy(np.concatenate([ele.reshape(-1,ele.shape[2]) for ele in value['pred_logits']], axis=0))
      dist_targets_tensor = torch.from_numpy(np.concatenate([ele.reshape(-1) for ele in value['targets_label']], axis=0))
      dist_class_weight = get_class_weight(dist_targets_tensor,dist_logits_tensor.size(1)).to(dist_logits_tensor)
      dist_loss_fct = nn.CrossEntropyLoss(weight=dist_class_weight,ignore_index=-1,reduction='mean')
      batch_ce = dist_loss_fct(dist_logits_tensor.view(-1, dist_logits_tensor.size(1)), dist_targets_tensor.view(-1))
      accumulator.update(batch_ce, {'ppl': torch.exp(batch_ce)})
    else:
      ## report
      out_json[f'{label_type}_ce_ave_exp'] = np.exp(accumulator.final_loss())
      out_json[f'{label_type}_ce_mvave_exp'] = np.exp(accumulator.loss())
      out_json[f'{label_type}_ppl_ave'] = accumulator.final_metrics()['ppl']
      out_json[f'{label_type}_ppl_mvave'] = accumulator.metrics()['ppl']
    
  if len(out_json) > 0 :
    ## save metrics
    eval_path = f'{eval_save_dir}/{task}/predictions/{model_dir}'
    Path(eval_path).mkdir(parents=True, exist_ok=True)
    with open(f'{eval_path}/{split}_{epoch_dir}_metrics.json','w') as oj:
      json.dump(out_json,oj)

    report_str = 'eval_report*> '
    for k,v in out_json.items():
      report_str += f'{k}: {v};'
    print(report_str,flush=True)

  return None

@registry.register_metric('multitask_seq_struct_eval')
def multitask_seq_struct_eval(value: dict,
                              set_nm: str,
                              eval_save_dir: str,
                              model_dir: str,
                              task: str,
                              **kwargs):
  """ Evaluation pipeline for sequence and structure tasks over multitask models
  * AA ppl; SS ppl, RSA ppl, Dist ppl

  Args:
    value: (dict) should contain 'aa_logits', 'ss_logits', 'rsa_logits', 'dist_logits'
  """
  pretrained_epoch = kwargs.get('pretrained_epoch')
  accumulator = kwargs.get('accumulator')
  batch_accumu = kwargs.get('batch_accumu',False)
  class_channel_last = kwargs.get('class_channel_last',False)

  if pretrained_epoch is not None:
    epoch_dir = pretrained_epoch
  else:
    epoch_dir = 'best'
  
  ## caution: Conv output size: [N,C_out,L_out], 2nd is # classes
  ##          MLP output size: [N,L_out,C_out], last is # classes
  
  out_json = {}
  
  if batch_accumu:
    if not class_channel_last: #[C,L,L]/[C,L] -> [L,L,C]/[L,C]
      value['dist_logits'] = [ele.transpose(1,2,0) for ele in value['dist_logits']]
    
    # batch (moving) average for dist_ppl
    dist_logits_tensor = torch.from_numpy(np.concatenate([ele.reshape(-1,ele.shape[2]) for ele in value['dist_logits']], axis=0))
    dist_targets_tensor = torch.from_numpy(np.concatenate([ele.reshape(-1) for ele in value['targets_dist']], axis=0))
    dist_class_weight = get_class_weight(dist_targets_tensor,dist_logits_tensor.size(1)).to(dist_logits_tensor)
    dist_loss_fct = nn.CrossEntropyLoss(weight=dist_class_weight,ignore_index=-1,reduction='mean')
    batch_ce = dist_loss_fct(dist_logits_tensor.view(-1, dist_logits_tensor.size(1)), dist_targets_tensor.view(-1))
    accumulator.update(batch_ce, {'ppl': torch.exp(batch_ce)})
  else:
    if not class_channel_last: #[N,C,L] -> [N,L,C]
      value['aa_logits'] = [ele.transpose(1,0) for ele in value['aa_logits']]
      value['ss_logits'] = [ele.transpose(1,0) for ele in value['ss_logits']]
      value['rsa_logits'] = [ele.transpose(1,0) for ele in value['rsa_logits']]

    ## aa_ppl
    aa_logits_tensor = torch.from_numpy(np.asarray(value['aa_logits']))
    aa_targets_tensor = torch.from_numpy(np.asarray(value['targets_aa']))
    aa_class_weight = get_class_weight(aa_targets_tensor,aa_logits_tensor.size(2)).to(aa_logits_tensor)
    aa_loss_fct = nn.CrossEntropyLoss(weight=aa_class_weight,ignore_index=-1)
    aa_ce_loss = aa_loss_fct(aa_logits_tensor.view(-1, aa_logits_tensor.size(2)), aa_targets_tensor.view(-1))
    ## ss_ppl
    ss_logits_tensor = torch.from_numpy(np.asarray(value['ss_logits']))
    ss_targets_tensor = torch.from_numpy(np.asarray(value['targets_ss']))
    ss_class_weight = get_class_weight(ss_targets_tensor,ss_logits_tensor.size(2)).to(ss_logits_tensor)
    ss_loss_fct = nn.CrossEntropyLoss(weight=ss_class_weight,ignore_index=-1)
    ss_ce_loss = ss_loss_fct(ss_logits_tensor.view(-1, ss_logits_tensor.size(2)), ss_targets_tensor.view(-1))
    ## rsa_ppl
    rsa_logits_tensor = torch.from_numpy(np.asarray(value['rsa_logits']))
    rsa_targets_tensor = torch.from_numpy(np.asarray(value['targets_rsa']))
    rsa_class_weight = get_class_weight(rsa_targets_tensor,rsa_logits_tensor.size(2)).to(rsa_logits_tensor)
    rsa_loss_fct = nn.CrossEntropyLoss(weight=rsa_class_weight,ignore_index=-1)
    rsa_ce_loss = rsa_loss_fct(rsa_logits_tensor.view(-1, rsa_logits_tensor.size(2)), rsa_targets_tensor.view(-1))
    ## dist_ppl
    # dist_logits_tensor = torch.from_numpy(np.asarray(value['dist_logits']))
    # dist_targets_tensor = torch.from_numpy(np.asarray(value['targets_dist']))
    # dist_class_weight = get_class_weight(dist_targets_tensor,dist_logits_tensor.size(1)).to(dist_logits_tensor)
    # dist_loss_fct = nn.CrossEntropyLoss(weight=dist_class_weight,ignore_index=-1)
    # dist_ce_loss = dist_loss_fct(dist_logits_tensor.view(-1, dist_logits_tensor.size(1)), dist_targets_tensor.view(-1))

    ## report
    out_json['distMap_ce_ave_exp'] = np.exp(accumulator.final_loss())
    out_json['distMap_ce_mvave_exp'] = np.exp(accumulator.loss())
    out_json['distMap_ppl_ave'] = accumulator.final_metrics()['ppl']
    out_json['distMap_ppl_mvave'] = accumulator.metrics()['ppl']
    out_json.update({
    'aa_ppl': torch.exp(aa_ce_loss).item(),
    'ss_ppl': torch.exp(ss_ce_loss).item(),
    'rsa_ppl': torch.exp(rsa_ce_loss).item(),
    })

    ## save metrics
    eval_path = f'{eval_save_dir}/{task}/predictions/{model_dir}'
    Path(eval_path).mkdir(parents=True, exist_ok=True)
    with open(f'{eval_path}/{set_nm}_{epoch_dir}_metrics.json','w') as oj:
      json.dump(out_json,oj)

    report_str = 'eval_report*> '
    for k,v in out_json.items():
      report_str += f'{k}: {v};'
    print(report_str,flush=True)

  return None

@registry.register_metric('multitask_unsupervise_mutagenesis')
def unSVFitness_muta_mutlitask(value: dict,
                              mutaSet_nm: str,
                              eval_save_dir: str,
                              model_dir: str,
                              task: str,
                              **kwargs) -> float:
  """Evaluation pipeline for multi-task models. The following metrics will be calculated:
  * Spearman's R, evaluated over aa loss
  * Pearson's R, evaluated over aa loss
  """
  fitness_unSV_list = [] #[bs,]
  #fitness_unSV_list_20renor = [] #[bs,]
  fitness_unSV_list_save = [] #[bs,]
  #fitness_unSV_list_save_20renor = [] #[bs,]
  fitness_gt_list = [] #[bs,]
  mutantStr_list = [] #[bs,]
  save_raw_score = kwargs.get('save_raw_score',True)
  pretrained_epoch = kwargs.get('pretrained_epoch')
  cls_eval = kwargs.get('cls_eval',False)
  class_channel_last = kwargs.get('class_channel_last',False)

  if pretrained_epoch is not None:
    epoch_dir = pretrained_epoch
  else:
    epoch_dir = 'best'
  
  for bs_i in range(len(value['fitness_gt'])):
    aa_logits = value['aa_logits'][bs_i]
    wt_aa_ids = value['wt_aa_ids'][bs_i]
    mut_aa_ids = value['mut_aa_ids'][bs_i]
    aa_masks = wt_aa_ids != -1
    if class_channel_last:
      aa_mask_logits = aa_logits[aa_masks,:] #[n_mask,aa_tokens]
      aa_mask_logits = np.transpose(aa_mask_logits)
    else:
      aa_mask_logits = aa_logits[:,aa_masks] #[aa_tokens,n_mask]

    ## un-normalized to 20 AA tokens
    wt_mask_ids = wt_aa_ids[aa_masks]
    mut_mask_ids = mut_aa_ids[aa_masks]
    aa_mask_softmax = softmax(aa_mask_logits,axis=0)
    prob_wt, prob_mut = 1.0,1.0
    for p in range(aa_mask_softmax.shape[1]):
      prob_wt *= aa_mask_softmax[:,p][wt_mask_ids[p]]
      prob_mut *= aa_mask_softmax[:,p][mut_mask_ids[p]]
    if math.isfinite(value['fitness_gt'][bs_i]):
      fitness_unSV_list.append(np.log((prob_mut / prob_wt).item()))
      fitness_gt_list.append(value['fitness_gt'][bs_i])
    ## save mutation name and ratio score
    mutant_list = value['mutation_list'][bs_i]
    mutantStr_list.append(':'.join(mutant_list))
    fitness_unSV_list_save.append(np.log((prob_mut / prob_wt).item()))

    ## normalized to 20 AA tokens (after taking ratio, 29-normalized is the same as 20-normalized)
    # aa_mask_20_logits = aa_mask_logits[PFAM_VOCAB_20AA_IDX,:]
    # wt_mask_ids = wt_aa_ids[aa_masks]
    # mut_mask_ids = mut_aa_ids[aa_masks]
    # aa_mask_softmax = softmax(aa_mask_20_logits,axis=0)
    # prob_wt, prob_mut = 1.0,1.0
    # for p in range(aa_mask_softmax.shape[1]):
    #   prob_wt *= aa_mask_softmax[:,p][PFAM_VOCAB_20AA_IDX_MAP[wt_mask_ids[p]]]
    #   prob_mut *= aa_mask_softmax[:,p][PFAM_VOCAB_20AA_IDX_MAP[mut_mask_ids[p]]]
    # if math.isfinite(value['fitness_gt'][bs_i]):
    #   fitness_unSV_list_20renor.append(np.log((prob_mut / prob_wt).item()))
    # ## process mutation name
    # fitness_unSV_list_save_20renor.append(np.log((prob_mut / prob_wt).item()))
  
  eval_path = f'{eval_save_dir}/{task}/predictions/{model_dir}'
  Path(eval_path).mkdir(parents=True, exist_ok=True)
  if save_raw_score: # save prediected ratio score for each mutation
    raw_scores = np.stack((mutantStr_list, fitness_unSV_list_save), axis=-1)
    np.savetxt(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_wtstruct_rawScores.csv',raw_scores,fmt='%s',delimiter=',')

  ## binary accuracy
  cls_eval_dict = None
  if cls_eval:
    cls_pred = np.where(np.array(fitness_unSV_list) >=0, 1, 0) # similar or better-than-wt: >=1 (class 1), worse-than-wt: <1 (class 0)
    cls_true = np.where(np.array(fitness_gt_list) >=0, 1, 0) # positive label means better than wt
    cls_eval_dict = classification_report(cls_true, cls_pred, output_dict=True)
 
  ## calculate mse and spearman, pearson
  # test finite
  #print(f"fitness_gt infinite: {np.sum(~np.isfinite(value['fitness_gt']))}",flush=True)
  #print(f"fitness_pred infinite: {np.sum(~np.isfinite(fitness_unSV_list))}",flush=True)
  spearR, spearPvalue = scipy.stats.spearmanr(fitness_gt_list, fitness_unSV_list)
  pearR, pearPvalue = scipy.stats.pearsonr(fitness_gt_list, fitness_unSV_list)
  #spearR_20renor, spearPvalue_20renor = scipy.stats.spearmanr(fitness_gt_list, fitness_unSV_list_20renor)
  #pearR_20renor, pearPvalue_20renor = scipy.stats.pearsonr(fitness_gt_list, fitness_unSV_list_20renor)
  out_json = {'mut_num':len(value['fitness_gt']),
              'spearmanR': spearR,
              'spearmanPvalue': spearPvalue,
              'pearsonR': pearR,
              'pearsonPvalue': pearPvalue}

  ## append classification metrics
  if cls_eval:
    for cls_key, cls_value in cls_eval_dict.items():
      out_json[cls_key] = cls_value

  ## save metrics
  with open(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_wtstruct_metrics.json','w') as oj:
    json.dump(out_json,oj)

  report_str = 'eval_report*> '
  for k,v in out_json.items():
    report_str += f'{k}: {v};'
  print(report_str,flush=True)

  return None

@registry.register_metric('multitask_unsupervise_mutagenesis_structure')
def unSVFitness_muta_mutlitask_structure_fast(value: dict,
                                              mutaSet_nm: str,
                                              eval_save_dir: str,
                                              model_dir: str,
                                              task: str,
                                              **kwargs) -> float:
  """Multitask model's evaluation pipeline for structure properties: SS, RSA, DistMap
  faster version
  Args: 
    value: data dict with keys: 'aa_seq_mask','aa_logits','aa_labels','ss3_logits','ss3_labels','rsa2_logits','rsa2_labels','distMap_logits','distMap_labels','mutants','mut_relative_idxs','fitness_score'
  """
  ## generate log ratio for structure properties (log(p(s|mut)/p(s|wt)))
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  pretrained_epoch = kwargs.get('pretrained_epoch')
  cls_eval = kwargs.get('cls_eval',False)
  class_channel_last = kwargs.get('class_channel_last',False)
  model_config = kwargs.get('model_config',None)
  assert model_config is not None

  if pretrained_epoch is not None:
    epoch_dir = pretrained_epoch
  else:
    epoch_dir = 'best'

  # load AA log-ratio
  fit_aa_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_dir}/{mutaSet_nm}_{epoch_dir}_wtstruct_rawScores.csv',delimiter=',',names=['var_name','aa_fit'])

  structure_ratio_wt_list = []
  structure_ratio_mut_list = []
  max_pos_num, max_env_pos_num, max_distMap_pos_num = 0,0,0
  ss3_flag_list, rsa2_flag_list, distMap_flag_list = [],[],[]
  for bs_i in range(len(value['mutants'])):
    aa_seq_mask = value['aa_seq_mask'][bs_i]
    ss3_logits = value['ss3_logits'][bs_i]
    ss3_labels = value['ss3_labels'][bs_i] #[L_seq,]
    rsa2_logits = value['rsa2_logits'][bs_i]
    rsa2_labels = value['rsa2_labels'][bs_i] #[L_seq,]
    distMap_logits = value['distMap_logits'][bs_i]
    distMap_labels = value['distMap_labels'][bs_i] #[L_seq,L_seq]
    mutants = value['mutants'][bs_i] #e.g. ['A16D:R20F','D16','F20']
    mut_relative_idxs = value['mut_relative_idxs'][bs_i]
    fit_true = value['fitness_score'][bs_i]
    
    ## class channel is the LAST dimension
    assert class_channel_last == True
    aa_seq_mask_bool = aa_seq_mask == 1
    ss3_prob_mut = softmax(ss3_logits[aa_seq_mask_bool],axis=-1) #[L_seq,ss_class]
    rsa2_prob_mut = softmax(rsa2_logits[aa_seq_mask_bool],axis=-1) #[L_seq,rsa_class]
    distMap_prob_mut = softmax(distMap_logits[np.ix_(aa_seq_mask_bool,aa_seq_mask_bool)],axis=-1) #[L_seq,L_seq,dist_class]

    ## extract target log_prob
    ss3_flag, rsa2_flag, distMap_flag = False,False,False
    ss3_log_sum, rsa2_log_sum, ss3_env_log_sum, rsa2_env_log_sum, distMap_log_sum = 0., 0., 0., 0., 0.
    ss3_prob_list, rsa2_prob_list, ss3_env_prob_list, rsa2_env_prob_list, distMap_prob_list = [], [], [], [], []
    for mut_i in mut_relative_idxs:
      if int(ss3_labels[mut_i]) != -1:
        prob_val = ss3_prob_mut[mut_i][int(ss3_labels[mut_i])]
        ss3_log_sum += np.log(prob_val)
        ss3_env_log_sum += np.log(prob_val)
        ss3_prob_list.append(prob_val)
        ss3_env_prob_list.append(prob_val)
        ss3_flag = True
      else:
        ss3_prob_list.append(1.0)
        ss3_env_prob_list.append(1.0)
      if int(rsa2_labels[mut_i]) != -1:
        prob_val = rsa2_prob_mut[mut_i][int(rsa2_labels[mut_i])]
        rsa2_log_sum += np.log(prob_val)
        rsa2_env_log_sum += np.log(prob_val)
        rsa2_prob_list.append(prob_val)
        rsa2_env_prob_list.append(prob_val)
        rsa2_flag = True
      else:
        rsa2_prob_list.append(1.0)
        rsa2_env_prob_list.append(1.0)
      for neib_i in np.argwhere(distMap_labels[mut_i] == 0).reshape(-1):
        if abs(mut_i - neib_i) >= 6:
          if distMap_labels[mut_i][neib_i] != -1:
            prob_val = distMap_prob_mut[mut_i,neib_i,int(distMap_labels[mut_i][neib_i])]
            distMap_log_sum += np.log(prob_val)
            distMap_prob_list.append(prob_val)
            distMap_flag = True
          else:
            distMap_prob_list.append(1.0)
        if mut_i != neib_i:
          if int(ss3_labels[neib_i]) != -1 and int(rsa2_labels[neib_i]) != -1:
            ss3_env_log_sum += np.log(ss3_prob_mut[neib_i][int(ss3_labels[neib_i])])
            rsa2_env_log_sum += np.log(rsa2_prob_mut[neib_i][int(rsa2_labels[neib_i])])
            ss3_env_prob_list.append(ss3_prob_mut[neib_i][int(ss3_labels[neib_i])])
            rsa2_env_prob_list.append(rsa2_prob_mut[neib_i][int(rsa2_labels[neib_i])])
          else:
            ss3_env_prob_list.append(1.0)
            rsa2_env_prob_list.append(1.0)
    ss3_flag_list.append(ss3_flag)
    rsa2_flag_list.append(rsa2_flag)
    distMap_flag_list.append(distMap_flag)

    if len(ss3_prob_list) > max_pos_num:
      max_pos_num = len(ss3_prob_list)
    if len(ss3_env_prob_list) > max_env_pos_num:
      max_env_pos_num = len(ss3_env_prob_list)
    if len(distMap_prob_list) > max_distMap_pos_num:
      max_distMap_pos_num = len(distMap_prob_list)

    ## handle NAN and unequal length
    if not ss3_flag:
      ss3_log_sum = np.nan
      ss3_env_log_sum = np.nan
      ss3_prob_list = [np.nan]
      ss3_env_prob_list = [np.nan]
    if not rsa2_flag:
      rsa2_log_sum = np.nan
      rsa2_env_log_sum = np.nan
      rsa2_prob_list = [np.nan]
      rsa2_env_prob_list = [np.nan]
    if not distMap_flag:
      distMap_log_sum = np.nan
      distMap_prob_list = [np.nan]
 
    ## determine wt or mut
    wt_aa,mut_aa,query_aa = '','',''
    for one_mut in re.split(':',mutants[0]):
      wt_aa += one_mut[0]
      mut_aa += one_mut[-1]
    for id_aa in mutants[1:]:
      query_aa += id_aa[0]
    
    if query_aa == wt_aa:
      structure_ratio_wt_list.append([mutants[0],fit_true,ss3_log_sum,ss3_env_log_sum,rsa2_log_sum,rsa2_env_log_sum,distMap_log_sum,ss3_prob_list,ss3_env_prob_list,rsa2_prob_list,rsa2_env_prob_list,distMap_prob_list])
    elif query_aa == mut_aa:
      structure_ratio_mut_list.append([mutants[0],ss3_log_sum,ss3_env_log_sum,rsa2_log_sum,rsa2_env_log_sum,distMap_log_sum,ss3_prob_list,ss3_env_prob_list,rsa2_prob_list,rsa2_env_prob_list,distMap_prob_list])
  
  structure_ratio_wt_df = pd.DataFrame(structure_ratio_wt_list,columns=['var_name','fit_true','ss3_log_sum_wt','ss3_env_log_sum_wt','rsa2_log_sum_wt','rsa2_env_log_sum_wt','distMap_log_sum_wt','ss3_prob_list_wt','ss3_env_prob_list_wt','rsa2_prob_list_wt','rsa2_env_prob_list_wt','distMap_prob_list_wt'])
  structure_ratio_mut_df = pd.DataFrame(structure_ratio_mut_list,columns=['var_name','ss3_log_sum_mut','ss3_env_log_sum_mut','rsa2_log_sum_mut','rsa2_env_log_sum_mut','distMap_log_sum_mut','ss3_prob_list_mut','ss3_env_prob_list_mut','rsa2_prob_list_mut','rsa2_env_prob_list_mut','distMap_prob_list_mut'])
    
  merged_ratio_df = fit_aa_pred_df.merge(structure_ratio_wt_df,how='left',on='var_name')
  merged_ratio_df = merged_ratio_df.merge(structure_ratio_mut_df,how='left',on='var_name')
  
  merged_ratio_df['log_ratio_ss'] = merged_ratio_df['ss3_log_sum_mut'] - merged_ratio_df['ss3_log_sum_wt']
  merged_ratio_df['log_ratio_rsa'] = merged_ratio_df['rsa2_log_sum_mut'] - merged_ratio_df['rsa2_log_sum_wt']
  merged_ratio_df['log_ratio_ss_env'] = merged_ratio_df['ss3_env_log_sum_mut'] - merged_ratio_df['ss3_env_log_sum_wt']
  merged_ratio_df['log_ratio_rsa_env'] = merged_ratio_df['rsa2_env_log_sum_mut'] - merged_ratio_df['rsa2_env_log_sum_wt']
  merged_ratio_df['log_ratio_dm'] = merged_ratio_df['distMap_log_sum_mut'] - merged_ratio_df['distMap_log_sum_wt']
  
  ## abs version
  ## handle unequal length
  ss3_prob_list_mut = np.array([i + [1.0]*(max_pos_num-len(i)) for i in merged_ratio_df['ss3_prob_list_mut'].to_list()])
  ss3_prob_list_wt = np.array([i + [1.0]*(max_pos_num-len(i)) for i in merged_ratio_df['ss3_prob_list_wt'].to_list()])
  rsa2_prob_list_mut = np.array([i + [1.0]*(max_pos_num-len(i)) for i in merged_ratio_df['rsa2_prob_list_mut'].to_list()])
  rsa2_prob_list_wt = np.array([i + [1.0]*(max_pos_num-len(i)) for i in merged_ratio_df['rsa2_prob_list_wt'].to_list()])
  ss3_env_prob_list_mut = np.array([i + [1.0]*(max_env_pos_num-len(i)) for i in merged_ratio_df['ss3_env_prob_list_mut'].to_list()])
  ss3_env_prob_list_wt = np.array([i + [1.0]*(max_env_pos_num-len(i)) for i in merged_ratio_df['ss3_env_prob_list_wt'].to_list()])
  rsa2_env_prob_list_mut = np.array([i + [1.0]*(max_env_pos_num-len(i)) for i in merged_ratio_df['rsa2_env_prob_list_mut'].to_list()])
  rsa2_env_prob_list_wt = np.array([i + [1.0]*(max_env_pos_num-len(i)) for i in merged_ratio_df['rsa2_env_prob_list_wt'].to_list()])
  distMap_prob_list_mut = np.array([i + [1.0]*(max_distMap_pos_num-len(i)) for i in merged_ratio_df['distMap_prob_list_mut'].to_list()])
  distMap_prob_list_wt = np.array([i + [1.0]*(max_distMap_pos_num-len(i)) for i in merged_ratio_df['distMap_prob_list_wt'].to_list()])

  merged_ratio_df['log_ratio_ss_abs'] = pd.Series(np.sum(np.abs(np.log(np.divide(ss3_prob_list_mut,ss3_prob_list_wt))),axis=1))
  merged_ratio_df['log_ratio_rsa_abs'] = pd.Series(np.sum(np.abs(np.log(np.divide(rsa2_prob_list_mut,rsa2_prob_list_wt))),axis=1))
  merged_ratio_df['log_ratio_ss_env_abs'] = pd.Series(np.sum(np.abs(np.log(np.divide(ss3_env_prob_list_mut,ss3_env_prob_list_wt))),axis=1))
  merged_ratio_df['log_ratio_rsa_env_abs'] = pd.Series(np.sum(np.abs(np.log(np.divide(rsa2_env_prob_list_mut,rsa2_env_prob_list_wt))),axis=1))
  merged_ratio_df['log_ratio_dm_abs'] = pd.Series(np.sum(np.abs(np.log(np.divide(distMap_prob_list_mut,distMap_prob_list_wt))),axis=1))
  
  target_merged_ratio_df = merged_ratio_df[merged_ratio_df['fit_true'].notnull()]
  out_json = {
    'mut_num':len(merged_ratio_df),
    'spearmanR_aa': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['aa_fit'].to_numpy(),nan_policy='omit')),
    'spearmanR_ss': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_ss'].to_numpy(),nan_policy='omit')),
    'spearmanR_rsa': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_rsa'].to_numpy(),nan_policy='omit')),
    'spearmanR_ss_env': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_ss_env'].to_numpy(),nan_policy='omit')),
    'spearmanR_rsa_env': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_rsa_env'].to_numpy(),nan_policy='omit')),
    'spearmanR_dm': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_dm'].to_numpy(),nan_policy='omit')),
    'spearmanR_ss_abs': tuple(scipy.stats.spearmanr(target_merged_ratio_df['fit_true'].to_numpy(),target_merged_ratio_df['log_ratio_ss_abs'].to_numpy(),nan_policy='omit')),
  }
    # 'pearsonR_aa': tuple(scipy.stats.pearsonr(filtered_fit_seq_structure_df['fit_true'].tolist(), filtered_fit_seq_structure_df['log_ratio_aa'].tolist())),
    # 'kendallTau_aa': tuple(scipy.stats.kendalltau(filtered_fit_seq_structure_df['fit_true'].tolist(), filtered_fit_seq_structure_df['log_ratio_aa'].tolist())),

  eval_path = f'{eval_save_dir}/{task}/predictions/{model_dir}'
  Path(eval_path).mkdir(parents=True, exist_ok=True)
  ## save metrics
  with open(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_structProp_metrics.json','w') as oj:
    json.dump(out_json,oj)

  report_str = 'eval_report*> '
  for k,v in out_json.items():
    report_str += f'{k}: {v};'
  print(report_str,flush=True)

  ## save log-ratio
  merged_ratio_df[['var_name','fit_true','aa_fit','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm','log_ratio_ss_abs','log_ratio_rsa_abs','log_ratio_ss_env_abs','log_ratio_rsa_env_abs','log_ratio_dm_abs']].to_csv(f'{eval_path}/{mutaSet_nm}_{epoch_dir}_structProp_rawScores.csv',index=False)

  return None

@registry.register_metric('mutation_embedding_umap')
def mutation_embedding_umap(value_dict:Dict, model_name:str=None, save_embeddings:bool=True, draw_fig:bool=True, eval_path:str='eval_results', pretrained_epoch:Union[str, int]=None, set_name:str=None):
  """generate umap for embedding space

  Arg:
    value_dict: embeddings of all_pos_ave, mut_pos_ave
  """
  if pretrained_epoch is None:
    pretrained_epoch = 'best'

  embedding_save_model_path = f'{eval_path}/embedding_analysis/embedding_save/{model_name}/{set_name}_{pretrained_epoch}'
  embedding_fig_path = f'{eval_path}/embedding_analysis/embedding_figure/{set_name}'
  if not os.path.isdir(embedding_save_model_path):
    os.makedirs(embedding_save_model_path,exist_ok=True)
  if not os.path.isdir(embedding_fig_path):
    os.makedirs(embedding_fig_path,exist_ok=True)

  if save_embeddings:
    for embed_range in ['all_pos_ave','mut_pos_ave','all_pos_ave_AA_head','mut_pos_ave_AA_head','fit_gt']:
      np.save(f'{embedding_save_model_path}/{embed_range}.npy',value_dict[embed_range])
 
  if draw_fig:
    metric_nm_list = ['euclidean', 'correlation', 'cosine']
    umap_obj_dict = {}
    for metric_nm in metric_nm_list:
      umap_corr_obj = umap.UMAP(n_components=2, metric=metric_nm, random_state=42, verbose=True)
      umap_obj_dict[metric_nm] = umap_corr_obj
    for metric_nm in metric_nm_list:
      for embed_range in ['all_pos_ave','mut_pos_ave']:
        embedding = umap_obj_dict[metric_nm].fit_transform(value_dict[embed_range])
        params = {'legend.fontsize': 10,
                'figure.figsize': (12, 12),
                'axes.labelsize': 14,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,}
        pylab.rcParams.update(params)
        fig, ax = plt.subplots()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=value_dict['fit_gt'], cmap="bwr", s=50)
        plt.colorbar()
        plt.savefig(f'{embedding_fig_path}/{metric_nm}_{embed_range}_umap_{model_name}_{pretrained_epoch}.png',dpi=600, bbox_inches='tight')
        plt.clf()

  return

@registry.register_metric('save_embedding')
@registry.register_metric('embed_antibody')
@registry.register_metric('embed_antibody_internal')
def none_metric():
  return
