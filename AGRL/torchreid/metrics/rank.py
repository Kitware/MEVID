from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import numpy as np
import copy
from collections import defaultdict
import sys
import warnings
import pdb

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


def evaluate_mars(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_q, num_g = distmat.shape
    cmc = np.zeros((num_q, max_rank))
    ap = np.zeros(num_q)

    for k in range(num_q):
        good_idx = np.where((q_pids[k] == g_pids) & (q_camids[k] != g_camids))[0]
        junk_mask1 = (g_pids == -1)
        junk_mask2 = (q_pids[k] == g_pids) & (q_camids[k] == g_camids)
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        score = distmat[k, :]
        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:max_rank]

        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)
    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP


def evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL'):
    """ Compute CMC and mAP with locations

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        mode: 'SL' for same locations; 'DL' for different locations.
    """
    assert mode in ['SL', 'DL']
    
    in_cam = [330, 329, 507, 508, 509]
    out_cam = [436, 505, 336, 340, 639, 301]

    dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    camera_file = '../test_track_scale.txt'
    camera_set = np.genfromtxt(camera_file,dtype='str')[:,0]
    q_locationids = camera_set[query_IDX]
    gallery_IDX = [i for i in range(camera_set.shape[0]) if i not in query_IDX]
    g_locationids = camera_set[gallery_IDX]
    for k in range(q_locationids.shape[0]):
        q_locationids[k] = 0 if int(q_locationids[k][9:12]) in in_cam else 1
    for k in range(g_locationids.shape[0]):
        g_locationids[k] = 0 if int(g_locationids[k][9:12]) in in_cam else 1

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    
    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        location_index = np.argwhere(g_locationids==q_locationids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'SL':
            good_index = np.setdiff1d(good_index, location_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, location_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP


def evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS'):
    """ Compute CMC and mAP with scales

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        mode: 'SS' for same size; 'DS' for diff. size.
    """
    assert mode in ['SS', 'DS']
    
    dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    scale_file = '../test_track_scale.txt'
    scale_set = np.genfromtxt(scale_file,dtype='str')[:,1]
    q_scaleids = scale_set[query_IDX]
    gallery_IDX = [i for i in range(scale_set.shape[0]) if i not in query_IDX]
    g_scaleids = scale_set[gallery_IDX]

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        scale_index = np.argwhere(g_scaleids==q_scaleids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'DS':
            good_index = np.setdiff1d(good_index, scale_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, scaleid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, scale_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different scaleid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP


def evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC'):
    """ Compute CMC and mAP with clothes

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): clothes IDs for query samples.
        g_clothids (numpy array): clothes IDs for gallery samples.
        mode: 'CC' for clothes-changing; 'SC' for the same clothes.
    """
    assert mode in ['CC', 'SC']
    
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothids==q_clothids[i])
        cloth_index = np.intersect1d(cloth_index, query_index)
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, cloth_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc


def Compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n - njunk:] = 1
            flag = 1  # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue  # junk image

        if flag == 1:
            intersect_size += 1
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap += (recall - old_recall) * (old_precision + precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == ngood:
            return ap, cmc
    return ap, cmc


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False,
                  use_metric_market1501=False, use_metric_mars=False, use_cython=True):
    """
    Evaluate CMC and mAP.
    :param distmat: (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
    :param q_pids: (numpy.ndarray): 1-D array containing person identities of each query instance.
    :param g_pids: (numpy.ndarray): 1-D array containing person identities of each gallery instance.
    :param q_camids: 1-D array containing camera views under which each query instance is captured.
    :param g_camids: 1-D array containing camera views under which each gallery instance is captured.
    :param max_rank: maximum CMC rank to be computed. Default is 50.
    :param use_metric_cuhk03:
    :param use_metric_market1501:
    :param use_metric_mars:
    :param use_metric_dukev: same as use_metric_mars
    :param use_cython:
    :return:
    """
    if use_metric_market1501 or use_metric_cuhk03:
        if use_cython and IS_CYTHON_AVAI:
            return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
        else:
            return evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
    elif use_metric_mars:
        return evaluate_mars(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


# add from Dukemtmc video reid
#from sklearn.metrics.base import _average_binary_score
#from sklearn.metrics import precision_recall_curve, auc

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None):
    def _binary_average_precision(y_true, y_score, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        return auc(recall, precision)

    return _average_binary_score(_binary_average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)


def cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100,
        separate_camera_set=False, single_gallery_shot=False, first_match_break=False):
    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams):
    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def evaluate_dukev(distmat, query_ids, gallery_ids, query_cams, gallery_cams, max_rank=50):
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

    # Compute all kinds of CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    return cmc_scores['market1501'], mAP
