import numpy as np
import torch
import torch.nn.functional as F
import sys
import pandas as pd
from progressbar import ProgressBar, AnimatedMarker, Percentage
import math
from tqdm import trange
import os
import os.path as osp


def evaluate_all(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def Video_Cmc(features, ids, cams, oids, query_idx, rank_size):
    """
    features: numpy array of shape (n, d)
    label`s: numpy array of shape (n)
    """
    # Sample query
    data = {'feature':features, 'id':ids, 'cam':cams, 'oid':oids}

    q_idx = query_idx
    g_idx = [i for i in range(len(ids)) if data['id'][i]!=-1]
    q_data = {k:v[q_idx] for k, v in data.items()}
    g_data = {k:v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size: rank_size = len(g_idx)

    all_CMC, all_mAP = Cmc(q_data, g_data, rank_size)

    #attribute evaluation
    distmat = np_cdist(q_data['feature'], g_data['feature'])
    q_pids, g_pids = q_data['id'], g_data['id']
    q_camids, g_camids = q_data['cam'], g_data['cam']
    q_oids, g_oids = q_data['oid'], g_data['oid']


    ranks = [1, 5, 10, 20]

    be_cmc, be_mAP = evaluate_all(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Overall Results ----------")
    print("mAP: {:.1%}".format(be_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, be_cmc[r - 1]))
    print("------------------")


    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='SC')
    print("Same Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='CC')
    print("Changing Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL')
    print("Same Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='DL')
    print("Changing Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS')
    print("Same Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='DS')
    print("Changing Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")    

    return all_CMC, all_mAP


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

    dataset_dir = '../../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'reid_final', 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    camera_file = osp.join(dataset_dir, 'scripts', 'test_track_sizes.txt')
    camera_set = np.genfromtxt(camera_file,dtype='str')[:,0]
    q_locationids = camera_set[query_IDX]
    gallery_IDX = [i for i in range(camera_set.shape[0]) if i not in query_IDX]
    g_locationids = camera_set[gallery_IDX]
    for k in range(q_locationids.shape[0]):
        q_locationids[k] = 0 if int(q_locationids[k][9:12]) in in_cam else 1
    for k in range(g_locationids.shape[0]):
        g_locationids[k] = 0 if int(g_locationids[k][9:12]) in in_cam else 1

    '''for k in range(len(in_cam)):
        if k == 0:
            location_index = np.argwhere(g_camids==in_cam[k])
        else:
            cur_index = np.argwhere(g_camids==in_cam[k])
            location_index = np.union1d(location_index, cur_index)'''


    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    
    '''if 1:#mode == 'IN':
        for k in range(len(in_cam)):
            if k == 0:
                location_index = np.argwhere(g_camids==in_cam[k])
            else:
                cur_index = np.argwhere(g_camids==in_cam[k])
                location_index = np.union1d(location_index, cur_index)
    else:
        for k in range(len(out_cam)):
            if k == 0:
                location_index = np.argwhere(g_camids==out_cam[k])
            else:
                cur_index = np.argwhere(g_camids==out_cam[k])
                location_index = np.union1d(location_index, cur_index)'''

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



    '''for i in range(num_q):
        if mode == 'IN':
            if q_camids[i] not in in_cam:
                continue
        else:
            if q_camids[i] not in out_cam:
                continue

        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        query_index = np.intersect1d(query_index, location_index)
        camera_index = np.intersect1d(camera_index, location_index)

        # groundtruth index
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp'''

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
    
    dataset_dir = '../../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'reid_final', 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    scale_file = osp.join(dataset_dir, 'scripts', 'test_track_sizes.txt')
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


def custom_video_cmc(g_features, g_ids, g_cams, q_features, q_ids, q_cams, rank_size):

    g_data = {'feature':g_features, 'id':g_ids, 'cam':g_cams}
    q_data = {'feature':q_features, 'id':q_ids, 'cam':q_cams}

    CMC, mAP = Cmc(q_data, g_data, rank_size)
    return CMC, mAP


def Cmc(q_data, g_data, rank_size):

    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    distmat = np_cdist(q_data['feature'], g_data['feature']) # Reture a n_query*n_gallery array
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(n_gallery)
    AP = 0

    for i in range(n_query):
        # groundtruth index
        query_index = np.argwhere(g_data['id']==q_data['id'][i])
        camera_index = np.argwhere(g_data['cam']==q_data['cam'][i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = Compute_AP(good_index, junk_index, index[i])
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (n_query - num_no_gt)
    mAP = AP / (n_query - num_no_gt)

    return CMC, mAP

def Compute_AP(good_index, junk_index, index):
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

def cdist(feat1, feat2):
    """Cosine distance"""
    feat1 = torch.FloatTensor(feat1)#.cuda()
    feat2 = torch.FloatTensor(feat2)#.cuda()
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
    dist = -1 * torch.mm(feat1, feat2)
    return dist.cpu().numpy()

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def np_norm_eudist(feat1,feat2):
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return np.sqrt(feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)+ 1e-12)


def sqdist(feat1, feat2, M=None):
    """Mahanalobis/Euclidean distance"""
    if M is None: M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)

if __name__ == '__main__':
    from scipy.io import loadmat
    q_feature = loadmat(sys.argv[1])['ff']
    q_db_txt = sys.argv[2]
    g_feature = loadmat(sys.argv[3])['ff']
    g_db_txt = sys.argv[4]
    #print(feature.shape)
    CMC, mAP = Self_Cmc(g_feature, g_db_txt, 100)
    #CMC, mAP = Vanilla_Cmc(q_feature, q_db_txt, g_feature, g_db_txt)
    print('r1 precision = %f, mAP = %f' % (CMC[0], mAP))
