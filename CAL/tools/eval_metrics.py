import logging
import numpy as np
import pdb
import os
import os.path as osp

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


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """
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
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP


def _get_names(fpath):
    names = []
    with open(fpath, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names


def _process_mevid_data(track_names, track_info, home_dir, relabel=False, clothes2label=None):
    pid_container = dict()
    loc_set = {339:0, 436:0, 424:0, 340:0, 336:0, 328:0,
               341:1, 506:1, 505:1,
               507:2, 331:2,
               299:3, 330:3,
               420:4, 419:4,
               639:5,
               326:6, 329:6,
               508:7,
               509:8,
               423:9,
               421:10,
               301:11,
               #509:12,
               #508:13,
               300:14,
               638:15}

    num_tracklets = track_info.shape[0]
    for track_idx in range(num_tracklets):
        cur_trk = track_info[track_idx,...]
        sta_idx, end_idx, pid, clothes_label, camid = cur_trk
        img_names = track_names[sta_idx:end_idx+1]
        if len(img_names) == 0:
            continue

        if int(pid) not in pid_container.keys():
            pid_container[int(pid)] = [int(clothes_label)]
        else:
            pid_container[int(pid)].append(int(clothes_label))

    aa = np.unique(track_info[:,4])
    
    oid_cnt = dict()
    pids = list(pid_container.keys())
    max_oid = 0
    for idx in range(len(pids)):
        oid_set = set(pid_container[pids[idx]])
        oid_cnt[int(pids[idx])] = len(oid_set)
        if len(oid_set) > max_oid:
            max_oid = len(oid_set)
   
    oid_histo = np.zeros(int(max_oid))
    cid_histo = np.zeros(16)

    for track_idx in range(num_tracklets):
        cur_trk = track_info[track_idx,...]
        sta_idx, end_idx, pid, clothes_label, camid = cur_trk
        num_outfits = oid_cnt[int(pid)]
        oid_histo[int(num_outfits-1)] += 1
    print(oid_histo)

    for track_idx in range(num_tracklets):
        cur_trk = track_info[track_idx,...]
        sta_idx, end_idx, pid, clothes_label, camid = cur_trk
        idx = loc_set[int(camid)]
        cid_histo[idx] += 1

    print(cid_histo)


def compute_data():
    dataset_dir = '../../mevid'
    # clothes
    root = dataset_dir
    train_name_path = osp.join(root, 'train_name.txt')
    train_info_path = osp.join(root, 'track_train_info.txt')
    test_name_path = osp.join(root, 'test_name.txt')
    test_info_path = osp.join(root, 'track_test_info.txt')
    train_info = np.loadtxt(train_info_path).astype(np.int)
    test_info = np.loadtxt(test_info_path).astype(np.int)
    train_names = _get_names(train_name_path)
    test_names = _get_names(test_name_path)
    #_process_mevid_data(train_names, train_info, 'bbox_train', relabel=True)
    #_process_mevid_data(test_names, test_info, 'bbox_test', relabel=True)
  
    # scales
    train_scale_file = '../train_track_sizes.txt'
    test_scale_file = '../test_track_sizes.txt'
    scale_set = np.loadtxt(train_scale_file)
    scale_set = np.loadtxt(test_scale_file)
    bins = [32, 48, 64, 96, 128, 192, 580]
    num = len(scale_set)
    scale_cnt = np.zeros(6)
    for i in range(num):
        cur_size = scale_set[i]
        for k in range(6):
            if cur_size >= bins[k] and cur_size < bins[k+1]:
                scale_cnt[k] += 1
 
def evaluate_fine_scales(distmat, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP with scales

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """

    
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    scale_file = '../test_track_sizes.txt'
    scale_set = np.loadtxt(scale_file)
    q_scales = scale_set[query_IDX]
    min_size, max_size = np.min(q_scales), np.max(q_scales)
    num_scale = 6
    bins = [32, 48, 64, 96, 128, 192, 480]
    scale_idx = np.zeros(num_q)
    for i in range(num_q):
        cur_size = q_scales[i]
        for k in range(num_scale):
            if cur_size >= bins[k] and cur_size < bins[k+1]:
                scale_idx[i] = k        

    num_query, num_no_gt = np.zeros(num_scale), np.zeros(num_scale) # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros((num_scale, len(g_pids)))
    AP, mAP = np.zeros(num_scale), np.zeros(num_scale)

    for i in range(num_q):
        idx = int(scale_idx[i])
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt[idx] += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC[idx] = CMC[idx] + CMC_tmp
        AP[idx] += ap_tmp
        num_query[idx] += 1
    
    for idx in range(num_scale):
        CMC[idx] = CMC[idx] / (num_query[idx] - num_no_gt[idx])
        mAP[idx] = AP[idx] / (num_query[idx] - num_no_gt[idx])
    print(num_query)
    return CMC, mAP

def evaluate_fine_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids):
    """ Compute CMC and mAP with clothes

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): clothes IDs for query samples.
        g_clothids (numpy array): clothes IDs for gallery samples.
    """

    
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    pid_set = np.unique(q_pids)
    cid_num = {}
    for pid in pid_set:
        query_index = np.argwhere(q_pids==pid)
        cid_set = q_clothids[query_index]  
        cid_num[int(pid)] = len(np.unique(cid_set))      

    num_clothes = 3
    num_query, num_no_gt = np.zeros(num_clothes), np.zeros(num_clothes) # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros((num_clothes, len(g_pids)))
    AP, mAP = np.zeros(num_clothes), np.zeros(num_clothes)

    for i in range(num_q):
        idx = cid_num[int(q_pids[i])]-1
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt[idx] += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC[idx] = CMC[idx] + CMC_tmp
        AP[idx] += ap_tmp
        num_query[idx] += 1
    
    for idx in range(num_clothes):
        CMC[idx] = CMC[idx] / (num_query[idx] - num_no_gt[idx])
        mAP[idx] = AP[idx] / (num_query[idx] - num_no_gt[idx])
    print(num_query)
    return CMC, mAP

def evaluate_fine_locations(distmat, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    in_cam = [330, 329, 507, 508, 509]
    out_cam = [436, 505, 336, 340, 639, 301]
    loc_set = {639:0, 436:1, 336:1, 340:1, 507:2, 505:3, 301:4, 330:5, 508:6, 509:7, 329:8}
    num_loc = 9
    num_query, num_no_gt = np.zeros(num_loc), np.zeros(num_loc) # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros((num_loc, len(g_pids)))
    AP, mAP = np.zeros(num_loc), np.zeros(num_loc)

    for i in range(num_q):
        idx = loc_set[q_camids[i]+1]
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt[idx] += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC[idx] = CMC[idx] + CMC_tmp
        AP[idx] += ap_tmp
        num_query[idx] += 1
    
    for idx in range(num_loc):
        CMC[idx] = CMC[idx] / (num_query[idx] - num_no_gt[idx])
        mAP[idx] = AP[idx] / (num_query[idx] - num_no_gt[idx])
    print(num_query)
    return CMC, mAP


def evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC'):
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
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

def evaluate_with_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL'):
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
        if mode == 'DL':
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


def evaluate_with_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS'):
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
    camera_file = '../test_track_scale.txt'
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
