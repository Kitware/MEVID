import torch

def global_KNN_cosine(feat_vec):#这个要要验证一下，cosine_similarity的输出是不是可以跨维度的。   reply:cosine_similarity是可以跨维度，或者说可以保持维度
    b,t,_ = feat_vec.size()
    refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
    feat_vec_avg = torch.mean(feat_vec, 1)
    similar_matrix = torch.zeros(b, t)

    for i in range(t):
        similar_score = torch.cosine_similarity(feat_vec_avg, feat_vec[:, i, :])
        similar_matrix[:, i] = similar_score

    remove_id = torch.argmin(similar_matrix, 1)

    for i in range(b):
        refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]  #b*t-1*1024

    cosine_sum_similar = 0
    for i in range(t-1):
        for j in range(i+1,t):
            cosine_similar_score = torch.cosine_similarity(refine_feature[:,i,:],refine_feature[:,j,:])
            cosine_similar_score = torch.div(cosine_similar_score+1,2)+0    #成比例压缩到(0,1)区间
            cosine_similar_score = -torch.log(cosine_similar_score)
            cosine_sum_similar = cosine_sum_similar + cosine_similar_score

    return refine_feature, cosine_sum_similar

def global_KNN_dist(feat_vec):
    b, t, _ = feat_vec.size()
    refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
    feat_vec_avg = torch.mean(feat_vec,1)
    similar_matrix = torch.zeros((b,t))

    for i in range(b):
        for j in range(t):
            similar_matrix[i,j] = torch.dist(feat_vec_avg[i],feat_vec[i,j,:])

    remove_id = torch.argmax(similar_matrix,1)

    for i in range(b):
        refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]

    cosine_sum_similar = 0
    for i in range(t - 1):
        for j in range(i + 1, t):
            cosine_similar_score = torch.cosine_similarity(refine_feature[:,i,:],refine_feature[:,j,:])
            cosine_similar_score = torch.div(cosine_similar_score+1,2)
            cosine_similar_score = -torch.log(cosine_similar_score)
            cosine_sum_similar = cosine_sum_similar + cosine_similar_score

    return refine_feature, cosine_sum_similar

def global_Center_cosine(feat_vec):
    b, t, _ = feat_vec.size()
    refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
    similar_matrix = torch.zeros(b,t,t)

    for i in range(t):
        for j in range(t):
            similar_matrix[:,i,j] = torch.cosine_similarity(feat_vec[:,i,:],feat_vec[:,j,:])

    similar_score = torch.sum(similar_matrix,2,keepdim=True)
    remove_id = torch.argmin(similar_score,1)

    for i in range(b):
        refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]

    cosine_sum_similar = 0
    for i in range(t - 1):
        for j in range(i + 1, t - 1):
            cosine_similar_score = torch.cosine_similarity(refine_feature[:,i,:],refine_feature[:,j,:])
            cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
            cosine_similar_score = -torch.log(cosine_similar_score)
            cosine_sum_similar = cosine_sum_similar + cosine_similar_score

    return refine_feature , cosine_sum_similar

def global_Center_dist(feat_vec):
    b, t, _ = feat_vec.size()
    refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
    similar_matrix = torch.zeros(b, t, t)

    for i in range(b):
        for j in range(t):
            for k in range(t):
                similar_matrix[i, j, k] = torch.dist(feat_vec[i, j, :], feat_vec[i, k, :])

    similar_score = torch.sum(similar_matrix, 2, keepdim=True)
    remove_id = torch.argmax(similar_score, 1)

    for i in range(b):
        refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]

    cosine_sum_similar = 0
    for i in range(t - 1):
        for j in range(i + 1, t - 1):
            cosine_similar_score = torch.cosine_similarity(refine_feature[:,i,:],refine_feature[:,j,:])
            cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
            cosine_similar_score = -torch.log(cosine_similar_score)
            cosine_sum_similar = cosine_sum_similar + cosine_similar_score

    return refine_feature, cosine_sum_similar

def local_KNN_cosine(feat_vec):   #我感觉这一步可能有问题，这一步可以在测试的时候做，训练的时候就别做了。 reply:没问题的，这一步是可以做的，当初出问题的地方在特征传递那部分，不在这里。
    b,t,c,n = feat_vec.size()
    similar_matrix = torch.zeros((b,n,t))

    for i in range(n):
        avg_feature = torch.mean(feat_vec[:,:,:,i],1)
        for j in range(t):
            similar_matrix[:,i,j] = torch.cosine_similarity(feat_vec[:,j,:,i],avg_feature)

    remove_id = torch.argmin(similar_matrix,2)
    refine_feature = torch.zeros((b, t-1, c, n))

    for i in range(b):
        for j in range(n):
            refine_feature[i,:,:,j] = feat_vec[i, torch.arange(t) != remove_id[i,j], :, j]

    refine_feature = torch.mean(refine_feature,-1)

    return refine_feature