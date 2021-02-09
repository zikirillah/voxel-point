import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
import numpy as np
from functools import partial
from scipy import stats
import math
import os
import sys
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, './utils'))
import tf_util

N = 16 # voxel size is N x N x N
K = 6 # each cell has K points
NUM_CATEGORY = 50
NUM_SEG_PART = 1+1
NUM_PER_POINT_FEATURES = 3
NUM_FEATURES = K * NUM_PER_POINT_FEATURES + 1

batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, scope='bn', updates_collections=None)

def leak_relu(x, leak=0.1, scope=None):
    return tf.where(x >= 0, x, leak * x)


def dist(p1, p2):
    return np.sqrt((p1[:, 0].reshape(1, -1) - p2[:, 0].reshape(-1, 1))**2 + \
                   (p1[:, 1].reshape(1, -1) - p2[:, 1].reshape(-1, 1))**2)


def choose_k_furthest(points, k):
    chosen = []
    chosen.append(np.random.randint(0, len(points)-1))
    for i in range(1, k):
        chosen.append(dist(points[chosen, :], points).min(axis=1).argmax())
    return chosen

def knn (points, labels, k):
    D = distance.squareform(distance.pdist(points))
    first_choice = np.random.randint(0, len(D))
    closest = np.argsort(D, axis=1)
    choice = closest[first_choice]
    choosen = choice[0:k]

    choice1 = []
    label = []
    for i in range(len(choosen)):
        choice1.insert(i, points[choosen[i]])
        label.insert(i, labels[choosen[i]])
    return label
    
    
def furthest_ppoint_sampling(points, label, k):
    D = pairwise_distances(points, metric='euclidean')
    def getGreedyPerm(D):
        """
        A Naive O(N^2) algorithm to do furthest points sampling

        Parameters
        ----------
            D : ndarray (N, N)
            An NxN distance matrix for points
        Return
        ------
        tuple (list, list)
            (permutation (N-length array of indices),
            lambdas (N-length array of insertion radii))
        """

        N = D.shape[0]
        # By default, takes the first point in the list to be the
        # first point in the permutation, but could be random
        perm = np.zeros(N, dtype=np.int64)
        lambdas = np.zeros(N)
        ds = D[0, :]
        for i in range(1, N):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        return perm
    perm = getGreedyPerm(D)

    selected = perm[0:k]

    choice = []
    labels = []
    for i in range(len(selected)):
        choice.insert(i, points[selected[i]])
        labels.insert(i, label[selected[i]])
    return labels


def integer_label_to_one_hot_label(integer_label):
    if (len(integer_label.shape) == 0):
        one_hot_label = np.zeros((NUM_CATEGORY))
        one_hot_label[int(integer_label)] = 1
    elif (len(integer_label.shape) == 1):
        one_hot_label = np.zeros((integer_label.shape[0], NUM_SEG_PART))
        for i in range(integer_label.shape[0]):
            one_hot_label[i, int(integer_label[i])] = 1
    elif (len(integer_label.shape) == 4):
        one_hot_label = np.zeros((N, N, N, K, NUM_SEG_PART))
        for i in range(N):
          for j in range(N):
            for k in range(N):
              for l in range(K):
                one_hot_label[i, j, k, l, int(integer_label[i, j, k, l])] = 1
    else:
        raise
    return one_hot_label

def voxel_feature(points, k):

    magnititudes = []
    angles = []
    distances = []
    i= 0
    while (i < k):
        if (i < k-1):
            magnititude = np.math.sqrt((points[i][0]) ** 2 + (points[i][1]) ** 2 + (points[i][2]) ** 2)
            dotproduct = points[i][0]*points[i+1][0] + points[i][1]*points[i+1][1]+points[i][2]*points[i+1][2]
            magnititude2 = np.math.sqrt((points[i+1][0])**2 + (points[i+1][1])**2+(points[i+1][2])**2)
            angle = dotproduct/(magnititude*magnititude2)
            distance = np.math.sqrt(((points[i][0] - points[i+1][0]) ** 2) + ((points[i][1] - points[i+1][1]) ** 2) + (
                    (points[i][2] - points[i+1][2]) ** 2))

            magnititudes.insert(i, magnititude)
            angles.insert(i, angle)
            distances.insert(i, distance)
        else:
            magnititude = np.math.sqrt((points[i][0]) ** 2 + (points[i][1]) ** 2 + (points[i][2]) ** 2)
            dotproduct = points[0][0] * points[i][0] + points[0][1] * points[i][1] + points[0][2] * points[i][2]
            magnititude2 = np.math.sqrt((points[0][0]) ** 2 + (points[0][1]) ** 2 + (points[0][2]) ** 2)
            angle = dotproduct / (magnititude * magnititude2)
            distance = np.math.sqrt(((points[i][0] - points[0][0]) ** 2) + ((points[i][1] - points[0][1]) ** 2) + (
                (points[i][2] - points[0][2]) ** 2))

            magnititudes.insert(i, magnititude)
            angles.insert(i, angle)
            distances.insert(i, distance)
        i = i+1

    voxel_features = [angles, distances, magnititudes]
    return voxel_features

def centeroidnp(point, lab):
    sum_x = np.mean(point, axis=0)
    centeroid = sum_x.tolist()
    label = lab[0]
    return centeroid, label



def pc2voxel(pc, pc_label):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     pc_label: size n x NUM_SEG_PART (one-hot encoding label)
    # Returns:
    #     voxel: N x N x N x K x (3+3)
    #     label: N x N x N x (K+1) x NUM_SEG_PART
    #     index: N x N x N x K

    num_points = pc.shape[0]
    data = np.zeros((N, N, N, NUM_FEATURES), dtype=np.float32)
    label = np.zeros((N, N, N, K+1, NUM_SEG_PART), dtype=np.float32)
    index = np.zeros((N, N, N, K), dtype=np.float32)
    xyz = pc[:, 0 : 3]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz -= centroid
    xyz /= np.amax(np.sqrt(np.sum(xyz ** 2, axis=1)), axis=0) * 1.05
    idx = np.floor((xyz + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
      for j in range(N):
        for k in range(N):
          u = int(i * N * N + j * N + k)
          if not L[u]:
              data[i, j, k, :] = np.zeros((NUM_FEATURES), dtype=np.float32)
              label[i, j, k, :, :] = 0
              label[i, j, k, :, 0] = 1
          elif (len(L[u]) >= K):
              #for VFE 1
              #choice = furthest_ppoint_sampling(pc[L[u]], L[u], K)
              
              #for VFE 2
              choice = knn(pc[L[u]], L[u], K)
              
              #for both VFE 1 and 2
              choice1 = pc[choice]
              voxel_features = voxel_feature(choice1, K)
              
              #for VFE 3              
              #voxel_features, choice = centeroidnp(pc[L[u]], L[u])
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(voxel_features, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
          else:
              choice = np.random.choice(L[u], size=K, replace=True)
              choice1 = pc[choice]
              voxel_features = voxel_feature(choice1, K)
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(voxel_features, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
              
    return data, label, index



def rotate_pc(pc):
    # Args:
    #     pc: size n x 3
    # Returns:
    #     rotated_pc: size n x 3
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc



def populateIntegerSegLabel(pc, voxel_label, index):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     voxel_label: size N x N x N x (K+1)
    #     index: size N x N x N x K
    # Returns:
    #     label: size n x 1

    num_points = pc.shape[0]
    label = np.zeros((num_points), dtype=np.int32)
    xyz = pc[:, 0 : 3]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz -= centroid
    xyz /= np.amax(np.sqrt(np.sum(xyz ** 2, axis=1)), axis=0) * 1.05
    idx = np.floor((xyz + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
      for j in range(N):
        for k in range(N):
          u = int(i * N * N + j * N + k)
          if not L[u]:
              pass
          elif (len(L[u]) >= K):
              label[L[u]] = voxel_label[i, j, k, K]
              for s in range(K):
                  label[int(index[int(i), int(j), int(k), int(s)])] = int(voxel_label[int(i), int(j), int(k), int(s)])
          else:
              for s in range(K):
                  label[int(index[int(i), int(j), int(k), int(s)])] = int(voxel_label[int(i), int(j), int(k), int(s)])
    return label



def populateOneHotSegLabel(pc, voxel_label, index):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     voxel_label: size N x N x N x (K+1) x NUM_SEG_PART
    #     index: size N x N x N x K
    # Returns:
    #     label: size n x 1
    return populateIntegerSegLabel(pc, np.argmax(voxel_label, axis=4), index)
       
    

def get_model(the_model, is_training):
    # Args:
    #     the_model: of size B x N x N x N x NUM_FEATURES
    #     is_training: boolean tensor
    # Returns:
    #     pred_cat: of size B x NUM_CATEGORY
    #     pred_seg: of size B x N x N x N x (K+1) x NUM_PART_SEG
   
    # Encoder
    batch_size = the_model.get_shape()[0].value
    conv1 = tf_util.conv3d(the_model, 64, [5,5,5], scope='conv1', activation_fn=leak_relu, bn=True, is_training=is_training) # N
    conv2 = tf_util.conv3d(conv1, 64, [5,5,5], scope='conv2', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/2
    conv3 = tf_util.conv3d(conv2, 64, [5,5,5], scope='conv3', activation_fn=leak_relu, bn=True, is_training=is_training) # N/2
    conv4 = tf_util.conv3d(conv3, 128, [3,3,3], scope='conv4', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/4
    conv5 = tf_util.conv3d(conv4, 128, [3,3,3], scope='conv5', activation_fn=leak_relu, bn=True, is_training=is_training) # N/4
    conv6 = tf_util.conv3d(conv5, 256, [3,3,3], scope='conv6', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/8
    conv7 = tf_util.conv3d(conv6, 256, [3,3,3], scope='conv7', activation_fn=leak_relu, bn=True, is_training=is_training) # N/8
    conv8 = tf_util.conv3d(conv7, 512, [3,3,3], scope='conv8', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/16
    conv9 = tf_util.conv3d(conv8, 512, [1,1,1], scope='conv9', activation_fn=leak_relu, bn=True, is_training=is_training) # N/16

    # Classification Network
    conv9_flat = tf.reshape(conv9, [batch_size, -1])
    fc1 = tf_util.fully_connected(conv9_flat, 512, activation_fn=leak_relu, bn=True, is_training=is_training, scope='fc1')
    do1 = tf_util.dropout(fc1, keep_prob=0.7, is_training=is_training, scope='do1')
    fc2 = tf_util.fully_connected(do1, 256, activation_fn=leak_relu, bn=True, is_training=is_training, scope='fc2')
    do2 = tf_util.dropout(fc2, keep_prob=0.7, is_training=is_training, scope='do2')
    pred_cat = tf_util.fully_connected(do2, NUM_CATEGORY, activation_fn=None, bn=False, scope='pred_cat')

    # Segmentation Network
    cat_features = tf.tile(tf.reshape(tf.concat([fc2, pred_cat], axis=1), [batch_size, 1, 1, 1, -1]), [1, N//16, N//16, N//16, 1])
    conv9_cat = tf.concat([conv9, cat_features], axis=4)
    deconv1 = tf_util.conv3d_transpose(conv9_cat, 256, [3,3,3], scope='deconv1', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/8
    conv7_deconv1 = tf.concat(axis=4, values=[conv7, deconv1])
    deconv2 = tf_util.conv3d(conv7_deconv1, 256, [3,3,3], scope='deconv2', activation_fn=leak_relu, bn=True, is_training=is_training) # N/8
    deconv3 = tf_util.conv3d_transpose(deconv2, 128, [3,3,3], scope='deconv3', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/4
    conv5_deconv3 = tf.concat(axis=4, values=[conv5, deconv3])
    deconv4 = tf_util.conv3d(conv5_deconv3, 128, [3,3,3], scope='deconv4', activation_fn=leak_relu, bn=True, is_training=is_training) # N/4
    deconv5 = tf_util.conv3d_transpose(deconv4, 64, [3,3,3], scope='deconv5', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/2
    conv3_deconv5 = tf.concat(axis=4, values=[conv3, deconv5])
    deconv6 = tf_util.conv3d(conv3_deconv5, 64, [5,5,5], scope='deconv6', activation_fn=leak_relu, bn=True, is_training=is_training) # N/2
    deconv7 = tf_util.conv3d_transpose(deconv6, 64, [5,5,5], scope='deconv7', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N
    conv1_deconv7 = tf.concat(axis=4, values=[conv1, deconv7])
    deconv8 = tf_util.conv3d(conv1_deconv7, 64, [5,5,5], scope='deconv8', activation_fn=leak_relu, bn=True, is_training=is_training) # N

    pred_seg = tf_util.conv3d(deconv8, (K+1) * NUM_SEG_PART, [5,5,5], scope='pred_seg', activation_fn=None, bn=False, is_training=is_training)
    pred_seg = tf.reshape(pred_seg, [batch_size, N, N, N, K+1, NUM_SEG_PART])

    return pred_cat, pred_seg



def get_loss(pred_cat, one_hot_cat, pred_seg, one_hot_seg):
    per_instance_cat_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_cat, labels=one_hot_cat)
    cat_loss = tf.constant(200.0, dtype=tf.float32) * tf.reduce_mean(per_instance_cat_loss)
    per_instance_seg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_seg, labels=one_hot_seg)
    seg_loss = tf.constant(800.0, dtype=tf.float32) * tf.reduce_mean(per_instance_seg_loss)
    total_var = tf.trainable_variables()
    reg_vars = [var for var in total_var if 'weights' in var.name]
    reg_loss = tf.zeros([], dtype=tf.float32)
    for var in reg_vars:
        reg_loss += tf.nn.l2_loss(var)
    reg_loss = tf.constant(1e-5, dtype=tf.float32) * reg_loss
    total_loss = cat_loss + seg_loss + reg_loss
    return total_loss, cat_loss, seg_loss



def intersection_over_union(pred_seg, integer_seg_label):
    iou, counts = 0.0, 0.0
    for i in range(1, NUM_SEG_PART):
        intersection = np.sum(np.logical_and(pred_seg == i, integer_seg_label == i))
        union = np.sum(np.logical_or(pred_seg == i, integer_seg_label == i))
        if (union > 0):
            counts += 1.0
            iou += (float(intersection) / float(union))
    iou /= counts
    return iou
