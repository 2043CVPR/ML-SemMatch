import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

# COLA
from utils.voxel_downsample import voxel_downsample_with_label


def extract_init_guidpost_outdoor(
    pts, labels,
    label_usefull,
    is_vis=False,
    **kwargs
):
    # extrct usefull pts
    indic = np.any(labels[:, None] == label_usefull[None, :], axis=1)
    pts = pts[indic]
    labels = labels[indic]

    if len(pts) == 0:
        return None, None

    label_unique, counts_unique = np.unique(labels, return_counts=True)
    label_unique = label_unique[counts_unique >= 5]
    counts_unique = counts_unique[counts_unique >= 5]

    # label -> cluster -> pts
    guidpost_centers = {}  # label: pts
    dbscaner = DBSCAN(eps=1.5, min_samples=3)
    for label_gd in label_usefull:
        if label_gd not in label_unique:
            continue

        if is_vis:
            print(label_gd)

        # Nx3 points with same label `label_gt`
        pts_group = pts[labels == label_gd]

        # cluster -> centers in xy-plane
        cluster_labels = dbscaner.fit_predict(
            pts_group[:, :2])  

        cluster_label_unique = np.unique(cluster_labels)
        clu_centers = []
        geo_list = []
        for cluster_l in cluster_label_unique: 
            if cluster_l == -1:
                continue

            pts_cluster = pts_group[cluster_labels == cluster_l]  
            cluster_center = np.mean(pts_cluster, axis=0)

            clu_centers.append(cluster_center)

        if len(clu_centers) != 0:  
            clu_centers = np.asarray(clu_centers)  # Nx2
            guidpost_centers[label_gd] = clu_centers

    # counts and std
    guidpost_eval = {}
    for label, pts in guidpost_centers.items():
        guidpost_eval[label] = [len(pts), np.linalg.norm(np.std(pts, axis=0))]

    return guidpost_centers, guidpost_eval


def get_anchor_pts_with_labels_outdoor(
    pts_src, label_src,
    pts_dst, label_dst,
    max_num_label,
    label_usefull,
    is_vis_guidpost=False,
    **args,
):
    label_usefull = np.asarray(label_usefull)
    # get guidpost
    guidpost_centers_src, guidpost_eval_src = extract_init_guidpost_outdoor(
        pts_src, label_src, is_vis=is_vis_guidpost, label_usefull=label_usefull)
    guidpost_centers_dst, guidpost_eval_dst = extract_init_guidpost_outdoor(
        pts_dst, label_dst, is_vis=is_vis_guidpost, label_usefull=label_usefull)

    # check None
    if np.any(np.asarray([guidpost_centers_src, guidpost_eval_src, guidpost_centers_dst, guidpost_eval_dst], dtype=object) == None):
        print("There are no guidport.")
        return False, None, None, None, None, None

    # guidpost to pts with label
    guidpost_pts_src = []
    guidpost_label_src = []
    guidpost_pts_dst = []
    guidpost_label_dst = []

    for label in guidpost_centers_src.keys():
        if label not in guidpost_centers_dst:
            continue

        if abs(guidpost_eval_src[label][0] - guidpost_eval_dst[label][0]) > 5:
            continue

        if (guidpost_eval_src[label][0] > 10 and guidpost_eval_src[label][1] <= 10) or (guidpost_eval_dst[label][0] > 10 and guidpost_eval_dst[label][1] <= 10):
            continue

        if len(guidpost_centers_src[label]) == 0 or len(guidpost_centers_dst[label]) == 0:
            continue

        # record
        guidpost_pts_src.append(guidpost_centers_src[label])
        guidpost_label_src.append(
            np.full(shape=(len(guidpost_pts_src[-1])), fill_value=label))
        guidpost_pts_dst.append(guidpost_centers_dst[label])
        guidpost_label_dst.append(
            np.full(shape=(len(guidpost_pts_dst[-1])), fill_value=label))

    is_use_gss = len(guidpost_pts_dst) > 0 and len(guidpost_pts_dst) > 0
    if not is_use_gss:
        return False, None, None, None, None, None

    # to numpy pts
    guidpost_pts_src = np.vstack(guidpost_pts_src)
    guidpost_label_src = np.concatenate(guidpost_label_src)
    guidpost_pts_dst = np.vstack(guidpost_pts_dst)
    guidpost_label_dst = np.concatenate(guidpost_label_dst)

    indic_list_src = []
    indic_list_dst = []
    for la in label_usefull:
        indic_src = guidpost_label_src == la
        indic_dst = guidpost_label_dst == la
        if not np.any(indic_src):  # 有这种label
            continue

        indic_list_src.append(indic_src)
        indic_list_dst.append(indic_dst)

        if len(indic_list_src) >= min(len(label_usefull), max_num_label): 
            break

    assert len(indic_list_src) == len(indic_list_dst)
    num_label_type = len(indic_list_dst) 
    for idx_indic in range(len(indic_list_src)):
        guidpost_label_src[indic_list_src[idx_indic]] = idx_indic
        guidpost_label_dst[indic_list_dst[idx_indic]] = idx_indic

    return (
        is_use_gss,
        guidpost_pts_src, guidpost_label_src,
        guidpost_pts_dst, guidpost_label_dst, num_label_type
    )


def get_anchor_pts_with_labels_indoor(
    pts_src, label_src,
    pts_dst, label_dst,
    anchor_voxel_size,
    is_vis_guidpost=False,
    is_return_guidpost=False,
    **kwargs

):
    # get_guidpost downsample with label
    gp_pts_src, gp_labels_src = voxel_downsample_with_label(
        pts_src, label_src, voxel_size=anchor_voxel_size)
    gp_pts_dst, gp_labels_dst = voxel_downsample_with_label(
        pts_dst, label_dst, voxel_size=anchor_voxel_size)

    # label affine
    label_unique = np.intersect1d(
        np.unique(gp_labels_src), np.unique(gp_labels_dst))
    label_unique = np.setdiff1d(
        label_unique, np.asarray([0]))  # filter label-0 (noise)

    gp_pts_src_list = []
    gp_labels_src_list = []
    gp_pts_dst_list = []
    gp_labels_dst_list = []

    num_label_type = len(label_unique)

    for idx_label, l in enumerate(label_unique):
        # src
        indic_src = gp_labels_src == idx_label
        gp_pts_src_list.append(gp_pts_src[indic_src])
        gp_labels_src_list.append(gp_labels_src[indic_src])

        # dst
        indic_dst = gp_labels_dst == idx_label
        gp_pts_dst_list.append(gp_pts_dst[indic_dst])
        gp_labels_dst_list.append(gp_labels_dst[indic_dst])

    gp_pts_src = np.vstack(gp_pts_src_list)
    gp_labels_src = np.concatenate(gp_labels_src_list)

    gp_pts_dst = np.vstack(gp_pts_dst_list)
    gp_labels_dst = np.concatenate(gp_labels_dst_list)

    return (
        gp_pts_src, gp_labels_src,
        gp_pts_dst, gp_labels_dst, num_label_type
    )


def grab_gss_with_power_dis(kpts, guidpost_pts, guidpost_label,
                            max_num_label,
                            N,
                            L,
                            num_label_type,
                            **args):
    # alpha = 1.1
    alpha = 1.5
    x = np.arange(N+1)
    inner_rs = x**alpha
    outer_rs = inner_rs[1:]
    inner_rs = inner_rs[:-1]
    max_radiu = outer_rs[-1]

    gss = np.full(shape=(
        len(kpts), N, num_label_type
    ), fill_value=False, dtype=np.ubyte)
    tree = KDTree(guidpost_pts)
    idxs, diss = tree.query_radius(
        kpts, r=max_radiu, return_distance=True)  
    for i in range(len(diss)):  
        # idxs_ring = np.floor(diss[i] / L).astype(np.int32)
        idxs_ring = np.floor(np.exp(np.log(diss[i]) / alpha)).astype(np.int32)

        for idx_r in range(N):
            if not np.any(idxs_ring == idx_r):  
                continue

            # 取出其中要填充的label
            indic = idxs_ring == idx_r
            label_seg = np.unique(
                guidpost_label[idxs[i][indic]])  

            gss[i, idx_r, label_seg] = True

    return gss



def grab_gss(
    kpts, guidpost_pts, guidpost_label,
    max_num_label,
    N,
    L,
    num_label_type,
    **args
):
    inner_rs = np.arange(N, dtype=np.float32) * L  
    max_radiu = inner_rs[-1] + L

    gss = np.full(shape=(
        len(kpts), N, num_label_type
    ), fill_value=False, dtype=np.ubyte)
    tree = KDTree(guidpost_pts)
    idxs, diss = tree.query_radius(
        kpts, r=max_radiu, return_distance=True)  
    for i in range(len(diss)):  
        idxs_ring = np.floor(diss[i] / L).astype(np.int32)

        for idx_r in range(N):
            if not np.any(idxs_ring == idx_r):  
                continue

            indic = idxs_ring == idx_r
            label_seg = np.unique(
                guidpost_label[idxs[i][indic]])  

            gss[i, idx_r, label_seg] = True

    return gss


def grab_gss_out_in_door(pts_src, labels_src, kpts_src,
                         pts_dst, labels_dst, kpts_dst, config):
    if config.dataset.name == "kitti":
        (
            is_use_gss,
            guidpost_pts_src, guidpost_label_src,
            guidpost_pts_dst, guidpost_label_dst, num_label_type
        ) = get_anchor_pts_with_labels_outdoor(pts_src, labels_src, pts_dst, labels_dst, **config.gss)

    elif config.dataset.name == "scannet":
        (
            guidpost_pts_src, guidpost_label_src,
            guidpost_pts_dst, guidpost_label_dst, num_label_type
        ) = get_anchor_pts_with_labels_indoor(pts_src, labels_src, pts_dst, labels_dst, **config.gss)
        is_use_gss = True
    else:
        raise NotImplemented("dataset name error")

    if not is_use_gss:
        return False, None, None
    else:
        gss_src = grab_gss(kpts_src, guidpost_pts_src,
                           guidpost_label_src, **config.gss, num_label_type=num_label_type)
        gss_dst = grab_gss(kpts_dst, guidpost_pts_dst,
                           guidpost_label_dst, **config.gss, num_label_type=num_label_type)

        
        return True, gss_src, gss_dst
