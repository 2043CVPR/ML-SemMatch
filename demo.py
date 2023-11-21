import open3d as o3d
import numpy as np
from easydict import EasyDict

from SemMatch.group_match import query_local_signature, GroupingMatcher 
from SemMatch.gss_constructor import grab_gss, get_anchor_pts_with_labels_outdoor
from SemMatch.mask_matcher import maskmatcher_nn

from utils.evaluator import PCRResultEvaluator

evaluator = PCRResultEvaluator(5, 0.6, 0.6)


def load_data():
    prefix = "000000"
    names = ["pts_src", "pts_dst", "kpts_src", "kpts_dst", "fpfh_src", "fpfh_dst", "gedi_src", "gedi_dst", "label_src", "label_dst", "trans"]
    pts_src, pts_dst, kpts_src, kpts_dst, fpfh_src, fpfh_dst, gedi_src, gedi_dst, label_src, label_dst, trans_gt = [np.load("data/{}_{}.npy".format(prefix, name), allow_pickle=True) for name in names]
    return (pts_src, pts_dst, kpts_src, kpts_dst, fpfh_src, fpfh_dst, gedi_src, gedi_dst, label_src, label_dst, trans_gt)


def ml_semmatch(pts_src, pts_dst, label_src, label_dst, kpts_src, kpts_dst, desc_src, desc_dst, trans_gt, config_lsig, config_gss, config_mask_match):
    evaluator.update_pair(pts_src, pts_dst, trans_gt=trans_gt)
    # 1. NSS
    nss_src = query_local_signature(pts_src, label_src, kpts_src, radiu=config_lsig.radiu_local_sig)
    nss_dst = query_local_signature(pts_dst, label_dst, kpts_dst, radiu=config_lsig.radiu_local_sig)

    # 2. SCM er
    scmer = GroupingMatcher(nss_src, nss_dst, config_lsig=config_lsig)

    # 3. BMR-SS
    (
        is_use_gss,
        guidpost_pts_src, guidpost_label_src,
        guidpost_pts_dst, guidpost_label_dst, num_label_type
    ) = get_anchor_pts_with_labels_outdoor(pts_src, label_src, pts_dst, label_dst, **config_gss)

    gss_src = grab_gss(kpts_src, guidpost_pts_src,
                        guidpost_label_src, **config_gss, num_label_type=num_label_type)
    gss_dst = grab_gss(kpts_dst, guidpost_pts_dst,
                        guidpost_label_dst, **config_gss, num_label_type=num_label_type)

    # 4. match
    corres = scmer.run_match_based_on_gss(maskmatcher_nn, desc_src, desc_dst, gss_src, gss_dst, **config_mask_match)

    # evaluation
    kpts_src = kpts_src[corres[:, 0]]
    kpts_dst = kpts_dst[corres[:, 1]]

    evaluator.eval_corr(kpts_src, kpts_dst, is_print=True)

    # reg
    corr = np.tile(np.arange(1, len(kpts_src))[:, None], reps=(1, 2))
    rslt = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kpts_src)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kpts_dst)),
        o3d.utility.Vector2iVector(corr),
        max_correspondence_distance=0.8,  
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            1000 * 1000 * 10,
            0.99
        )
    )
    # print(rslt.transformation)

    # vis
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pts_src)
    pcd_src.paint_uniform_color([1, 0.706, 0])
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(pts_dst)
    pcd_dst.paint_uniform_color([0, 0.651, 0.929])


    o3d.visualization.draw_geometries([pcd_src, pcd_dst])

    pcd_src.transform(rslt.transformation)
    o3d.visualization.draw_geometries([pcd_src, pcd_dst])

    # eval
    eval = evaluator.eval_trans(trans=rslt.transformation)[3]
    print(eval)

def main():
    # load data 
    data = load_data()
    (pts_src, pts_dst, kpts_src, kpts_dst, fpfh_src, fpfh_dst, gedi_src, gedi_dst, label_src, label_dst, trans_gt) = data

    # config
    config_lsig = EasyDict()
    config_lsig.radiu_local_sig = 0.8
    config_lsig.label_unuse = np.asarray([30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])
    config_lsig.label_undefine = np.asarray([0])

    config_gss = EasyDict()
    config_gss.label_usefull = [16, 18, 80, 81, 10]
    config_gss.max_num_label = 8
    config_gss.N = 33
    config_gss.L = 1.5

    config_mask_match = EasyDict()
    config_mask_match.topx = 2

    # match 
    ml_semmatch(pts_src, pts_dst, label_src, label_dst, kpts_src, kpts_dst, 
                        fpfh_src, fpfh_dst,
                        # gedi_src, gedi_dst,
                        trans_gt, config_lsig, config_gss, config_mask_match)


if __name__=="__main__":
    main()


"""
python -m demo
"""