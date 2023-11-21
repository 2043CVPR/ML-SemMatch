import numpy as np


def get_topx_same_consis_matrix(same_matrix, topx):
    N_dst = same_matrix.shape[1]
    top_l2 = np.max(same_matrix, axis=1).astype(np.int32) - (topx - 1)
    # top_l2 = top_l1  # NOTE: base one the grade is continue with high probbility
    top_l2_broad = np.tile(top_l2.reshape(-1, 1), reps=(1, N_dst))
    global_consis = same_matrix >= top_l2_broad
    return global_consis


def get_same_matrix(gss_src, gss_dst):
    same_matrix = np.sum(np.bitwise_and(
        gss_src[:, None, ...], gss_dst[None, ...]), axis=(-1, -2))

    return same_matrix


def get_nn_select_raw_material(desc_src, desc_dst,
                               gss_src, gss_dst,
                               topx):
    # global match
    same_matrix = get_same_matrix(gss_src, gss_dst)
    # NN match
    cross_desc_dis = np.linalg.norm(
        desc_src[:, None] - desc_dst[None, :], axis=-1)  # TODO: not all calc dis
    grades = np.exp(-cross_desc_dis)

    return same_matrix, grades


def _matcher_nn_mm(global_consis, grades):
    N_src = global_consis.shape[0]

    mask_grades = grades
    # grade set to zeros for not satisfy gss consistency
    mask_grades[np.logical_not(global_consis)] = 0

    corres = np.tile(np.arange(N_src).reshape(-1, 1), reps=(1, 2))
    corres[:, 1] = np.argmax(mask_grades, axis=1)

    return corres


def maskmatcher_nn(
    desc_src, desc_dst,
    gss_src, gss_dst,
    topx,
    **kwargs
):
    # global match
    same_matrix, grades = get_nn_select_raw_material(desc_src, desc_dst,
                                                     gss_src, gss_dst,
                                                     topx)

    global_consis = get_topx_same_consis_matrix(same_matrix, topx=topx)
    corres = _matcher_nn_mm(global_consis, grades)

    return corres
