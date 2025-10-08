import numpy as np
from scipy.optimize import linear_sum_assignment


def maha_dist_matrix(mesur_list, all_tracks, kf):
    '''
    Calculate cost matrix C and gate matrix B btw measurement
    and kalman pred from t-1 for mahalanobis dist
    '''
    C = np.zeros((len(mesur_list), len(all_tracks)))
    B = np.zeros((len(mesur_list), len(all_tracks)))
    for i in range(len(mesur_list)):
        for j in range(len(all_tracks)):
            C[i][j] = kf.mahalanobis_dist(all_tracks[j].mean[-1], all_tracks[j].covariance, mesur_list[i])
            if C[i][j] <= 9.4877:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return C, B


def sift_dist_matrix(des_list, all_tracks, sift, min_score, compare_number):
    '''
    Calculate cost matric C and gate matrix B btw measurement
    and kalman pred from t-1 for sift scores

    Reject descriptors that are less than 10 in number

    For a given measurement, take the maximum score when
    compared with previous 3 time steps descriptors
    '''
    C = np.zeros((len(des_list), len(all_tracks)))
    B = np.zeros((len(des_list), len(all_tracks)))

    for i in range(len(des_list)):
        if des_list[i] is None or des_list[i].shape[0] < 11:
            continue
        for j in range(len(all_tracks)):
            max_val = 0
            for k in all_tracks[j].descriptor[-compare_number:]:
                if k is None:
                    continue
                val = sift.percent_matching(des_list[i], k)
                if val > max_val:
                    max_val = val
            C[i, j] = max_val
            if C[i, j] >= min_score:
                B[i, j] = 1

    return C, B


def matching_assignment(C, B, C2, B2, all_tracks, unmatches, des_list, frame_no, kf):
    """
    Corrected version of matching_assignment.
    Performs Linear Sum Assignment and correctly updates and returns the lists.
    """
    l1 = 0.2
    l2 = 0.8
    C2 = 1 - C2 / 100  # Invert SIFT score to be a cost
    cost = l1 * C + l2 * C2

    # Apply gating matrices to the cost matrix to forbid impossible matches
    cost[B == 0] = 1e6
    cost[B2 == 0] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_indices = set()
    for r, c in zip(row_ind, col_ind):
        # Ensure the match is valid (not one of the forbidden high-cost ones)
        if cost[r, c] < 1e5:
            obj = all_tracks[c]
            obj.measurement.append(unmatches[r])
            obj.descriptor.append(des_list[r])
            obj.frame.append(frame_no)
            obj.status = 'matched'
            obj.reset()  # Reset the unmatched counter

            # Kalman filter correction
            new_m, new_c = kf.update(obj.mean[-1], obj.covariance, obj.measurement[-1])
            obj.mean[-1] = new_m
            obj.covariance = new_c

            matched_indices.add(r)

    # --- FIX: Rebuild the unmatched lists correctly ---
    # Create new lists containing only the items whose indices were NOT matched.
    unmatches_final = [item for i, item in enumerate(unmatches) if i not in matched_indices]
    des_list_final = [item for i, item in enumerate(des_list) if i not in matched_indices]

    return all_tracks, unmatches_final, des_list_final
