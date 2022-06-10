# (KUO) 2022/Jun/8
import numpy as np
import matplotlib.pyplot as plt
import os
def cal_far_frr(garr, iarr, resolu=2500):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        far[i] = np.sum(iarr < d) / iarr.size
        frr[i] = np.sum(garr > d) / garr.size
    return far, frr

def cal_eer_tar(garr, iarr):
    far, frr = cal_far_frr(garr, iarr)
    result=dict()
    result['eer'] = far[np.argmin(np.abs(far - frr))]
    result['tar03'] = 1 - frr[np.argmin(np.abs(far - 0.001))]
    result['tar04'] = 1- frr[np.argmin(np.abs(far - 0.0001))]
    return result

def plot_roc(garr, iarr, saveDir):
    euc_far, euc_frr = cal_far_frr(garr, iarr)
    euc_result = cal_eer_tar(garr, iarr) 
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.title("Euclidean Distance ROC Curve")
    plt.xlabel("False Accept Rate")
    plt.ylabel("Genuine Accept Rate")
    plt.xlim([1e-4, 1])
    plt.ylim([0, 1.0])
    plt.plot(euc_far, 1 - euc_frr, label="Euclidean")
    plt.legend()
    if not os.path.dirname(saveDir):
        os.makedirs(os.path.dirname(saveDir))
    plt.savefig(saveDir)
    print(f"EER is {euc_result['eer']} and GAR is {euc_result['tar03']} and {euc_result['tar04']}")
    return euc_result
