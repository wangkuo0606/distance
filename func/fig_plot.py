# (KUO) 2022/Jun/8
import numpy as np
import matplotlib.pyplot as plt
def cal_far_frr(garr, iarr, resolu=2000):
    d_max = max(garr.max(), iarr.max())
    d_min = min(garr.min(), iarr.min())
    far = np.empty(resolu)
    frr = np.empty(resolu)
    for i, d in enumerate(np.linspace(d_min, d_max, resolu)):
        far[i] = np.sum(iarr < d) / iarr.size
        frr[i] = np.sum(garr > d) / garr.size
    return far, frr

def cal_eer(garr, iarr):
    far, frr = cal_far_frr(garr, iarr)
    return far[np.argmin(np.abs(far - frr))]

def plot_roc(garr, iarr, savefig=True):
    euc_far, euc_frr = cal_far_frr(garr, iarr)
    euc_eer = cal_eer(garr, iarr) 
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.title("Euclidean Distance ROC Curve")
    plt.xlabel("False Accept Rate")
    plt.ylabel("Genuine Accept Rate")
    plt.xlim([1e-4, 1])
    plt.ylim([0, 1.0])
    plt.plot(euc_far, 1 - euc_frr, label="Euclidean")
    plt.legend()
    plt.savefig('ROC.png')
    print(euc_eer)
