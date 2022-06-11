import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
from func.cal_dist import cal_hamming, get_label
from func.fig_plot import plot_roc
featDir = './Data/CASIA/Feat'
maskDir = './Data/CASIA/Mask'
resultDir = './Result/CASIA'
if not os.path.isdir(resultDir):
    os.makedirs(resultDir)
iters = os.listdir(featDir)
lg, lp, labelMat = get_label(maskDir, maskDir, 'bmp')
labelMat[np.eye(len(lg),dtype=np.bool)] = 0 
results = dict()
for it in tqdm(iters):
    featDirT = os.path.join(featDir, it)
    scores, mrates = cal_hamming(featDirT, featDirT, maskDir, maskDir)
    sio.savemat(os.path.join(resultDir, it+'_scores.mat'), {'scores':scores, 'labelMat':labelMat, 'mrates':mrates})
    gg = np.array(scores[labelMat==1])
    ii = np.array(scores[labelMat==-1])
    result = plot_roc(gg, ii, os.path.join(resultDir, it+'_roc.png'))
    results[it] = result
sio.savemat('result.mat', {'results':results})
