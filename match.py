import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
from func.cal_dist import cal_hamming, get_label
from func.fig_plot import plot_roc
featDirG = './Data/PolyU_2S/Gallery/Feat'
featDirP = './Data/PolyU_2S/Probe/Feat'
maskDirG = './Data/PolyU_2S/Gallery/Mask'
maskDirP = './Data/PolyU_2S/Probe/Mask'
resultDir = './Result/PolyU_2S'
if not os.path.isdir(resultDir):
    os.makedirs(resultDir+'/score')
    os.makedirs(resultDir+'/figure')
iters = os.listdir(featDirG)
lg, lp, labelMat = get_label(maskDirG, maskDirP, 'bmp')
results = dict()
for it in tqdm(iters):
    featDirGT = os.path.join(featDirG, it)
    featDirPT = os.path.join(featDirP, it)
    if 'G1D' in it:
        maskDirGT = maskDirG+'1D'
        maskDirPT = maskDirP+'1D'
    else:
        maskDirGT = maskDirG
        maskDirPT = maskDirP
    scores, mrates = cal_hamming(featDirGT, featDirPT, maskDirGT, maskDirPT, 100, 2)
    sio.savemat(os.path.join(resultDir, 'score', it+'_scores.mat'), {'scores':scores, 'labelMat':labelMat, 'mrates':mrates})
    gg = np.array(scores[labelMat==1])
    ii = np.array(scores[labelMat==-1])
    result = plot_roc(gg, ii, os.path.join(resultDir, 'figure', it+'_roc.png'))
    results[it] = result
sio.savemat(os.path.join(resultDir, 'result.mat'), {'results':results})
