import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
import scipy.io as sio
from glob import glob
from tqdm import tqdm

def get_fm(featDir, maskDir, maskThreshold=0.3):
    imList = sorted(glob(os.path.join(featDir, '*.mat')))
    feats, masks = list(), list()
    for im in imList:
        feat = sio.loadmat(im)['output']
        if feat.shape[0] < feat.shape[1]:
            feat = feat.T
        mask = np.array(cv2.imread(os.path.join(maskDir, os.path.basename(im)[:-4]+'.bmp'), cv2.IMREAD_GRAYSCALE).T, dtype=bool)
        assert feat.shape == mask.shape
        assert feat.shape[0] > feat.shape[1]
        mask[abs(feat-feat.mean())<maskThreshold]=False
        feat = feat > feat.mean()
        feats.append(feat)
        masks.append(mask)
    return np.dstack(feats), np.dstack(masks)

def rotate_map(featmap:np.array, shift:int=16):
    feats=list()
    for i in range(featmap.shape[2]):
        feattemp = list()
        for sf in range(-shift, shift+1):
            feattemp.append(np.roll(featmap[:,:,i], sf, axis = 0).ravel()) 
        feats.append(np.vstack(feattemp))
    return np.dstack(feats)

def repeat_map(featmap:np.array, shift:int=16):
    feats=list()
    for i in range(featmap.shape[2]):
        feattemp = list()
        for sf in range(-shift, shift+1):
            feattemp.append(featmap[:,:,i].ravel()) 
        feats.append(np.vstack(feattemp))
    return np.dstack(feats)

def get_label(gallery, probe, type, lr=-1):
    #TODO(KUO): add the left right difference.
    gList = sorted(glob(os.path.join(gallery, '*.'+type)))
    pList = sorted(glob(os.path.join(probe, '*.'+type)))
    labelG, labelP = list(), list()
    labelMat = np.zeros((len(gList), len(pList)))
    for g in gList:
        labelG.append(os.path.basename(g).split('_')[0])
    for p in pList:
        labelP.append(os.path.basename(p).split('_')[0])
    for ig, gg in enumerate(labelG):
        for ip, pp in enumerate(labelP):
            labelMat[ig, ip] = 1 if gg==pp else -1
    return labelG, labelP, labelMat


def cal_hamming(featDir_G, featDir_P, maskDir_G, maskDir_P, batch=200):
    assert torch.cuda.is_available()
    featG, maskG = get_fm(featDir_G, maskDir_G)
    featP, maskP = get_fm(featDir_P, maskDir_P)
    featG = rotate_map(featG)
    maskG = rotate_map(maskG)
    featP = repeat_map(featP)
    maskP = repeat_map(maskP)
    scores = torch.zeros((featG.shape[2], featP.shape[2])).cuda()
    mrates = torch.zeros((featG.shape[2], featP.shape[2])).cuda()

    for ga in tqdm(range(featG.shape[2]//batch+1)):
        for pr in range(featP.shape[2]//batch+1):
            featGT = torch.from_numpy(featG[...,ga*batch:min((ga+1)*batch, featG.shape[2])]).cuda() # gallery feat tensor
            featPT = torch.from_numpy(featP[...,pr*batch:min((pr+1)*batch, featP.shape[2])]).cuda() # probe feat tensor           
            maskGT = torch.from_numpy(maskG[...,ga*batch:min((ga+1)*batch, maskG.shape[2])]).cuda() # gallery mask tensor
            maskPT = torch.from_numpy(maskP[...,pr*batch:min((pr+1)*batch, maskP.shape[2])]).cuda() # probe mask tensor

            assert featGT.shape == maskGT.shape
            assert featPT.shape == maskPT.shape

            if featGT.shape[2] == 0 or featPT.shape[2] == 0:
                continue
            if featPT.shape[2] > featGT.shape[2]:
                scoresTemp = torch.zeros((featGT.shape[2], featPT.shape[2]))
                mratesTemp = torch.zeros((featGT.shape[2], featPT.shape[2]))
                for i in range(featPT.shape[2]):
                    tempMask = torch.bitwise_and(torch.roll(maskPT, -i, 2)[...,:maskGT.shape[2]], maskGT)
                    tempDist = torch.bitwise_and(torch.bitwise_xor(featGT, torch.roll(featPT,-i,2)[...,:featGT.shape[2]]), tempMask)
                    tempMsum = torch.sum(tempMask, 1)
                    tempScore = torch.min(torch.sum(tempDist, 1)/tempMsum, 0, True)
                    scoresTemp[:,i] = tempScore.values.squeeze()
                    mratesTemp[:,i] = torch.gather(tempMsum, 0, tempScore.indices).squeeze()
                for j in range(len(scoresTemp)):
                    scoresTemp[j,:] = torch.roll(scoresTemp[j,:], j)
                    mratesTemp[j,:] = torch.roll(mratesTemp[j,:], j)
                scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min((pr+1)*batch, featP.shape[2])] = scoresTemp
                mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min((pr+1)*batch, featP.shape[2])] = mratesTemp
            else:
                scoresTemp = torch.zeros((featPT.shape[2], featGT.shape[2])).cuda()
                mratesTemp = torch.zeros((featPT.shape[2], featGT.shape[2])).cuda()
                for i in range(maskGT.shape[2]):
                    tempMask = torch.bitwise_and(torch.roll(maskGT, -i, 2)[...,:maskPT.shape[2]], maskPT)
                    tempDist = torch.bitwise_and(torch.bitwise_xor(featPT, torch.roll(featGT,-i,2)[...,:featPT.shape[2]]), tempMask)
                    tempMsum = torch.sum(tempMask, 1)
                    tempScore = torch.min(torch.sum(tempDist, 1)/tempMsum, 0, True)
                    scoresTemp[:,i] = tempScore.values.squeeze()
                    mratesTemp[:,i] = torch.gather(tempMsum, 0, tempScore.indices).squeeze()
                for j in range(len(scoresTemp)):
                    scoresTemp[j,:] = torch.roll(scoresTemp[j,:], j)
                    mratesTemp[j,:] = torch.roll(mratesTemp[j,:], j)
                scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min((pr+1)*batch, featP.shape[2])] = scoresTemp.transpose(0,1)
                mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min((pr+1)*batch, featP.shape[2])] = mratesTemp.transpose(0,1)

    return scores.cpu().numpy(), mrates.cpu().numpy()/featG.shape[1]