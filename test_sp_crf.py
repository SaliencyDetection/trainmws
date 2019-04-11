import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
from evaluate_sal import fm_and_mae
from skimage.segmentation import slic
from tqdm import tqdm
import cv2


sal_set = 'DUT-train'
img_root = '../data/datasets/saliency_Dataset/%s/images'%sal_set
prob_root = '/home/zeng/mycvmodels/dut-train'
output_root = '/home/zeng/mycvmodels/dut-train-crf'
# prob_root = '/home/crow/data/datasets/saliency_Dataset/results/%s-Sal/Ours_semi1600_woDeeplab'%sal_set
# output_root = '/home/crow/data/datasets/saliency_Dataset/results/%s-Sal/Ours_semi1600_woDeeplab_crf'%sal_set

if not os.path.exists(output_root):
    os.mkdir(output_root)
if not os.path.exists(output_root+'_bin'):
    os.mkdir(output_root+'_bin')

files = os.listdir(prob_root)


# for img_name in files:


def myfunc(img_name):
    img = Image.open(os.path.join(img_root, img_name[:-4]+'.jpg')).convert('RGB')
    W, H = img.size
    img = np.array(img, dtype=np.uint8)
    probs = Image.open(os.path.join(prob_root, img_name[:-4]+'.png'))
    probs = probs.resize((W, H))
    probs = np.array(probs)
    probs = probs.astype(np.float)/255.0

    # superpixel
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
    sp_label = slic(img_lab, n_segments=200, compactness=20)
    # in case of empty superpixels
    sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
    sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
    rs, cs, num = np.where(sp_onehot)
    for i, n in enumerate(num):
        sp_label[rs[i], cs[i]] = n
    sp_num = sp_label.max() + 1
    sp_prob = []
    for i in range(sp_num):
        sp_prob.append(probs[sp_label == i].mean())
    sp_prob = np.array(sp_prob)
    msk = np.zeros(probs.shape)
    for i in range(sp_num):
        msk[sp_label==i] = sp_prob[i]
    probs = msk

    probs = np.concatenate((1-probs[None, ...], probs[None, ...]), 0)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)

    # get unary potentials (neg log probability)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    BMAP = np.argmax(Q, axis=0).reshape((H, W))
    BMAP = (BMAP*255).astype(np.uint8)
    BMAP = Image.fromarray(BMAP)
    BMAP.save(os.path.join(output_root, img_name), 'png')

    MAP = np.array(Q)[1].reshape((H, W))
    MAP = (MAP*255).astype(np.uint8)
    msk = Image.fromarray(MAP)
    msk.save(os.path.join(output_root, img_name), 'png')
    BMAP.save(os.path.join(output_root+'_bin', img_name), 'png')


if __name__ == '__main__':
    # for file in tqdm(files):
    #     myfunc(file)
    print('start crf')
    pool = multiprocessing.Pool(processes=8)
    pool.map(myfunc, files)
    pool.close()
    pool.join()
    print('done')
    fm, mae, _, _ = fm_and_mae(output_root, '../data/datasets/saliency_Dataset/%s/masks'%sal_set)
    print(fm)
    print(mae)

    # hhh='/home/zeng/ROTSfiles/Two3_densenet169/DUT-train_avg_sp_crf'
    # fm, mae, _, _ = fm_and_mae(hhh, '../data/datasets/saliency_Dataset/%s/masks'%sal_set)
    # print(fm)
    # print(mae)
