from skimage.color import rgb2lab, rgb2gray
import numpy as np

def make_colorization_sample(img, hint_density=0.005):
    h,w,c = img.shape

    lab = rgb2lab(img)
    hint_mask = np.random.random([h,w,1])
    hint_mask[hint_mask>hint_density] = 0
    hint_mask[hint_mask != 0] = 1

    sample = {'rgb':img, 'lab':lab, 'hint':hint_mask}
    return sample
