import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

x = pywt.data.camera().astype(np.float32)
shape = x.shape

level = 2

# compute the 2D DWT
c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
# normalize each coefficient array independently for better visibility
print(type(c))
c[0] /= np.abs(c[0]).max()
for detail_level in range(level):
    c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
# show the normalized coefficients
arr, slices = pywt.coeffs_to_array(c)
plt.imshow(arr, cmap=plt.cm.gray)
# plt.set_title('Coefficients\n({} level)'.format(level))
# plt.set_axis_off()
l0, l1, l2 = c
print(l0.shape)
h1, v1, d1 = l1
h2, v2, d2 = l2
print(h1.shape, v1.shape, d1.shape)
print(h2.shape, v2.shape, d2.shape)