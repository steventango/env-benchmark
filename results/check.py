#%%

import numpy as np

forager = np.load('forager.npy')

jbw = np.load('jbw.npy')

# %%

forager[:, :, 0].sum() / 65 **2 , forager[:, :, 1].sum() / 65**2

# %%
# (jbw[:, :] == [.82, .27, .2]).sum()
# get count of elements in matri
# x that are equal to [.82, .27, .2]

np.count_nonzero(np.sum(jbw, axis=2) == 1.2900001) / 65**2
# %%
np.count_nonzero(np.sum(jbw, axis=2) == 1.6800001) / 65**2
# %%
65*65*.1
# %%
