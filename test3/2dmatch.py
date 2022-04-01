

import numpy as np

cam1_coords = np.array([742,883,1])

cam2_coords = np.array([491, 969,1])


Tcam1_cam2 =np.array([[ 0.6782608 , -1.0919471 , 0.00999542],
[ 0.43418501 , 0.67941068, -0.01816987],
[ 0.05765369 , 0.1860458 ,  0.81927659]])

res = Tcam1_cam2 @ cam2_coords
print(res)