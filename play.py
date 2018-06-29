import numpy as np
showim = np.zeros((224, 224, 3), dtype=np.float)
print (showim.shape)
showim = showim.swapaxes(0,2)
print(showim.shape)