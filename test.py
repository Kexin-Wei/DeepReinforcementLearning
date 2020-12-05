import numpy as np

inputs = np.random.rand(5)
w = np.random.rand(4,5)

o = np.dot(inputs,w[1])
print(o)

o2 = np.dot(inputs,np.transpose(w))
print(o2[1])

print("DONE")