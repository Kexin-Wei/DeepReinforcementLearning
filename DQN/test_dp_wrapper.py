# %%
import os
import imageio
import shutil
import numpy as np
import matplotlib.pyplot as plt

from dm_wrapper import make_env

# %%
env = make_env("BreakoutNoFrameskip-v4")

# %%

ROOT_DIR = "Test_Wrapper"
DIR_PNG  = os.path.join(ROOT_DIR,"png")
try:
    os.mkdir(ROOT_DIR)
except:
    pass
try:
    os.mkdir(DIR_PNG)
except:
    pass

# %%

ob = env.reset()
images = []

for i in range(100):
    images.append(env.render(mode='rgb_array'))
    act = env.action_space.sample()
    ob_next, reward, done, info = env.step(act)
    
    plt.imsave(os.path.join(DIR_PNG,str(i)+".png"),np.concatenate(ob_next,axis=1))
    
    if done:
        break    
imageio.mimsave("Test_Wrapper/test.gif",images,fps=40)
# %%
images = []
for f in os.listdir(DIR_PNG):
    images.append(imageio.imread(os.path.join(DIR_PNG,f)))
imageio.mimsave(os.path.join(ROOT_DIR,'4in1.gif'),images,fps=50)
shutil.rmtree(DIR_PNG)
# %%
#print(ob_next._frames[0].shape,act,reward,done,info)
print(ob_next[0].shape)
print(ob_next._force().shape)
print(len(ob_next))

# %%
for i in range(len(ob_next)):
    plt.subplot(1,4,i+1)
    plt.imshow(ob_next[i],cmap='gray')
    plt.axis('off')
plt.show()  
# %%
print(np.concatenate(ob_next,axis=1).shape)
plt.imshow(np.concatenate(ob_next,axis=1))
# %%
