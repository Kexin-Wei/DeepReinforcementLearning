import os
import imageio
import shutil
import matplotlib.pyplot as plt


def gif_save(DIR,DIRNAME,ep,reward):
    # make gif at dir with ep and reward
    # with png from dir/dirname
    # delete dir/dirname

    DIR_PNG = os.path.join(DIR,DIRNAME)

    # make gif
    images = []
    for f in os.listdir(DIR_PNG):
        images.append(imageio.imread(os.path.join(DIR_PNG,f)))
    imageio.mimsave(os.path.join(DIR,str(ep)+'_r_'+str(reward)+'.gif'),images)

    shutil.rmtree(DIR_PNG)


def png_save(DIR_PNG,env,step):
    # save env render result as png at DIR_PNG
    plt.imsave(os.path.join(DIR_PNG,str(step)+'.png'),env.render(mode='rgb_array'))