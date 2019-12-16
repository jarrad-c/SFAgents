import random
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import retro
import WorkerUtils as wu
from torch.autograd import Variable
from Model import Model
import torch
import torch.nn.functional as F
import imageio
import numpy as np
import os




def prepro(frame, isGrey):
    # frame = frame[32:214, 12:372]  # crop
    frame = frame[::2, ::2]
    if isGrey:
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
    return frame


def setupModel(episode, framesPerStep, loadPath):
    model = Model(framesPerStep, 4, 6)
    if episode > 0:  # For loading a saved model
        model.load_state_dict(torch.load(loadPath + "models/" + str(episode), map_location=lambda storage, loc: storage))
    model.cpu()  # Moves the network matrices to the GPU
    return model

isGrey = True
framesPerStep = 3
loadPath = '../pytorch/saves/'
savePath = 'vid/'
env_id = 'MortalKombat3-Genesis'
# set to model to load
episode = 200
show_viz = False

if not os.path.exists(savePath):
    os.makedirs(savePath)

model = setupModel(episode, framesPerStep, loadPath)

roms_path = "../roms/"  # Replace this with the path to your ROMs
env = retro.make(env_id)

obs = env.reset()
if show_viz:
    fig = plt.figure()
    plt.ion()
    im: AxesImage = plt.imshow(prepro(obs, isGrey), cmap="gray" if isGrey else None)
    plt.axis("off")
    plt.show()
done = False

frames = [obs, obs, obs]
images = [env.render(mode='rgb_array')]

while not done:
    x = wu.prepro(frames)
    moveOut, attackOut = model(Variable(x))
    moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))
    attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))
    action = wu.map_action(moveAction, attackAction)
    
    frames = []
    for j in range(framesPerStep):
        if(j < framesPerStep-1):
            obs, rew, done, info = env.step(action)
        else:
            obs, rew, done, info = env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        frames.append(obs)
        images.append(env.render(mode='rgb_array'))
        if show_viz:
            im.set_data(prepro(obs, isGrey))
            plt.pause(0.00001)
    
imageio.mimsave(savePath + env_id + '_sfagent.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
