from pytorch.Worker import Worker
from pytorch.Model import Model
from pytorch.Statistics import Statistics
import pytorch.WorkerUtils as wu
from torch.autograd import Variable
import torch.nn.functional as F

import retro
import torch
import torch.optim as opt
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import logging

logging.basicConfig(filename='logs/pytorch_stats.log', level=logging.INFO)


# Creates the model used for training
def setupModel(learning_rate, episode, framesPerStep, loadPath):
    model = Model(framesPerStep, 4, 6)
    optim = opt.Adam(model.parameters(), lr=learning_rate)
    if episode > 0:  # For loading a saved model
        model.load_state_dict(torch.load(loadPath + "models/" + str(episode), map_location=lambda storage, loc: storage))
        optim.load_state_dict(torch.load(loadPath + "optims/" + str(episode)))
    #model.cuda()  # Moves the network matrices to the GPU
    model.share_memory()  # For multiprocessing
    return model, optim


def setupWorkers(roms_path, epoch_size, learning_rate, frameRatio, framesPerStep, episode, workerCount, loadPath):
    env_ids = ['MortalKombat3-Genesis', 'StreetFighterIISpecialChampionEdition-Genesis']
    model, optim = setupModel(learning_rate, episode, framesPerStep, loadPath)
    criterion = nn.CrossEntropyLoss(reduce=False)

    rewardQ = Queue()

    workers = [Worker(env_ids[i], roms_path, epoch_size, model, optim, criterion, rewardQ, frameRatio, framesPerStep) for i in range(workerCount)]
    return workers, model, optim, rewardQ


# Check pytorch_stats.log for results
def simulate(episode, workers, model, optim, rewardQueue, batch_save, path):
    [w.start() for w in workers]

    stats = Statistics(episode)

    while True:
        episode += 1
        if episode % batch_save == 0:
            print("Saving", episode)
            torch.save(model.state_dict(), path + "models/" + str(episode))
            torch.save(optim.state_dict(), path + "optims/" + str(episode))
        if not rewardQueue.empty():
            reward = rewardQueue.get()
            print("Got rewardque")
            stats.update(reward)


def evaluate(framesPerStep, loadPath, learning_rate, episode):
    model = Model(framesPerStep, 4, 6)
    optim = opt.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load(loadPath + "models/" + str(episode), map_location=lambda storage, loc: storage))
    optim.load_state_dict(torch.load(loadPath + "optims/" + str(episode)))
    done = False
    frames = []

    ## run the model
    env = retro.make(game='MortalKombat3-Genesis')
    initial_obs = env.reset()
    
    for k in range(framesPerStep):
        frames.append(initial_obs)

    while not done:
        x = wu.prepro(frames)

        moveOut, attackOut = model(Variable(x))
        moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))
        attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))

        frames = []
        action = Worker.map_action(moveAction, attackAction)
        for j in range(framesPerStep):
            if(j < framesPerStep-1):
                obs, rew, done, info = env.step(action)
            else:
                env.render()
                obs, rew, done, info = env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            frames.append(obs)

    env.close()


# spawn must be called inside main
if __name__ == '__main__':
    mp.set_start_method('spawn')

    roms_path = "../roms"
    frameRatio = 2
    framesPerStep = 3
    learning_rate = 5e-5
    episode = 0  # Change episode to load from presaved model, check saves for saves
    epoch_size = 1  # How many episodes before training
    batch_save = 100  # How many results before saving the current model and optimiser
    workerCount = 2  # How many workers to train the model in parallel
    difficulty = 3  # Story mode difficulty
    loadPath = "saves/"

    workers, model, optim, rewardQueue = setupWorkers(roms_path, difficulty, epoch_size, learning_rate, frameRatio, framesPerStep, episode, workerCount, loadPath)

    simulate(episode, workers, model, optim, rewardQueue, batch_save, loadPath)
