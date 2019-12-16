import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch.WorkerUtils as wu
import retro
import traceback
import logging

logging.basicConfig(filename='logs/pytorch_stats.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# The worker class for running agent vs Computer training, aka story mode training
class Worker(mp.Process):

    def __init__(self, env_id, roms_path, epoch_size, model, optim, criterion, rewardQueue, frameRatio, framesPerStep):
        super(Worker, self).__init__()
        self.env_id = env_id
        self.roms_path = roms_path
        self.epoch_size = epoch_size
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.rewardQueue = rewardQueue
        self.frameRatio = frameRatio
        self.framesPerStep = framesPerStep

    def run(self):
        try:
            logger.info("Starting Worker")
            self.env = retro.make(self.env_id, record='rec/')
            initial_obs = self.env.reset()
            while True:
                self.model.eval()

                observations, histories, frames = self.generate_playthrough(initial_obs)

                self.model.train()

                dataset = wu.compileHistories(observations, histories)
                wu.train(self.model, self.optim, self.criterion, dataset)

        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())

    def map_action(moveAction, attackAction):
        move_act_idxs = [4, 5, 6, 7]
        attack_act_idxs = [0, 1, 8, 9, 10, 11]
        action_multi_binary = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_multi_binary[move_act_idxs[moveAction]] = 1
        action_multi_binary[attack_act_idxs[attackAction]] = 1

        return action_multi_binary

    def generate_playthrough(self, initial_obs):
        observations = [[]]
        histories = [{"moveAction": [], "attackAction": [], "reward": []}]
        epoch_reward = 0
        done = False
        frames = []

        for i in range(self.epoch_size):
            for k in range(framesPerStep):
                frames.append(initial_obs)

            while not done:
                x = wu.prepro(frames)

                observations[total_round].append(x.cpu())

                moveOut, attackOut = self.model(Variable(x))
                moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))
                attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))

                histories[total_round]["moveAction"].append(torch.FloatTensor(1).fill_(moveAction))
                histories[total_round]["attackAction"].append(torch.FloatTensor(1).fill_(attackAction))

                frames = []
                action = map_action(moveAction, attackAction)
                for j in range(framesPerStep):
                    if(j < framesPerStep-1):
                        obs, rew, done, info = self.env.step(action)
                    else:
                        obs, rew, done, info = self.env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    frames.append(obs)

                histories[total_round]["reward"].append(torch.FloatTensor(1).fill_(rew))

                epoch_reward += rew

            histories.append({"moveAction": [], "attackAction": [], "reward": []})
            observations.append([])
            self.rewardQueue.put({"reward": epoch_reward})
            initial_obs = self.env.reset()

        return observations, histories, frames
