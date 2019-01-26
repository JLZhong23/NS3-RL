#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from ns3gym import ns3env

import matplotlib as mpl
import matplotlib.pyplot as plt

from tcp_base import TcpTimeBased, TcpTimeDQLearning
from tcp_newreno import TcpNewReno

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
parser.add_argument('--episodes',
                    type=int,
                    default=200,
                    help='Number of episodes, Default: 200')
parser.add_argument('--lr',
                    type=int,
                    default=0.95,
                    help='Learning rate, Default: 0.95')
parser.add_argument('--exr',
                    type=int,
                    default=0.1,
                    help='Exploration rate, Default: 0.1')
parser.add_argument('--max_env_step',
                    type=int,
                    default=4000,
                    help='max env step, Default: 4000')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10 # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = False
lr = args.lr
episodes = args.episodes
exr = args.exr
max_env_step = args.max_env_step


env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

ssThresh_all = []
cWnd_all = []
segmentSize_all = []
bytesInFlightSum_all = []
bytesInFlightAvg_all = []
segmentsAckedSum_all = []
segmentsAckedAvg_all = []
avgRtt_all = []
minRtt_all = []
avgInterTx_all = []
avgInterRx_all = []
throughput_all = []

def draw_pic(data, label, x_label, y_label, save_path):
    mpl.rcdefaults()
    mpl.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.grid(True, linestyle='--')
    plt.title('Learning Performance')
    plt.plot(range(len(data)), data, label=label, marker="", linestyle="-")  # , color='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(prop={'size': 12})

    plt.savefig(save_path, bbox_inches='tight')

def utilites(throughput, rtt):
    return 0.01 * np.log(throughput) + 0.001 * np.log(rtt)


def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            # time-based = 1
            tcpAgent = TcpTimeDQLearning(ob_space)
            #tcpAgent = TcpNewReno()
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent

# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space


try:
    while True:
        print("Start iteration: ", currIt)
        obs = env.reset()
        reward = 0
        done = False
        info = None
        print("Step: ", stepIdx)
        print("---obs: ", obs)

        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs)

        while True:
            if stepIdx % 1 == 0:
                ssThresh_all.append(obs[4])
                cWnd_all.append(obs[5])
                segmentSize_all.append(obs[6])
                bytesInFlightSum_all.append(obs[7])
                bytesInFlightAvg_all.append(obs[8])
                segmentsAckedSum_all.append(obs[9])
                segmentsAckedAvg_all.append(obs[10])
                avgRtt_all.append(obs[11])
                minRtt_all.append(obs[12])
                avgInterTx_all.append(obs[13])
                avgInterRx_all.append(obs[14])
                throughput_all.append(obs[15])

            stepIdx += 1
            action = tcpAgent.get_action(obs, reward, done, info)
            print("---action: ", action)

            print("Step: ", stepIdx)
            next_obs, reward, done, info = env.step(action)
            print("---obs, reward, done, info: ", next_obs, reward, done, info)

            if utilites(next_obs[15], next_obs[11]) > utilites(obs[15], obs[11]):
                reward = 10
            else:
                reward = -20

            target = reward
            if not done:
                target = (reward + 0.95 * np.amax(tcpAgent.model.predict(np.reshape(next_obs, [1, ob_space.shape[0]]))[0]))

            tmp_action = 2
            if obs[5] > next_obs[5]:
                tmp_action = 1
            elif obs[5] < next_obs[5]:
                tmp_action = 0
            else:
                tmp_action = 2
            tcpAgent.fit(obs, target, tmp_action)

            # get existing agent of create new TCP agent if needed
            tcpAgent = get_agent(obs)

            obs = next_obs

            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    draw_pic(ssThresh_all, 'ssThresh', 'step*100', 'byte', 'ssTresh.pdf')
    draw_pic(cWnd_all, 'cWnd', 'step*100', 'byte', 'cWnd.pdf')
    draw_pic(segmentSize_all, 'segmentSize', 'step*100', 'byte', 'segmentSize.pdf')
    draw_pic(bytesInFlightSum_all, 'bytesInFlightSum', 'step*100', 'byte', 'bytesInFlightSum.pdf')
    draw_pic(bytesInFlightAvg_all, 'bytesInFlightAvg', 'step*100', 'byte', 'bytesInFlightAvg.pdf')
    draw_pic(segmentsAckedSum_all, 'segmentsAckedSum', 'step*100', 'byte', 'segmentsAckedSum.pdf')
    draw_pic(segmentsAckedAvg_all, 'segmentsAckedAvg', 'step*100', 'byte', 'segmentsAckedAvg.pdf')
    draw_pic(avgRtt_all, 'avgRtt', 'step*100', 'byte', 'avgRtt.pdf')
    draw_pic(minRtt_all, 'minRtt', 'step*100', 'byte', 'minRtt.pdf')
    draw_pic(avgInterTx_all, 'avgInterTx', 'step*100', 'byte', 'avgInterTx.pdf')
    draw_pic(avgInterRx_all, 'avgInterRx', 'step*100', 'byte', 'avgInterRx.pdf')
    draw_pic(throughput_all, 'throughput', 'step*100', 'byte', 'throughput.pdf')
    print("Ctrl-C -> Exit")

finally:
    env.close()
    print("Done")