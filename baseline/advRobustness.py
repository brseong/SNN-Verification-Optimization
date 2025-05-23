import pdb, logging
from time import strftime, localtime

import numpy as np
from snnTrain import Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import spikegen
from multiprocessing import Pool
from mnist import MNIST

from z3 import *
from mnist_net import *
from collections import defaultdict
import functools

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])


log_name = f"{strftime('%m%d%H%M', localtime())}_mnist_baseline_z3_{num_steps}_{'_'.join(str(l) for l in neurons_in_layers)}_delta_{tuple(delta)}.log"
logging.basicConfig(filename=f"../log/{log_name}", level=logging.INFO)
stdout = print
print = lambda x: logging.getLogger().info(x) or stdout(x)

print(f"neurons in layers {neurons_in_layers}, number of steps {num_steps}")

print('Reading Model')
net_dict = torch.load(f'{location}/models/model_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.pth')
net = Net(layers=neurons_in_layers)
net.load_state_dict(net_dict)
print('Model loaded')

print('Loading data')
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=True)
print('data loaded')

tx = time.time()
spike_indicators = {}
for t in range(num_steps):
    for j, m in enumerate(neurons_in_layers):
        if j == 1:
            continue
        else:
            for i in range(m):
                spike_indicators[(i, j, t+1)] = Bool(f'x_{i}_{j}_{t+1}')
print(f"Spikes created in {time.time()-tx} sec")

# tx = time.time()
# data, target = next(iter(test_loader))
# inp = spikegen.rate(data, num_steps=num_steps)
# op = net.forward(inp.view(num_steps, -1))[0]
# label = int(torch.cat(op).sum(dim=0).argmax())
# print(f'single input ran in {time.time()-tx} sec')

samples_list = [41905, 7296, 1639, 48598, 18024, 16049, 14628,
                9144, 48265, 6717, 44348, 48540, 58469, 35741]

def check_sample(sample_no:int):
    tx = time.time()
    data, target = mnist_train[sample_no]
    inp = spikegen.rate(data, num_steps=num_steps)
    
    print(inp.shape)
    op = net.forward(inp.view(num_steps, -1))[0]
    label = int(torch.stack(op, dim=0).sum(dim=0).argmax())
    print(f'single input ran in {time.time()-tx} sec')
    
    # Input property
    tx = time.time()
    s = [[] for _ in range(num_steps)]
    sv = [Int(f's_{i + 1}') for i in range(num_steps)]
    prop = []
    for timestep, spike_train in enumerate(inp):
        for i, spike in enumerate(spike_train.view(neurons_in_layers[0])):
            if spike == 1:
                s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
            else:
                s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
    prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
    prop.append(sum(sv) <= dt)
    # print(prop[0])
    print(f"Inputs Property Done in {time.time() - tx} sec")

    # Output property
    tx = time.time()
    op = []
    intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
    for t in range(neurons_in_layers[-1]):
        if t != label:
            op.append(
                Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
            )
    print(f'Output Property Done in {time.time() - tx} sec')

    tx = time.time()
    S = Solver()
    S.from_file(f'{location}/eqn/eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
    print(f'Network Encoding read in {time.time() - tx} sec')
    S.add(Or(op))
    S.add(prop)
    print(f'Total model ready in {time.time() - tx}')

    print('Query processing starts')
    tx = time.time()
    result = S.check()
    print(f'Checking done in time {time.time() - tx}')
    if result == sat:
        print(f'Not robust for sample {sample_no} and delta={dt}')
    elif result == unsat:
        print(f'Robust for sample {sample_no} and delta={dt}')
    else:
        print(f'Unknown for sample {sample_no} and delta={dt} for reason {S.reason_unknown()}')

# For each delta
for dt in delta:
    with Pool(processes=len(samples_list)) as pool:
        pool.map(check_sample, samples_list)
        pool.close()
        pool.join()

# # For each delta
# for dt in delta:

#     # Input property
#     tx = time.time()
#     s = [[] for i in range(num_steps)]
#     sv = [Int(f's_{i + 1}') for i in range(num_steps)]
#     prop = []
#     for timestep, spike_train in enumerate(inp):
#         for i, spike in enumerate(spike_train.view(neurons_in_layers[0])):
#             if spike == 1:
#                 s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 0.0, 1.0))
#             else:
#                 s[timestep].append(If(spike_indicators[(i, 0, timestep + 1)], 1.0, 0.0))
#     prop = [sv[i] == sum(s[i]) for i in range(num_steps)]
#     prop.append(sum(sv) <= dt)
#     # print(prop[0])
#     print(f"Inputs Property Done in {time.time() - tx} sec")

#     # Output property
#     tx = time.time()
#     op = []
#     intend_sum = sum([2 * spike_indicators[(label, 2, timestep + 1)] for timestep in range(num_steps)])
#     for t in range(neurons_in_layers[-1]):
#         if t != label:
#             op.append(
#                 Not(intend_sum > sum([2 * spike_indicators[(t, 2, timestep + 1)] for timestep in range(num_steps)]))
#             )
#     print(f'Output Property Done in {time.time() - tx} sec')

#     tx = time.time()
#     S = Solver()
#     S.from_file(f'{location}/eqn/eqn_{num_steps}_{"_".join([str(i) for i in neurons_in_layers])}.txt')
#     print(f'Network Encoding read in {time.time() - tx} sec')
#     S.add(op + prop)
#     print(f'Total model ready in {time.time() - tx}')

#     print('Query processing starts')
#     tx = time.time()
#     result = S.check()
#     print(f'Checking done in time {time.time() - tx}')
#     if result == sat:
#         print(f'Not robust for sample and delta={dt}')
#     else:
#         print(f'Robust for sample and delta={dt}')