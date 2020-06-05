import matplotlib.pyplot as plt
import h5py
import numpy as np 
from glob import glob as glb
# from train import ResNet,ResidualBlock
#from train import *
import torch
import os
import sys
import time
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

device = torch.device('cuda')

def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def main():
    
    model1 = torch.load(sys.argv[1],map_location=lambda storage, loc: storage).to(device)
    model1.eval()

    # print(model)
    filenames = glb('./data/computed_car_flow/*/')
    shape = [128, 256]
    boun = []
    sflow = []
    sflow_gen = []
    with torch.no_grad():
        fig_num = 0 
        for run in filenames:
            if fig_num > 15 :
                break
            fig_num = fig_num + 1
            flow_name = run + '/fluid_flow_0002.h5'
            boundary_np = load_boundary(flow_name, shape).reshape([1, shape[0], shape[1]])
            sflow_true = load_flow(flow_name, shape)
            boun.append(boundary_np)
            sflow.append(sflow_true)


        boundary_t = torch.FloatTensor(boun).to(device)
        sflow_t = torch.FloatTensor(sflow)
        sflow_true = sflow_t.permute(0,3,1,2).numpy()
#             print(type(sflow_true))
#             sflow_generated = model(boundary_t)[0].cpu().numpy()
        
        # warmup
        for i in range(10):
            temp = model1(boundary_t)
        
        start = time.time()
        for i in range(100):
            temp = model1(boundary_t)
        end = time.time()
        running_time = (end - start)*1000/100
        print('model1 time cost : %.5f ms' %running_time)

        # warmup
        #for i in range(10):
        #    temp = model2(boundary_t)


if __name__ == '__main__':
    main()
