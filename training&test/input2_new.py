import os
import numpy as np
from glob import glob as glb
import matplotlib.pyplot as plt
import torch

import h5py
import math
import time
import csv
import re

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'



def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_flow_new(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1], 2])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def load_boundary_new(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1], 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def flow_train_inputs():
  filenames = glb('./data/traindata/*/')
  shape = [128, 256]
  boun = []
  sflow = []
  sflow_gen = []
  
  for run in filenames:
      flow_name = run + '/fluid_flow_0002.h5'
      boundary_np = load_boundary_new(flow_name, shape).reshape([1, shape[0], shape[1]])
      sflow_true = load_flow_new(flow_name, shape)
      boun.append(boundary_np)
      sflow.append(sflow_true)

  
  boundary_t = torch.FloatTensor(boun)
  sflow_t = torch.FloatTensor(sflow)
  print(sflow_t.shape)
  sflow_true = sflow_t.permute(0,3,1,2)

  return boundary_t, sflow_true

def flow_test_inputs():
  filenames = glb('./data/computed_car_flow_validation/*/')
  shape = [128, 256]
  boun = []
  sflow = []
  sflow_gen = []
  with torch.no_grad():   
      for run in filenames:
          flow_name = run + '/fluid_flow_0002.h5'
          boundary_np = load_boundary(flow_name, shape).reshape([1, shape[0], shape[1]])
          sflow_true = load_flow(flow_name, shape)
          boun.append(boundary_np)
          sflow.append(sflow_true)


      boundary_t = torch.FloatTensor(boun)
      sflow_t = torch.FloatTensor(sflow)
      sflow_true = sflow_t.permute(0,3,1,2)

      return boundary_t, sflow_true

