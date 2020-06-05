import matplotlib.pyplot as plt
import h5py
import numpy as np 
from glob import glob as glb
#from train import *
#from DfpNet import *
#from DfpNet_own import TurbNetG
from LossFunc_mass_all import MyLoss as loss_mass
import torch
import os
import time
import sys
import seaborn as sns
from sectional_drawing import painting

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

device = torch.device('cpu')

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

def main():
   

    lossPer_v_accum1 = 0
    lossPer_v_accum2 = 0 

    lossphy_v_accum1 = 0
    lossphy_v_accum2 = 0

    err1 = []
    err2 = []

# model_test_xin_unet
# model_test_encoder_share
# model_test_new_baseline_L1
# model_test_xin_phy
# model_prunned_419

    model1 = torch.load("./model_car_flowcnn",map_location=lambda storage, loc: storage).to(device)
    model1.eval()
    model2 = torch.load("./model_car_flowcnn_phy",map_location=lambda storage, loc: storage).to(device)
    #model2 = torch.load("model_selu",map_location=lambda storage, loc: storage).to(device)
    model2.eval()
    # print(model)
    filenames = glb('./data/computed_car_flow/*/')
    shape = [128, 256]
    with torch.no_grad():
        fig_num = 0 
        for run in filenames:
            #print ("case %d :" %(fig_num + 1))
            fig_num = fig_num + 1
            # if fig_num > 1 :
            #     break

            flow_name = run + '/fluid_flow_0002.h5'
            boundary_np = load_boundary(flow_name, shape).reshape([1,1, shape[0], shape[1]])
            sflow_true = load_flow(flow_name, shape)
            boundary_t = torch.FloatTensor(boundary_np).to(device)
            sflow_t = torch.FloatTensor(sflow_true)
            sflow_true = sflow_t.permute(2,0,1).numpy()
            sflow_t = torch.FloatTensor(sflow_true.reshape([1,2, shape[0], shape[1]]))
#             print(type(sflow_true))
#             sflow_generated = model(boundary_t)[0].cpu().numpy()
            
            start = time.time()
#            for i in range(100):
            sflow_generated_t1 = model1(boundary_t)
            end = time.time()
            sflow_generated1 = sflow_generated_t1[0].cpu().numpy()

            running_time = (end - start)*1000
            #print('model1 time cost : %.5f ms' %running_time)
            lossPer_v1 = ( np.sum(np.abs(sflow_generated1[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated1[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum1 += lossPer_v1.item()
            #print("model1: ", lossPer_v1.item())
            crit = loss_mass().to(device)
            real , lossphy_v1, real2 , lossphy_v1_mo = crit(sflow_generated_t1, sflow_t)
            # lossphy_v_accum1 += lossphy_v1.item()
            # print (lossphy_v1)




            start = time.time()
#            for i in range(100):
            sflow_generated_t2 = model2(boundary_t)
            end = time.time()
            sflow_generated2 = sflow_generated_t2[0].cpu().numpy()
            running_time = (end - start)*1000
            #print('model2 time cost : %.5f ms' %running_time)
            lossPer_v2 = ( np.sum(np.abs(sflow_generated2[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated2[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum2 += lossPer_v2.item()
            #print("model1: ", lossPer_v2.item())
            crit = loss_mass().to(device)
            real, lossphy_v2, real2 , lossphy_v2_mo  = crit(sflow_generated_t2, sflow_t)
            # lossphy_v_accum2 += lossphy_v2.item()

#             print(sflow_true.shape,sflow_generated.shape)

            input_ndarray = boundary_t[0].cpu().numpy()



            # sflow_plot1 = np.concatenate([real,lossphy_v1,real - lossphy_v1], axis=0) 
            # sflow_plot2 = np.concatenate([real,lossphy_v2,real - lossphy_v2], axis=0)

            # sflow_plot = np.concatenate([sflow_plot1,sflow_plot2], axis=0)

            # plt.imsave('./pic_physical/purn_'+str(fig_num)+'.png',sflow_plot)
            
            sflow_plot1 = np.concatenate([real,lossphy_v1,lossphy_v2], axis=0)
            sflow_plot2 = np.concatenate([real2,lossphy_v1_mo,lossphy_v2_mo], axis=0)

            sns.set(font_scale=4)
        
            f, (ax1,ax2) = plt.subplots(figsize=(24,18),ncols=2,gridspec_kw={'width_ratios':[12,12]})


            sns.heatmap(sflow_plot1, ax= ax1, cmap='coolwarm',xticklabels = False, yticklabels = False, cbar_kws={}) #,cbar = False
            sns.heatmap(sflow_plot2, ax= ax2, cmap='coolwarm',xticklabels = False, yticklabels = False) #vmax = 0.05,vmin= 0.0,

            # ax1.set_xlabel(r'$L_1$')

            # ax2.set_xlabel(r'$L_{physical}$')
            fig = ax2.get_figure()
            fig.savefig('./pic_physical/purn_'+str(fig_num)+'.png')



        lossPer_v_accum1 /= len(filenames)
        lossPer_v_accum2 /= len(filenames)

        # lossphy_v_accum1 /= len(filenames)
        # lossphy_v_accum2 /= len(filenames)

        print( "Loss, mass_loss percentage v model1:   %f %% , %f %%  " % ( lossPer_v_accum1*100, lossphy_v_accum1*100) )
        print( "Loss, mass_loss percentage v model2:   %f %% , %f %%  " % ( lossPer_v_accum2*100, lossphy_v_accum2*100) )



if __name__ == '__main__':
    main()
