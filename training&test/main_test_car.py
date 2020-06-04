import matplotlib.pyplot as plt
import h5py
import numpy as np 
from glob import glob as glb
#from train import *
#from DfpNet import *
#from DfpNet_own import TurbNetG
from LossFunc_mass import MyLoss as loss_mass
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



    model1 = torch.load("./model_car_flowcnn",map_location=lambda storage, loc: storage).to(device)
    model1.eval()
    model2 = torch.load("./model_car_flowcnn_phy",map_location=lambda storage, loc: storage).to(device)
    #model2 = torch.load("model_selu",map_location=lambda storage, loc: storage).to(device)
    model2.eval()
    # print(model)
    # filenames = glb('./data/computed_car_flow/*/')
    filenames = glb('./data/traindata/*/') # load new
    shape = [128, 256]
    with torch.no_grad():
        fig_num = 0 
        for run in filenames:
            #print ("case %d :" %(fig_num + 1))
            fig_num = fig_num + 1
            if fig_num > 60 :
                break
            flow_name = run + '/fluid_flow_0002.h5'
            boundary_np = load_boundary_new(flow_name, shape).reshape([1,1, shape[0], shape[1]])
            sflow_true = load_flow_new(flow_name, shape)
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
            lossphy_v1 = crit(sflow_generated_t1, sflow_t)
            lossphy_v_accum1 += lossphy_v1.item()




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
            lossphy_v2 = crit(sflow_generated_t2, sflow_t)
            lossphy_v_accum2 += lossphy_v2.item()

#             print(sflow_true.shape,sflow_generated.shape)

            input_ndarray = boundary_t[0].cpu().numpy()
            erx1,erx2 = painting(fig_num,input_ndarray[0],sflow_true[0],sflow_true[1],sflow_generated1[0],sflow_generated2[0])
            ery1,ery2 = painting(fig_num,input_ndarray[0],sflow_true[1],sflow_true[0],sflow_generated1[1],sflow_generated2[1])

            filename1 = 'model5_error_s.txt'
            with open(filename1, 'a') as file_object:
                file_object.write(str(erx1 + ery1) + "\n")
            # filename2 = 'model2_error_s.txt'
            # with open(filename2, 'a') as file_object:
            #     file_object.write(str(erx2 + ery2) + "\n")

            err1.append(erx1)
            err1.append(ery1)
            err2.append(erx2)
            err2.append(ery2)


            sflow_temp = np.concatenate([sflow_true], axis=1) 

            boundary_concat = np.concatenate([boundary_np], axis=2)
            #print(sflow_plot.shape,boundary_concat.shape)

            # sflow_plot = np.sqrt(np.square(sflow_temp[0,:,:]) + np.square(sflow_temp[1,:,:]))  - 0.05 *boundary_concat[0,0,:,:]
            sflow_plot = boundary_concat[0,0,:,:]
            # diff1 = sflow_true - sflow_generated1
            # diff1 = np.sqrt(np.square(diff1[0,:,:]) + np.square(diff1[1,:,:]))
            # diff2 = sflow_true - sflow_generated2
            # diff2 = np.sqrt(np.square(diff2[0,:,:]) + np.square(diff2[1,:,:]))

            # sflow_plot = np.concatenate([sflow_plot,diff1,diff2], axis=0)



            #sns.set()
            #ax = sns.heatmap(sflow_plot)
            #fig = ax.get_figure()
            #fig.savefig('./pic2/purn_'+str(fig_num)+'.png')
            
            plt.imsave('./pic2/purn_'+str(fig_num)+'.png',sflow_plot)
        lossPer_v_accum1 /= len(filenames)
        lossPer_v_accum2 /= len(filenames)

        lossphy_v_accum1 /= len(filenames)
        lossphy_v_accum2 /= len(filenames)

        print( "Loss, mass_loss percentage v model1:   %f %% , %f %%  " % ( lossPer_v_accum1*100, lossphy_v_accum1*100) )
        print( "Loss, mass_loss percentage v model2:   %f %% , %f %%  " % ( lossPer_v_accum2*100, lossphy_v_accum2*100) )

        print("err1: %f" % (  2 * np.sum(err1) / len(err1)) )
        print("err2: %f" % (  2 * np.sum(err2) / len(err2)) )

if __name__ == '__main__':
    main()
