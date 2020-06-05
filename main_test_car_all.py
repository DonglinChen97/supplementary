import matplotlib.pyplot as plt
import h5py
import numpy as np 
from glob import glob as glb
#from train import *
#from DfpNet import *
#from DfpNet_own import TurbNetG
import torch
import os
import time
import sys
import seaborn as sns


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
    lossPer_v_accum3 = 0 
    lossPer_v_accum4 = 0
    lossPer_v_accum5 = 0

    model1 = torch.load("./model_car_cnet",map_location=lambda storage, loc: storage).to(device)
    model1.eval()
    model2 = torch.load("./model_car_tnet",map_location=lambda storage, loc: storage).to(device)
    #model2 = torch.load("model_selu",map_location=lambda storage, loc: storage).to(device)
    model2.eval()
    model3 = torch.load("./model_car_unet",map_location=lambda storage, loc: storage).to(device)
    model3.eval()
    model4 = torch.load("./model_car_flowcnn_phy",map_location=lambda storage, loc: storage).to(device)
    model4.eval()
    model5 = torch.load("./model_car_flowcnn_purnned_best",map_location=lambda storage, loc: storage).to(device)
    model5.eval()
    # print(model)
    filenames = glb('./data/computed_car_flow/*/')
    shape = [128, 256]
    with torch.no_grad():
        fig_num = 0 
        for run in filenames:
            #print ("case %d :" %(fig_num + 1))
            fig_num = fig_num + 1
            flow_name = run + '/fluid_flow_0002.h5'
            boundary_np = load_boundary(flow_name, shape).reshape([1,1, shape[0], shape[1]])
            sflow_true = load_flow(flow_name, shape)
            boundary_t = torch.FloatTensor(boundary_np).to(device)
            sflow_t = torch.FloatTensor(sflow_true)
            sflow_true = sflow_t.permute(2,0,1).numpy()
#             print(type(sflow_true))
#             sflow_generated = model(boundary_t)[0].cpu().numpy()
            
            start = time.time()
#            for i in range(100):
            temp = model1(boundary_t)
            end = time.time()
            sflow_generated1 = temp[0].cpu().numpy()

            running_time = (end - start)*1000
            #print('model1 time cost : %.5f ms' %running_time)
            lossPer_v1 = ( np.sum(np.abs(sflow_generated1[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated1[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum1 += lossPer_v1.item()
            #print("model1: ", lossPer_v1.item())

            start = time.time()
#            for i in range(100):
            temp = model2(boundary_t)
            end = time.time()
            sflow_generated2 = temp[0].cpu().numpy()
            running_time = (end - start)*1000
            #print('model2 time cost : %.5f ms' %running_time)
            lossPer_v2 = ( np.sum(np.abs(sflow_generated2[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated2[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum2 += lossPer_v2.item()
            #print("model1: ", lossPer_v2.item())

            start = time.time()
#            for i in range(100):
            temp = model3(boundary_t)
            end = time.time()
            sflow_generated3 = temp[0].cpu().numpy()

            running_time = (end - start)*1000
            #print('model1 time cost : %.5f ms' %running_time)
            lossPer_v3 = ( np.sum(np.abs(sflow_generated3[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated3[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum3 += lossPer_v3.item()
            #print("model1: ", lossPer_v1.item())

            start = time.time()
#            for i in range(100):
            temp = model4(boundary_t)
            end = time.time()
            sflow_generated4 = temp[0].cpu().numpy()

            running_time = (end - start)*1000
            #print('model1 time cost : %.5f ms' %running_time)
            lossPer_v4 = ( np.sum(np.abs(sflow_generated4[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated4[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum4 += lossPer_v4.item()


            start = time.time()
#            for i in range(100):
            temp = model5(boundary_t)
            end = time.time()
            sflow_generated5 = temp[0].cpu().numpy()

            running_time = (end - start)*1000
            #print('model1 time cost : %.5f ms' %running_time)
            lossPer_v5 = ( np.sum(np.abs(sflow_generated5[0] - sflow_true[0])) + np.sum(np.abs(sflow_generated5[1] - sflow_true[1])) ) / ( np.sum(np.abs(sflow_true[0])) + np.sum(np.abs(sflow_true[1])) )
            lossPer_v_accum5 += lossPer_v5.item()


            filename1 = 'model1_error.txt'
            with open(filename1, 'a') as file_object:
                file_object.write(str(lossPer_v1) + "\n")
            filename2 = 'model2_error.txt'
            with open(filename2, 'a') as file_object:
                file_object.write(str(lossPer_v2) + "\n")
            filename3 = 'model3_error.txt'
            with open(filename3, 'a') as file_object:
                file_object.write(str(lossPer_v3) + "\n")
            filename4 = 'model4_error.txt'
            with open(filename4, 'a') as file_object:
                file_object.write(str(lossPer_v4) + "\n")
            filename5 = 'model5_error.txt'
            with open(filename5, 'a') as file_object:
                file_object.write(str(lossPer_v5) + "\n")


#             print(sflow_true.shape,sflow_generated.shape)


            sflow_temp = np.concatenate([sflow_true, sflow_generated1, sflow_generated2,sflow_generated3,sflow_generated4,sflow_generated5], axis=2) 
            boundary_concat = np.concatenate(6*[boundary_np], axis=3)
            sflow_plot = np.sqrt(np.square(sflow_temp[0,:,:]) + np.square(sflow_temp[1,:,:]))  - 0.05 *boundary_concat[0,0,:,:]

            # diff1 = sflow_true - sflow_generated1
            # diff1 = np.sqrt(np.square(diff1[0,:,:]) + np.square(diff1[1,:,:]))
            # diff2 = sflow_true - sflow_generated2
            # diff2 = np.sqrt(np.square(diff2[0,:,:]) + np.square(diff2[1,:,:]))
            # diff3 = sflow_true - sflow_generated3
            # diff3 = np.sqrt(np.square(diff3[0,:,:]) + np.square(diff3[1,:,:]))
            # diff4 = sflow_true - sflow_generated4
            # diff4 = np.sqrt(np.square(diff4[0,:,:]) + np.square(diff4[1,:,:]))
            # diff5 = sflow_true - sflow_generated5
            # diff5 = np.sqrt(np.square(diff5[0,:,:]) + np.square(diff5[1,:,:]))

            # sflow_plot = np.concatenate([diff1, diff2, diff3,diff4,diff5], axis=1) 


            # sflow_plot = np.concatenate([sflow_plot,diff1,diff2,diff3], axis=1)



            #print (sflow_plot.shape)
            #plt.imsave('./pic2/purn_'+str(fig_num)+'.jpg',sflow_plot)
            #print (sflow_plot.shape)
            

            #sns.set()
            #ax = sns.heatmap(sflow_plot)
            #fig = ax.get_figure()
            #fig.savefig('./pic2/purn_'+str(fig_num)+'.png')
            
            plt.imsave('./pic_xin/final_'+str(fig_num)+'.png',sflow_plot)
        lossPer_v_accum1 /= len(filenames)
        lossPer_v_accum2 /= len(filenames)
        lossPer_v_accum3 /= len(filenames)
        lossPer_v_accum4 /= len(filenames)
        lossPer_v_accum5 /= len(filenames)
        print( "Loss percentage v model1:   %f %%  " % ( lossPer_v_accum1*100 ) )
        print( "Loss percentage v model2:   %f %%  " % ( lossPer_v_accum2*100 ) )
        print( "Loss percentage v model3:   %f %%  " % ( lossPer_v_accum3*100 ) )
        print( "Loss percentage v model4:   %f %%  " % ( lossPer_v_accum4*100 ) )
        print( "Loss percentage v model5:   %f %%  " % ( lossPer_v_accum5*100 ) )

if __name__ == '__main__':
    main()
