import torch
import numpy as np



def find_layer(m_length,s_module,layer_index):
    ind = 0
    temp = layer_index
    for item in m_length:
        if temp - item >= 0:
            temp = temp - item 
            ind = ind + 1
        else:
            return temp, ind , list(s_module[ind]._modules.items())[temp]


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_conv_layer(model, layer_index, filter_index):
    m_length = []
    s_module = []
    # print("yessss")
    temp_model = torch.nn.Sequential(*list(model.children()))

    # print(temp_model[1][0]._modules['ca']._modules['fc1'])

    for _, (name, module) in enumerate(model._modules.items()):
        m_length.append(len(module))
        s_module.append(module)

    # print (m_length)
    conv_local_ind , ind ,(conv_name, conv) = find_layer(m_length,s_module,layer_index)

    # print (conv_local_ind , ind)
    next_conv = None

    next_conv_local_ind = 0
    if ind == 12:
        for item in list(s_module[ind+2]._modules.items()):
            if isinstance(item[1], torch.nn.modules.conv.ConvTranspose2d):
                # print("yes")
                next_name, next_conv = item
                break
            next_conv_local_ind = next_conv_local_ind + 1 		
    else:
        for item in list(s_module[ind+2]._modules.items()):
            if isinstance(item[1], torch.nn.modules.conv.Conv2d):
                # print("yes")
                next_name, next_conv = item
                break
            next_conv_local_ind = next_conv_local_ind + 1 


    next_conv1 = None
    next_conv_local_ind1 = 0

    # print(s_module[ind+1][0])

    for item in list(s_module[20-int(ind/2)]._modules.items()):
        # print (len(item))
        if isinstance(item[1], torch.nn.modules.conv.ConvTranspose2d):
            next_name1, next_conv1 = item
            break
        next_conv_local_ind1 =  next_conv_local_ind1 + 1  
    # while layer_index + offset <  len(model.features._modules.items()):
    #     res = list(model.features._modules.items())[layer_index+offset]
    #     if isinstance(res[1], torch.nn.modules.conv.Conv2d):
    #         next_name, next_conv = res
    #         break
    #     offset = offset + 1
    
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
                        out_channels = conv.out_channels - 1,
                        kernel_size = conv.kernel_size, \
                        stride = conv.stride,
                        padding = conv.padding,
                        dilation = conv.dilation,
                        groups = conv.groups,
                        bias = (conv.bias is not None))
    
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    
    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()
        
        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(bias).cuda()
    
    if not next_conv is None:
        if ind == 12:
            next_new_conv = \
                torch.nn.ConvTranspose2d(in_channels = next_conv.in_channels - 1,\
                                out_channels =  next_conv.out_channels, \
                                kernel_size = next_conv.kernel_size, \
                                stride = next_conv.stride,
                                padding = next_conv.padding,
                                output_padding = next_conv.output_padding,
                                groups = next_conv.groups,
                                bias = (next_conv.bias is not None))
            
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[: filter_index,:, :, :] = old_weights[: filter_index,:, :, :]
            new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
            next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
            if next_conv.bias is not None:
                next_new_conv.bias.data = next_conv.bias.data	
        else:
            next_new_conv = \
                torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                                out_channels =  next_conv.out_channels, \
                                kernel_size = next_conv.kernel_size, \
                                stride = next_conv.stride,
                                padding = next_conv.padding,
                                dilation = next_conv.dilation,
                                groups = next_conv.groups,
                                bias = (next_conv.bias is not None))
			
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
            next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
            if next_conv.bias is not None:
                next_new_conv.bias.data = next_conv.bias.data

    if not next_conv1 is None:
        next_new_conv1 = \
            torch.nn.ConvTranspose2d(in_channels = next_conv1.in_channels - 1,\
                            out_channels =  next_conv1.out_channels, \
                            kernel_size = next_conv1.kernel_size, \
                            stride = next_conv1.stride,
                            padding = next_conv1.padding,
                            output_padding = next_conv1.output_padding,
                            groups = next_conv1.groups,
                            bias = (next_conv1.bias is not None))
        
        old_weights = next_conv1.weight.data.cpu().numpy()
        new_weights = next_new_conv1.weight.data.cpu().numpy()
        
        new_weights[: filter_index,:,  :, :] = old_weights[: filter_index,:,  :, :]
        new_weights[filter_index : ,:,  :, :] = old_weights[filter_index + 1 :,:,  :, :]
        next_new_conv1.weight.data = torch.from_numpy(new_weights).cuda()
        if next_conv1.bias is not None:
            next_new_conv1.bias.data = next_conv1.bias.data



    ############ replace ####################
  
    temp_model[ind][conv_local_ind] = new_conv       
    temp_model[ind][conv_local_ind + 1] = torch.nn.BatchNorm2d(new_conv.out_channels)
    temp_model[ind+2][next_conv_local_ind] = next_new_conv
    temp_model[20-int(ind/2)][next_conv_local_ind1] = next_new_conv1


    model.layer1 = temp_model[0]
    model.CBAM1 = temp_model[1]
    model.layer2 = temp_model[2]
    model.CBAM2 = temp_model[3]
    model.layer3 = temp_model[4]
    model.CBAM3 = temp_model[5]
    model.layer4 = temp_model[6]
    model.CBAM4 = temp_model[7]
    model.layer5 = temp_model[8]
    model.CBAM5 = temp_model[9]
    model.layer6 = temp_model[10]
    model.CBAM6 = temp_model[11]
    model.layer7 = temp_model[12]
    model.CBAM7 = temp_model[13]
    model.dlayer7 = temp_model[14]
    model.dlayer6 = temp_model[15]
    model.dlayer5 = temp_model[16]
    model.dlayer4 = temp_model[17]
    model.dlayer3 = temp_model[18]
    model.dlayer2 = temp_model[19]
    model.dlayer1 = temp_model[20]


    del temp_model

    return model
