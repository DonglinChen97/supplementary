import os
import numpy as np
import sys
import matplotlib.pyplot as plt



def Find_boundary(target, array):
    # write code here
    lis= []
    m, n = len(array),len(array[0])
    for i in range(m):
        for j in range(n-1):
            cur = array[i][j]
            cur2 = array[i][j+1]

            if cur2 == target and cur != target:
                lis.append((i,j))
            if cur == target and cur2 != target:
                lis.append((i,j+1))
    return lis



def painting(i,shape,real,real2,pred,pred2):
    lis = Find_boundary(1,shape)
    #print(lis)
    #print(pres)
    max_x = max([x[0] for x in lis])
    min_x = min([x[0] for x in lis])

    max_y = max([x[1] for x in lis])
    min_y = min([x[1] for x in lis])

    lis = []
    for x in range(min_x, max_x, 1):
        for y in range(min_y, max_y, 1):
            lis.append((x,y))

    pixel_x = [x[0] for x in lis]

    pixel_y_real = [real[x[0]][x[1]] for x in lis]
    pixel_y_real2 = [real2[x[0]][x[1]] for x in lis]

    pixel_y_pred = [pred[x[0]][x[1]] for x in lis] 
    # pixel_y_pred = [x for x in pixel_y_pred]

    pixel_y_pred2 = [pred2[x[0]][x[1]] for x in lis] 
    # pixel_y_pred2 = [x for x in pixel_y_pred2]

    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    ax1.scatter(pixel_x,pixel_y_real,c = 'r',marker = 'o', s = 6,label= 'Ground truth')
    ax1.scatter(pixel_x,pixel_y_pred,c = 'green',marker = '^',s = 6,label = 'Ours')
    plt.ylabel('x-velocity',fontsize=20)
    plt.legend(loc = 'upper left',fontsize=10,ncol = 2)
    plt.xticks([])
    plt.yticks(fontsize=20)

    ax2 = fig.add_subplot(212)
    ax2.scatter(pixel_x,pixel_y_real,c = 'r',marker = 'o', s = 6,label= 'Ground truth')
    ax2.scatter(pixel_x,pixel_y_pred2,c = 'green',marker = '^', s = 6,label = 'Coder')

    plt.xlabel('X',fontsize=20)
    plt.ylabel('x-velocity',fontsize=20)
    plt.legend(loc = 'upper left',fontsize=10,ncol = 2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Scatter Plot')

    # ax1.scatter(pixel_x,pixel_y_real,c = 'r',marker = 'o', label= 'real')
    # ax1.scatter(pixel_x,pixel_y_pred,c = 'green',marker = '^',label = 'pred')

    plt.savefig("./draw/sectional_drawing_"+ str(i)+ ".pdf",dpi=400,bbox_inches='tight')
    plt.close()
    '''

    e1 =  list(map(lambda x: x[0]-x[1], zip(pixel_y_pred, pixel_y_real)))
    e2 =  list(map(lambda x: x[0]-x[1], zip(pixel_y_pred2, pixel_y_real))) 

    error_s1 = np.sum(np.abs(e1)) /  ( np.sum(np.abs(pixel_y_real)) + np.sum(np.abs(pixel_y_real2)) )
    error_s2 = np.sum(np.abs(e2)) /  ( np.sum(np.abs(pixel_y_real)) + np.sum(np.abs(pixel_y_real2)) )

    return error_s1 , error_s2 






