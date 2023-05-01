import numpy as np
from PIL import Image

def color_calculation(pcd,index='NDI',density = 0.001,border = 0.02):
            array_point = np.asarray(pcd.points)
            array_color = np.asarray(pcd.colors)
            R = array_color[:,0]
            G = array_color[:,1]
            B = array_color[:,2]

            # #Excess Green Index
            # EGI = (2*G) - R - B
            # #Green Index
            # GI = G/R
            # #Green Leaf Index
            # GLI = ((2*G) - R - B) / ((2*G) + R + B)
            # #GreenBlue Index
            # GBI = (G - B)/(G + B)
            #Normalized Difference Index
            NDI = (G - R)/(G + R)

            # #Other color index
            # RB = (R - B)/(R + B)
            # KN1 = R/(R + G + B)
            # KN2 = G/(R + G + B)
            # KN3 = B/(R + G + B)
            # KN4 = G - B
            # Wang1 = R - B
            # Wang2 = (R + B + G)/3

            # Veg_Ind = {'EGI':EGI,'GI':GI,'GLI':GLI,'GBI':GBI,'NDI':NDI,'RB':RB,'KN1':KN1,'KN2':KN2,'KN3':KN3,'KN4':KN4,'Wang1':Wang1,'Wang2':Wang2}
            np_data = np.asarray(['Index', 'Mean', 'Std', 'Min', 'Max'])
            normalized_NDI = (NDI-(-0.15))/((0.35)-(-0.15))
            invert_NDI = abs(normalized_NDI - 1)

            k = index.upper()
            v = NDI

            np_data = np.vstack((np_data,np.asarray([k,v.mean(),v.std(),v.min(),v.max()])))

            zero = np.zeros((v.shape[0],1))
            array_ind = np.concatenate([invert_NDI.reshape(NDI.shape[0],1),normalized_NDI.reshape(NDI.shape[0],1),zero],axis=1)
            array = np.concatenate([array_point,array_ind],axis=1)

            min_x = np.min(array_point,axis=0)[0]-density
            min_y = np.min(array_point,axis=0)[1]-density
            max_x = np.max(array_point,axis=0)[0]+density
            max_y = np.max(array_point,axis=0)[1]+density

            size_x = int((max_x-min_x)*(1/density))+1
            size_y = int((max_y-min_y)*(1/density))+1
            max_value = np.full((size_x,size_y),border) 
            img_color=Image.new('RGB',(size_x,size_y),(0,0,0))

            for loop in array:
                    tmp_x,tmp_y,tmp_z = (int((loop[0]-min_x) * (1/density)) , int((loop[1]-min_y) * (1/density)), loop[2])
                    if max_value[tmp_x][tmp_y] < tmp_z:
                        max_value[tmp_x][tmp_y] = tmp_z

                        if len(array_color) == 0:
                            img_color.putpixel((tmp_x,tmp_y),(255,255,255))
                        else:
                            img_color.putpixel((tmp_x,tmp_y),(int(loop[3]*255),int(loop[4]*255),int(loop[5]*255)))
                
            return  img_color, v.mean()
