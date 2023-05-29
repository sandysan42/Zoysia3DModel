import numpy as np
import pandas as pd
import open3d as o3d
import pathlib
import cv2
import imutils
from PIL import Image
from Voxelization import voxelize_fill
from ColorCalculation import color_calculation

path = pathlib.Path('./')
density = 0.001
border = 0.02
height = 0.19
radius = 0.1
pixel_size = 1

def CreateContour(binary):
    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    if len(cntsSorted) != 0:
        return cntsSorted[-1]

def main():
    dir_list = list(path.glob('*'))
    dir_list.sort()

    for ply_dir in dir_list:
        plant_ID = str(ply_dir.relative_to(path))
        date_list = list(ply_dir.glob('*.ply'))
        date_list.sort()
        for date_path in date_list:
            input_PLY = str(date_path)
            the_date = str(date_path.relative_to(ply_dir))
            date = the_date.split('_')[1]
            data_csv_path = pathlib.Path(pathlib.Path.cwd(),"CSV_DATA","{}_data.csv".format(plant_ID))
            topview_png = pathlib.Path(pathlib.Path.cwd(),"PNG_BIRDEYE",plant_ID,"{}_{}.png".format(plant_ID,date))
            index_png = pathlib.Path(pathlib.Path.cwd(),"PNG_BIRDEYE",plant_ID,"{}_{}_GRNDI.png".format(plant_ID,date))

            pcd = o3d.io.read_point_cloud(str(input_PLY))

            #open3d->numpy
            array_point = np.asarray(pcd.points)
            array_color = np.asarray(pcd.colors)
            array = np.concatenate([array_point,array_color],1) 

            #Plant Height
            try:
                max_height = np.max(array_point,axis=0)[2]
            except ValueError:
                max_height = 0

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
        
            img_color = np.asarray(img_color)
            gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
            (T,binary) = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

            c = CreateContour(binary)
            cv2.drawContours(img_color, [c], -1, (0, 255, 255), 1)

            area = cv2.contourArea(c)
            area = area * (pixel_size**2)/100

            #3D analysis
            _,_,filled_vol = voxelize_fill(pcd=pcd,voxel_size=0.003)

            index_img, NDI = color_calculation(pcd=pcd)

            write = [plant_ID,date,max_height,area,filled_vol,NDI]

            if len(array_point) == 0:
                print('{} {} No points'.format(plant_ID,date))
            else:
                with open(data_csv_path, 'w',newline='\n') as file_object:
                    file_write = csv.writer(file_object) 
                    file_write.writerow(write)
                    file_object.close() 

                topview_png.parent.mkdir(parents = True,exist_ok=True)
                cv2.imwrite(str(topview_png),img_color)

                index_png.parent.mkdir(parents = True,exist_ok=True)
                index_img.save(str(index_png))
                print("{} {} Done".format(plant_ID,date))

if __name__ == "__main__":
    main()
