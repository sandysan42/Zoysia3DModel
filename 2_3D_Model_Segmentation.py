import tensorflow as tf
import numpy as np
import open3d as o3d
import pathlib

segment_model = tf.keras.models.load_model('model.h5')
path = pathlib.Path('./')

dir_list = list(path.glob('*'))
dir_list.sort()
# print(dir_list)
for ply_dir in dir_list:
    Plant_dir = pathlib.Path(ply_dir,'PLY')
    plant_ID = str(ply_dir.relative_to(path))
    date_list = list(Plant_dir.glob('*'))
    date_list.sort()
    for date_path in date_list:
        input_list = list(date_path.glob('*.ply'))
        try:
            input_PLY = str(input_list[0])
        except IndexError:
            continue
        date = str(date_path.relative_to(Plant_dir))
        output_PLY = pathlib.Path('PLY_PLANT',plant_ID, plant_ID+'_'+date+'_plant.ply')
        output_pot_PLY = pathlib.Path('PLY_POT',plant_ID,plant_ID+'_'+date+'_pot.ply')


        pcd = o3d.io.read_point_cloud(str(input_PLY))

    #open3d->numpy
        array_point = np.asarray(pcd.points)
        array_color = np.asarray(pcd.colors)

        array = np.concatenate([array_point,array_color],1) 

        tensor = tf.convert_to_tensor(array, dtype=tf.float32)
        #predict
        array_prob = segment_model.predict(tensor)
        array_pred = (array_prob > 0.5).astype(int)
        
        #segment and save
        array_segment = array[array_pred.reshape(-1)==0,:]
        array_segment_point = array_segment[:,:3]
        array_segment_color = array_segment[:,3:]
        pcd_segment = o3d.geometry.PointCloud()
        pcd_segment.points = o3d.utility.Vector3dVector(array_segment_point)
        pcd_segment.colors = o3d.utility.Vector3dVector(array_segment_color)

        pcd_segment, ind = pcd_segment.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=20.0)

        output_PLY.parent.mkdir(parents = True,exist_ok=True)
        o3d.io.write_point_cloud(str(output_PLY),pcd_segment)

        #pot segment and save
        array_pot = array[array_pred.reshape(-1)==1,:]
        array_pot_point = array_pot[:,:3]
        array_pot_color = array_pot[:,3:]
        pcd_pot = o3d.geometry.PointCloud()
        pcd_pot.points = o3d.utility.Vector3dVector(array_pot_point)
        pcd_pot.colors = o3d.utility.Vector3dVector(array_pot_color)
        output_pot_PLY.parent.mkdir(parents = True,exist_ok=True)
        o3d.io.write_point_cloud(str(output_pot_PLY),pcd_pot)

        print('{} {} saved'.format(plant_ID,date))
