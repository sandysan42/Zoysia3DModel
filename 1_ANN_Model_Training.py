import tensorflow as tf
import numpy as np
import open3d as o3d
import random
import pathlib

path = pathlib.Path('./')
dir_list = list(path.glob('PLY_PLANT*'))
dir_list.sort()
pot_dir_list = list(path.glob('PLY_POT*'))
pot_dir_list.sort()

training_point = [8192]
all_eval = []

for point in training_point:
    train_data = np.empty((0,7))
    test_data = np.empty((0,7))
    train_plant = np.empty((0,7))
    train_pot = np.empty((0,7))
    test_plant =np.empty((0,7))
    test_pot = np.empty((0,7))
    for ply_dir in dir_list:
        dir_name = str(ply_dir.relative_to(path))
        if dir_name == 'PLY_PLANT_TRAIN':
            pot_dir = pathlib.Path(pathlib.Path.cwd(),'PLY_POT_TRAIN')
            ply_list = list(ply_dir.glob('*.ply'))
            pot_ply_list = list(pot_dir.glob('*.ply'))
            for ply in ply_list:
                ply_name = str(ply.relative_to(ply_dir))
                pcd_plant = o3d.io.read_point_cloud(str(ply))
                array_point_plant = np.asarray(pcd_plant.points)
                array_color_plant = np.asarray(pcd_plant.colors)
                array_plant = np.concatenate((array_point_plant,array_color_plant,np.zeros(len(array_point_plant)).reshape(-1,1)),axis=1)
                try:
                    sampled_indices = random.sample(list(range(len(array_plant))), point)
                except ValueError:
                    sampled_indices = random.sample(list(range(len(array_plant))), len(array_plant))
                array_plant = np.array([array_plant[i] for i in sampled_indices])
                train_plant = np.concatenate((train_plant,array_plant),0)

            for pot_ply in pot_ply_list:
                pot_name = str(pot_ply.relative_to(pot_dir))
                pcd_pot = o3d.io.read_point_cloud(str(pot_ply))
                array_point_pot = np.asarray(pcd_pot.points)
                array_color_pot = np.asarray(pcd_pot.colors)
                array_pot = np.concatenate((array_point_pot,array_color_pot,np.ones(len(array_point_pot)).reshape(-1,1)),axis=1)
                try:
                    sampled_indices = random.sample(list(range(len(array_pot))), point)
                except ValueError:
                    sampled_indices = random.sample(list(range(len(array_pot))), len(array_pot))

                array_pot = np.array([array_pot[i] for i in sampled_indices])
                train_pot = np.concatenate((train_pot,array_pot),0)

            full_data = np.concatenate((train_plant,train_pot),axis=0)
            np.random.shuffle(full_data)

            train_data = np.concatenate((train_data,full_data),0)
                

        if dir_name == 'PLY_PLANT_TEST':
            pot_dir = pathlib.Path(pathlib.Path.cwd(),'PLY_POT_TEST')
            ply_list = list(ply_dir.glob('*.ply'))
            pot_ply_list = list(pot_dir.glob('*.ply'))
            for ply in ply_list:
                ply_name = str(ply.relative_to(ply_dir))
                pcd_plant = o3d.io.read_point_cloud(str(ply))
                array_point_plant = np.asarray(pcd_plant.points)
                array_color_plant = np.asarray(pcd_plant.colors)
                array_plant = np.concatenate((array_point_plant,array_color_plant,np.zeros(len(array_point_plant)).reshape(-1,1)),axis=1)
                try:
                    sampled_indices = random.sample(list(range(len(array_plant))), point)
                except ValueError:
                    sampled_indices = random.sample(list(range(len(array_plant))), len(array_plant))

                array_plant = np.array([array_plant[i] for i in sampled_indices])
                test_plant = np.concatenate((test_plant,array_plant),0)

            for pot_ply in pot_ply_list:
                pot_name = str(pot_ply.relative_to(pot_dir))
                pcd_pot = o3d.io.read_point_cloud(str(pot_ply))
                array_point_pot = np.asarray(pcd_pot.points)
                array_color_pot = np.asarray(pcd_pot.colors)
                array_pot = np.concatenate((array_point_pot,array_color_pot,np.ones(len(array_point_pot)).reshape(-1,1)),axis=1)
                try:
                    sampled_indices = random.sample(list(range(len(array_pot))), point)
                except ValueError:
                    sampled_indices = random.sample(list(range(len(array_pot))), len(array_pot))

                array_pot = np.array([array_pot[i] for i in sampled_indices])
                test_pot = np.concatenate((test_pot,array_pot),0)

            full_data = np.concatenate((test_plant,test_pot),axis=0)
            np.random.shuffle(full_data)

            test_data = np.concatenate((test_data,full_data),0)

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    #Create model using Sequential API
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')]
    )

    #Compile model
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # model.summary()

    print('Start training...')

    X,y = train_data[:,:6], train_data[:,6]
    model.fit(X,y,epochs=20,validation_data=(test_data[:,:6],test_data[:,6]))

    print('Start predicting...')
    evaluation = model.evaluate(test_data[:,:6],test_data[:,6])
    print(evaluation)
    all_eval.append(evaluation)

    model.save('saved_model/pot_rm_'+str(point)+'_points.h5')

print(all_eval)