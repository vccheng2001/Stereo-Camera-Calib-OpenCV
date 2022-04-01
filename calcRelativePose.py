# calc relative pose 
'''
Given T_cam1_world, T_cam2_world, solve T_cam1_cam2
'''
import numpy as np
import yaml


d = {} 
VERSION = 2

def yaml_load(file_name, item_name):
    with open(file_name) as file:
        res = yaml.full_load(file)
        return res[item_name]

file1 = f'ARUCO_TWOCAMS_v{VERSION}/CAM1_T_v{VERSION}.yaml'
file2 = f'ARUCO_TWOCAMS_v{VERSION}/CAM2_T_v{VERSION}.yaml'



Tcam1_world = yaml_load(file1, 'Tcam1_world')
Tcam2_world = yaml_load(file2, 'Tcam2_world')

Tcam1_cam2 = Tcam1_world @ np.linalg.inv(Tcam2_world)
print('Tcam1_cam2: ', Tcam1_cam2)

data = {'Tcam1_cam2': np.asarray(Tcam1_cam2).tolist()}
with open(f"ARUCO_TWOCAMS_v{VERSION}/Tcam1_cam2_v{VERSION}.yaml", "w") as f:
    yaml.dump(data, f)