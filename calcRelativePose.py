'''

Calculates relative pose between two cameras 
Given T_cam1_world, T_cam2_world, solve T_cam1_cam2
'''
import numpy as np
import yaml
from utils import yaml_load
np.set_printoptions(precision = 4, suppress = True)



def main(args): 
   
    file1 = f'ARUCO_TWOCAMS_v{VERSION}/CAM1_T_v{VERSION}.yaml'
    file2 = f'ARUCO_TWOCAMS_v{VERSION}/CAM2_T_v{VERSION}.yaml'

    Tcam1_world = yaml_load(file1, 'Tcam1_world')
    Tcam2_world = yaml_load(file2, 'Tcam2_world')
    Tcam1_cam2 = np.linalg.inv(Tcam2_world) @ Tcam1_world 


    data = {'Tcam1_cam2': np.asarray(Tcam1_cam2).tolist()}
    with open(f"ARUCO_TWOCAMS_v{VERSION}/Tcam1_cam2_v{VERSION}.yaml", "w") as f:
        yaml.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculates relative pose between two cameras ")
    parser.add_argument("-file_left", help="left file")
    parser.add_argument("-file_right", help="right file")
    args = parser.parse_args()
    main(args)
    