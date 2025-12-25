import carla
import numpy as np

import cv2

from pathlib import Path
import os
import argparse
import time
import subprocess
import json
import sys
import pickle

CARLA_ROOT=os.environ.get("CARLA_ROOT")

sys.path.append(CARLA_ROOT + "/PythonAPI")
sys.path.append(CARLA_ROOT + "/PythonAPI/carla")

sys.path.append(CARLA_ROOT + "/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
max_dist=0

class LankMarkingGettor(object):
    '''
        structure of lane_marking_dict:
        {
            road_id_0: {
                lane_id_0: [{'Points': [((location.x,y,z) array, (rotation.roll, pitch, yaw))], 'Type': 'lane_marking_type', 'Color':'color', 'Topology':[neighbor array]}, ...]
                ... ...
                'Trigger_Volumes': [{'Points': [(location.x,y,z) array], 'Type': 'trigger volume type', 'ParentActor_Location': (location.x,y,z)}]
            }
            ... ...
        }
        "location array" is an array formed as (location_x, location_y, location_z) ...
        'lane_marking_type' is string of landmarking type, can be 'Broken', 'Solid', 'SolidSolid', 'Other', 'NONE', etc.
        'color' is string of landmarking color, can be 'Blue', 'White', 'Yellow',  etc.
         neighbor array contains the ('road_id', 'lane_id') of the current landmarking adjacent to, it is directional.
         and if current 'Type' == 'Center', there will exist a 'TopologyType' key which record the current lane's topology status.
         if there exist a trigger volume in current road, key 'Trigger_Volumes' will be added into dict
         where 'Points' refer to the vertexs location array, 'Type' can be 'StopSign' or 'TrafficLight'
         'ParentActor_Location' is the location of parent actor relevant to this trigger volume.
    '''

    @staticmethod
    def get_lanemarkings(carla_map,max_dist, lane_marking_dict={}, pixels_per_meter=2, precision=0.05):

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.road_id)

        map_list=[]

        for waypoint in topology:
            waypoints = [waypoint]
            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                temp_wp = nxt
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            # print("current road id: ", waypoint.road_id)
            # print("lane id:", waypoint.lane_id)
            maps=[]
            for waypoint in waypoints:
                w_transform=waypoint.transform
                road_lane_id=waypoint.road_id+waypoint.lane_id*0.001
                maps.append((w_transform.location.x,-w_transform.location.y,waypoint.lane_width*0.5,road_lane_id))

            maps=np.array(maps).astype(np.float32)[::20]

            if len(maps)>1:

                way_dist=np.linalg.norm(maps[1:,:2]-maps[:-1,:2],axis=-1)

                width=np.maximum(maps[1:,2],maps[:-1,2])

                maps[:-1,2]=np.sqrt(way_dist*way_dist/4+width*width)

            map_list.append(maps)

        all_map=np.concatenate(map_list)

        return all_map,max_dist



if __name__ == '__main__':
    map_dict={}
    cmd1 = f"{os.path.join(CARLA_ROOT, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port=20001"
    server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
    print(cmd1, server.returncode, flush=True)
    time.sleep(10)
    client = carla.Client('localhost', 20001)
    client.set_timeout(300)

    for id in ['11','01','02','03','04','05','06','07','10HD','12','13','15']:
        carla_town = 'Town'+id

        world = client.load_world(carla_town)
        print("******** sucessfully load the town:", carla_town, " ********")
        carla_map = world.get_map()

        arr,max_dist = LankMarkingGettor.get_lanemarkings(world.get_map(),max_dist)
        print("****** get all lanemarkings ******")

        map_dict[carla_town[:6]]=arr
        # time.sleep(100)

    with open(os.getenv('NAVSIM_EXP_ROOT') + "/map.pkl", 'wb') as f:
        pickle.dump(map_dict, f)
