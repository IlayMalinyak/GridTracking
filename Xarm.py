# __author:IlayK
# data:16/03/2022
from xarm.wrapper import XArmAPI
import json
from configparser import ConfigParser
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


class Xarm():
    def __init__(self):
        self.arm = None
        self.code_dict = {}
        self.err = ""
        try:
            parser = ConfigParser()
            parser.read('./static/config/robot.conf')
            self.ip = parser.get('xArm', 'ip')
            self.offsets = [json.loads(parser.get('xArm', 'sensor_1_offset')),json.loads(parser.get('xArm', 'sensor_2_offset')),
                            json.loads(parser.get('xArm', 'sensor_3_offset'))]
            self.directions = [json.loads(parser.get('xArm', 'sensor_1_direction')),json.loads(parser.get('xArm', 'sensor_2_direction')),
                            json.loads(parser.get('xArm', 'sensor_3_direction'))]
            self.staging_angles = json.loads(parser.get('xArm', 'staging'))
            self.speed = parser.get('xArm', 'speed')
            self.acc = parser.get('xArm', 'acc')
            with open("static/XarmCodes.txt", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    key_val = line.split(':')
                    self.code_dict[int(key_val[0])] = key_val[1]

        except Exception as e:
            print(e)
            self.offsets = None
            self.staging_angles = None
            self.ip = None
            print('ERROR: Xarm input error. check static/config/robot.conf file')
            exit(1)
        print(f"IP - {self.ip}, offsets - {self.offsets} type ({type(self.offsets)})\n")
        try:
            self.arm = XArmAPI(self.ip, is_radian=True, do_not_open=False)
        except Exception as e:
            print(e)
            self.arm = None

    def get_ip(self):
        return self.ip

    def get_arm(self):
        return self.arm

    def reset(self):
        self.arm.reset(wait=True)

    def configure_motion(self, enable=True, set_mode=0, set_state=0):
        self.arm.motion_enable(enable=enable)
        self.arm.set_mode(set_mode)
        self.arm.set_state(set_state)

    def rotate_end(self):
        print("rotating...")
        for i in np.linspace(0, np.pi, 100):
            self.arm.set_servo_angle(servo_id=6, angle=i, is_radian=True, speed=5)
            time.sleep(0.001)

    def staging(self):
        self.arm.set_mode(0)
        print("moving to staging position...")
        code = self.arm.set_servo_angle(angle=np.deg2rad(np.array(self.staging_angles)), wait=True)
        self.err = self.code_dict[code]
        return code

    def rotate(self, angles):
        pos_code, cur_pos = self.arm.get_position()
        cur_pos = np.array(cur_pos)
        cur_pos[3:] += angles
        print(f"angles  {angles} cure pos {cur_pos}")
        # TODO ********erase when there is real data*******
        code = self.arm.set_position(x=cur_pos[0], y=cur_pos[1], z=cur_pos[2],roll=cur_pos[3], pitch=cur_pos[4], yaw=cur_pos[5]
                              , speed=self.speed, mvacc=self.acc, wait=True, is_radian=True)
        # TODO ********erase when there is real data*******
        # res_code = self.arm.set_position(x=cur_pos[0], y=cur_pos[1], z=cur_pos[2], roll=angles[0], pitch=angles[1],
        #                                  yaw=angles[2]
        #                                  , speed=self.speed, mvacc=self.acc, wait=True, is_radian=True)
        self.err = self.code_dict[code]
        return code

    def move(self, coords, angles):
        # TODO implement move_arc_line
        if self.arm.mode == 0:
            # print("moving in speed ", self.speed)
            code = self.arm.set_position(x=coords[0], y=coords[1], z=coords[2], roll=angles[0], pitch=angles[1], yaw=angles[2],
                                   speed=self.speed, mvacc=self.acc, wait=True, is_radian=True)
        else:
            code = -2
        self.err = self.code_dict[code]
        return code

    def move_circle(self, p1, p2):
        _, cur_pos = self.arm.get_position()
        p1 = np.concatenate((p1, cur_pos[3:]))
        p2 = np.concatenate((p2, cur_pos[3:]))
        # TODO ********erase when there is real data*******
        p1 = np.array([cur_pos[0]+10, cur_pos[1]+10, cur_pos[2]-10, cur_pos[3], cur_pos[4], cur_pos[5]])
        p2 = np.array([cur_pos[0]+20, cur_pos[1]+15, cur_pos[2]-15, cur_pos[3], cur_pos[4], cur_pos[5]])
        # TODO ********erase when there is real data*******

        print(f'p1 shape - {p1.shape} values - {p1}\np2 shape - {p2.shape} values - {p2}')
        print(f"distnace to move {p2 - cur_pos}")
        code = self.arm.move_circle(p1, p2, percent=50, speed=self.speed, is_radian=True, wait=True)
        self.err = self.code_dict[code]
        return code

    def move_arc_line(self, path):
        _, cur_pos = self.arm.get_position()
        cur_pos = np.repeat(np.array(cur_pos[3:])[None,:], len(path), axis=0)
        path = np.hstack((path, cur_pos))
        code = self.arm.move_arc_lines(path)
        self.err = self.code_dict[code]
        return code

    def move_servos(self, angles):
        self.arm.set_servo_angle(angle=angles, is_radian=True, wait=True)

    def get_position(self):
        return self.arm.get_position()[1]

    def get_last_position(self):
        return self.arm.last_used_position

    def get_servos(self):
        return self.arm.get_servo_angle()[1]

    def get_offsets(self):
        return self.offsets

    def get_err(self):
        return self.err

    def get_sensors_position(self):
        pos = np.repeat(np.array(self.arm.get_position()[1])[:,None].T, 3, axis=0)
        rot_mat = R.from_euler('xyz', [pos[0, 3], pos[0,4], pos[0,5]]).as_matrix()
        rotated_offsets = rot_mat @ np.array(self.offsets).T
        rotated_directions = rot_mat @ np.array(self.directions).T
        pos[0, :3] += np.array(rotated_offsets[:,0])
        pos[1, :3] += np.array(rotated_offsets[:,1])
        pos[2, :3] += np.array(rotated_offsets[:,2])
        added_pts = np.zeros((0,3))
        for i in range(len(self.directions)):
            forward = pos[i, :3] + np.array([j * rotated_directions[:,i] for j in range(1, 5)])
            backward = pos[i, :3] - np.array([j * rotated_directions[:,i] for j in range(1, 5)])
            added_pts = np.vstack((added_pts, forward, backward))
        # t = pos[0,:3] - np.array([234.5, 105.4, 70])
        return pos, added_pts

