#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Arc line(linear arc motion)
"""

import os
import sys
import time
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# from xarm.wrapper import XArmAPI
import Xarm as x

#######################################################
# """
# Just for test example
# """
# if len(sys.argv) >= 2:
#     ip = sys.argv[1]
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
########################################################
def normalize(vec):
    n = np.linalg.norm((vec))
    if n == 0:
        return vec
    return vec / n


def get_sphere_points(r, center):
    phi = np.linspace(0, np.pi, 40)
    theta = np.linspace(-np.pi/2, np.pi/2, 40)
    x = center[0] + r*np.outer(np.sin(theta), np.cos(phi))
    y = center[1] + r*np.outer(np.sin(theta), np.sin(phi))
    z = center[2] + r*np.outer(np.cos(theta), np.ones_like(phi))
    return x,y,z


def draw_circle_3d(start_point, end_point, radius):
    direction = normalize(end_point - start_point)
    zeros_in_direction = np.where(direction == 0)[0]
    vec1, vec2 = np.zeros_like(direction), np.zeros_like(direction)
    if len(zeros_in_direction) == 2:
        vec1[zeros_in_direction[0]] = 1
        vec2[zeros_in_direction[1]] = 1
    else:

        vec1 = normalize(np.array([0,0,-direction[2]])) if 2 not in zeros_in_direction else normalize(np.array([-direction[0],0,0]))
        vec2 = normalize(np.cross(direction, vec1))
    try:
        assert np.dot(direction, vec1) == np.dot(vec1, vec2) == np.dot(direction, vec2) == 0
    except AssertionError:
        pass

        # print("spanning vector are not perpendicular. dot is ", np.dot(direction, vec1), np.dot(vec1, vec2), np.dot(direction, vec2))
    x_ = np.linspace(start_point[0], end_point[0], 30)
    y_ = np.linspace(start_point[1], end_point[1], 30)
    z_ = np.linspace(start_point[2], end_point[2], 30)
    x, y, z = np.meshgrid(x_, y_, z_)
    mesh = np.array((x,y,z))
    theta = np.linspace(np.pi, -np.pi, 30)[...,None]
    circle = start_point + radius*np.cos(theta)*(vec1[None,...]) + radius*np.sin(theta)*(vec2[None, ...])
    return vec1, vec2, circle


def send_position(arm, relative):
    pos = arm.get_position()
    arm.move(np.array(pos[:3]) + np.array(relative[:3]), np.array(pos[3:]) + np.array(relative[3:]))

def send_servos(arm, relative):
    servos = arm.get_servos()
    arm.move_servos(np.array(servos[:6]) + np.array(relative))

arm = x.Xarm()
arm.configure_motion()
arm.reset()
arm.staging()
send_servos(arm, [np.deg2rad(35),0,0,0,0,0])
send_servos(arm, [0,np.deg2rad(30),-np.deg2rad(25),0,0,0])
send_servos(arm, [0,0,np.deg2rad(25),0,0,0])
send_servos(arm, [0,0,0,np.deg2rad(75),np.deg2rad(10),0])
send_servos(arm, [-np.deg2rad(10),0,0,0,0,0])
for i in range(10):
    if not i%5:
        angle5 = np.random.randint(-10,10)
        angle6 = np.random.randint(-90, 90)
        angles = [0,0,0,0,np.deg2rad(angle5), np.deg2rad(angle6)]
        send_servos(arm, angles)
    else:
        move = (np.random.rand(3) - 0.5)*30
        move = np.concatenate((move, np.zeros(3)))
        print(move)
        send_position(arm, move)




# send_position(arm, [20,10,10, 0,0,np.pi/8])
# pos = arm.get_position()[1]
# print(pos)
#
# start = np.array([pos[0] + 80, pos[1], pos[2]])
# end = np.array([pos[0] + 80, pos[1], pos[2] + 20])
# vec1, vec2, points_on_circle = draw_circle_3d(start, end, 80)
# arm.move_arc_line(points_on_circle[:8, :])
# arm.staging()
# arm.move_circle([500,500,500],[0,0,0])
#
# # print(points_on_circle)
# current_robot = np.array([2,4,6])
# target = np.array([10,5,2])
#
# r_to_target = np.sqrt(np.sum((current_robot - target)**2))/2
# print(r_to_target)
# x,y,z = get_sphere_points(r_to_target,target)
# sphere_arr = np.array([x.flatten(),y.flatten(),z.flatten()]).T
# distance_to_robot = np.sqrt(np.sum((sphere_arr - current_robot[None,...])**2, axis=1))
# min_point = np.argmin(distance_to_robot)
# mid_point = sphere_arr[min_point]
# p_arr = np.hstack((mid_point, target))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x,y,z, alpha=0.05)
# ax.scatter(current_robot[0], current_robot[1], current_robot[2], color='red')
# ax.scatter(sphere_arr[min_point,0], sphere_arr[min_point,1], sphere_arr[min_point,2], color='black')
# ax.scatter(target[0], target[1], target[2], color='green')
#
#
# # ax.scatter(points_on_circle[4, 0],points_on_circle[4, 1],points_on_circle[4, 2], color='red')
# # ax.scatter(pos[0], pos[1], pos[2], color='red')
# plt.show()


# arm = XArmAPI('192.168.1.225', is_radian=True)
# arm.motion_enable(enable=True)
# arm.set_mode(0)
# arm.set_state(state=0)
#
# arm.reset(wait=True)
#
#
# for i in np.linspace(0, np.pi, 100):
#     arm.set_servo_angle(servo_id=6, angle=i, is_radian=True, speed=5)
#     time.sleep(0.001)
#     angle = arm.get_servo_angle()
#     print(f'angles are {angle}')
#
# arm.reset(wait=True)
# arm.disconnect()