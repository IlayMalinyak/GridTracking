# __author:IlayK
# data:30/03/2022
import numpy as np
from sksurgerycore.algorithms.procrustes import orthogonal_procrustes
import GuidanceTracker
import time
import Xarm
from scipy.spatial.transform import Rotation as rot
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from algorithms import procrustes
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes


BASE_POS = [234.5,105.4,70,-3.14159,0,0]
# SENSORS_OFFSET_IMAGE = [[83, -51.96, 32], [-25, 43.04, 15], [85, 13.04, 12]]
ORIGIN_IMAGE = [10, 20, 30]
IMAGE_SENSOR_COORDS = [[-6,-24.1,0], [20, 29, 0], [-38.8,19.9,-4]]
IMAGE_OFFSETS = [[0,0,0], [-8,0,0],[-8,4*np.sqrt(3),0]]
ROBOT_OFFSETS = [[-31, 0, -70], [-23, 0, -70], [-23, -4*np.sqrt(3), -70]]
G_TRUTH_ROBOT = [492.7, 27.6, -65.2]
G_TRUTH_ROBOT_ANGLE = [-np.pi, 0, -np.deg2rad(32.6)]

A1_offset = [-82, -51.96,3]
E1_deep_offset = [-66, -51.96, -20]
C6_middle_offset = [-74, -34.64 ,-10]

TEST_POINTS_IMAGE = [[10,20,30],[26,20,-20],[18,37.32,-10]]

TRUE_POINTS = [[ 132.94072266 , -16.51992187 ,-106.26328125],
 [ 132.82910156,  -8.81806641, -106.48652344],
 [ 139.63798828,   -8.81806641, -105.92841797],
 [ 139.41474609,   -0.55810547, -106.26328125],
 [ 146.446875,     -0.66972656, -105.81679687],
 [ 146.33525391,    7.47861328, -106.04003906],
 [ 153.25576172,   7.36699219, -105.70517578],
 [ 153.25576172,   15.29208984, -105.92841797],
 [ 160.17626953,   15.29208984, -105.3703125 ],
 [ 160.06464844,   23.44042969, -105.70517578],
 [ 166.98515625,   23.44042969, -105.25869141],
 [ 166.98515625,   31.47714844, -105.59355469],
 [ 173.90566406,   31.70039062, -105.14707031],
 [ 173.79404297,  39.73710937, -105.3703125 ],
 [ 180.93779297,   39.96035156, -104.92382812],
 [ 184.28642578,   42.19277344, -104.92382812]]





# def get_robot_sensor_in_image(arm):
#     p2 = get_robot_test_points(arm)
#     p1 = np.array(TEST_POINTS_IMAGE)
#     d, z_patient, tdict_patient = procrustes(p1, p2[:,:3])
#     R_image, b_image, T_image = tdict_patient["rotation"], tdict_patient["scale"], tdict_patient["translation"]
#     sensors1_robot = [-A1_offset[0],A1_offset[1],A1_offset[2]]
#
#     sensors_image = b_image*(sensors_robot[:,:3] @ R_image) + T_image
#     return sensors_image


def get_transformation_mat(fixed, moving):
    tdict = {}
    R, T, fre = orthogonal_procrustes(fixed, moving)
    b = 1
    tdict["rotation"], tdict["scale"], tdict["translation"] = R, b, T
    return fre, None, tdict

def get_homogenous_point(point):
    return np.vstack((point, np.ones((1, point.shape[1]))))


def track(tracker):
    tracker.trackSensors()
    return record_to_np(tracker.getRecords())


def record_to_np(record):
    # TODO switch to zeros when we will have 6 sensors
    arr = np.zeros((len(record), 6))
    for i in range(len(record)):
        arr[i, 0] = record[i].x
        arr[i, 1] = record[i].y
        arr[i, 2] = record[i].z
        arr[i, 3] = record[i].r
        arr[i, 4] = record[i].e
        arr[i, 5] = record[i].a
    return arr


def add_points(sensors_arr):
    pts = np.zeros((0, 3))
    for i in range(len(sensors_arr)):
        r = rot.from_euler('xyz', sensors_arr[i, 3:], degrees=True).as_matrix()
        direction_vec = (r @ np.array([[1],[0],[0]])).squeeze()
        forward = sensors_arr[i, :3] + np.array([i*direction_vec for i in range(1,5)])
        backward = sensors_arr[i, :3] - np.array([i*direction_vec for i in range(1,5)])
        pts = np.vstack((pts, forward, backward))
    return pts


def track_and_average(tracker, t_end):
    arr = []
    while time.time() < t_end:
        tracker.trackSensors()
        record = tracker.getRecords()
        arr.append(record_to_np(record))
    arr = np.array(arr)
    average_arr = np.mean(arr, axis=0)
    return average_arr

def random_position():
    position = (np.random.rand(1,3) - 0.5) * 20
    angles = np.zeros((1,3))
    return position, angles

def random_servos():
    angle5 = np.random.randint(-10, 10)
    angle6 = np.random.randint(-90, 90)
    angles = [0, 0, 0, 0, np.deg2rad(angle5), np.deg2rad(angle6)]
    return angles

def send_position(arm, relative):
    pos = arm.get_position()
    arm.move(np.array(pos[:3]) + np.array(relative[:3]), np.array(pos[3:]) + np.array(relative[3:]))

def send_servos(arm, relative):
    servos = arm.get_servos()
    arm.move_servos(np.array(servos[:6]) + np.array(relative))


def test_servo_move(arm, tracker, move, last, err, moves, moves_tracking, tdict=None):
    send_servos(arm, move)
    fixed, mid_points = arm.get_sensors_position()
    moves.append(fixed[0] - last[0])
    e, moves_tracking = get_error(tracker, fixed, mid_points, procrustes, moves_tracking)
    err.append(e)
    return fixed, err, moves, moves_tracking


def test_initial_movements(arm, tracker, err, moves, moves_tracking, tdict=None, moving_init=None):
    arm.reset()
    fixed, mid_points = arm.get_sensors_position()
    last = fixed[:,:]
    e, moves_tracking = get_error(tracker,fixed,mid_points, procrustes, [])
    err.append(e)
    moves.append(fixed[0] - last[0])

    arm.staging()
    fixed, mid_points = arm.get_sensors_position()
    moves.append(fixed[0] - last[0])
    last = fixed[:,:]
    e, moves_tracking = get_error(tracker, fixed, mid_points, procrustes, moves_tracking)
    err.append(e)
    # if tdict is not None:
    #     err_fixed.append(get_error_fixed_transformation(tracker, fixed, tdict, moving_init))

    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [np.deg2rad(35), 0, 0, 0, 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [0, np.deg2rad(30), -np.deg2rad(25), 0, 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [0, 0, np.deg2rad(25), 0, 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [0, 0, 0, np.deg2rad(75), np.deg2rad(10), 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [-np.deg2rad(10), 0, 0, 0, 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [0, 0, 0, -np.deg2rad(30), 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [np.deg2rad(30), 0, 0, 0, 0, 0], last, err, moves, moves_tracking, tdict)
    last = fixed[:, :]
    fixed, err, moves, moves_tracking = test_servo_move(arm, tracker, [0, np.deg2rad(35), -np.deg2rad(40), np.deg2rad(40), 0, 0], last, err, moves, moves_tracking, tdict)

    return err, moves, moves_tracking

# def get_error_fixed_transformation(tracker, fixed, tdict):
#     moving = track(tracker)[:3, :3]
#     t = moving - moving_init
#     R, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]
#     z = b * (moving_init @ R) + T
#     z = z + (b * (t @ R))
#     return np.sqrt(np.sum((fixed[0, :3] - z[0]) ** 2))


def get_error(tracker, fixed,mid, func, moves):
    fixed = np.vstack((fixed[:,:3], mid))
    # fixed = fixed[:,:3]
    moving = track(tracker)
    moves.append(moving[:,:3])
    mid_points = add_points(moving)
    moving = np.vstack((moving[:,:3], mid_points))
    # moving = moving[:,:3]
    d, Z, tdict = func(fixed, moving)
    R, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]
    # print(f"b - {b}")
    z = b * (moving @ R) + T
    return np.sqrt(np.sum((fixed[0, :3] - z[0]) ** 2)), moves


def test_image_patient(image_sensor_points):
    tracker = GuidanceTracker.GuidanceTracker()
    record_arr = track(tracker)
    fixed = record_arr[:3,:3]
    moving = image_sensor_points
    d, z, tdict = procrustes(fixed, moving)
    R, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]
    z = b * (moving @ R) + T
    errors = np.sqrt(np.sum((fixed - z) ** 2, axis=1))
    return errors


def test_patient_robot(iter):
    tracker = GuidanceTracker.GuidanceTracker()
    arm = Xarm.Xarm()
    arm.configure_motion()
    # arm.reset()
    record_arr = track(tracker)
    fixed, mid_points = arm.get_sensors_position()
    moving = record_arr[:3, :3]
    d, z, tdict = procrustes(fixed[:, :3], moving)
    errors = np.sqrt(np.sum((fixed[:,:3] - z) ** 2, axis=1))
    return errors
    # R, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]

    # fre,_, patient_robot_fre = get_transformation_mat(fixed[:, :3], moving)
    err, moves, moves_tracking = test_initial_movements(arm, tracker, [], [], [], tdict, moving)
    for i in range(iter):
        fixed, mid_points = arm.get_sensors_position()
        # if not i % 5:
        #     angles = random_servos()
        #     send_servos(arm, angles)
        #     move = fixed[0] - arm.get_sensors_position()[0]
        # else:
        position, angles = random_position()
        # angles = np.array(arm.get_position()[3:])[None,:]
        move = np.concatenate((position[0,:], angles[0,:]))
        send_position(arm, move)
        last = arm.get_last_position()
        moves.append(move)
        e, moves_tracking = get_error(tracker, fixed, mid_points, procrustes, moves_tracking)
        err.append(e)
    # arm.reset()
    return np.array(err), np.array(moves), np.array(moves_tracking)


def cuboid_data(center, size):
    """
       Create a data array for cuboid plotting.
       center        center of the cuboid, triple
       size          size of the cuboid, triple, (x_length,y_width,z_height)
       :type size: tuple, numpy.array, list
       :param size: size of the cuboid, triple, (x_length,y_width,z_height)
       :type center: tuple, numpy.array, list
       :param center: center of the cuboid, triple, (x,y,z)
      """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x, y, z


def plot_errors(err, moves, err_fixed=None, save=False):
    plt.scatter(range(len(err)), err)
    if err_fixed is not None:
        plt.scatter(range(len(err_fixed)), err_fixed)
    plt.hlines(np.mean(err), xmin=0, xmax=len(err), label="average - {:.2f}".format(np.mean(err)), color='red',
               linestyle='dashed')
    plt.hlines(np.mean(err), xmin=0, xmax=len(err), label="average small movements - {:.2f}".format(np.mean(err[10:])),
               color='magenta', linestyle='dashed')
    plt.xlabel("# movement")
    plt.ylabel("RMS Error (mm)")
    plt.title(f"Patient To Robot transformation error")
    plt.xticks(np.arange(0, len(err), 5))
    plt.legend()
    if save:
        plt.savefig('tests/patient_robot.jpg')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cube_x = np.linspace(200, 500, 10).astype(np.int16)
    cube_y = np.linspace(-230,230, 10).astype(np.int16)
    cube_z = np.linspace(-150,150, 10).astype(np.int16)
    X,Y,Z = cuboid_data((350,0,0),(300, 460, 300))
    grid = np.meshgrid(cube_x,cube_y,cube_z)
    cube = np.hstack((cube_x[:,None], cube_y[:,None], cube_z[:,None]))
    # colors = np.empty(cube + [4], dtype=np.float32)
    # colors[:] = [1, 0, 0, alpha]  # red
    ax.plot_surface(X, Y, np.array(Z), color='r', alpha=0.05)
    cs = ax.scatter(moves[:,0, 0], moves[:,0, 1], -moves[:,0, 2], label="sensor 1", s=(err[:]+1)**2)
    # cs = ax.scatter(moves[:,1, 0], moves[:,1, 1], -moves[:,1, 2], label="2")
    # cs = ax.scatter(moves[:,2, 0], moves[:,2, 1], -moves[:,2, 2], label="3")
    plt.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Patient To Robot transformation error vs grid position")
    if save:
        plt.savefig('tests/patient_robot_3d.jpg')
    plt.show()

def get_robot_test_points(arm):
    offsets = np.array([A1_offset,E1_deep_offset,C6_middle_offset])
    pos = np.repeat(np.array(arm.get_position())[:, None].T, 3, axis=0)
    rot_mat = R.from_euler('xyz', [pos[0, 3], pos[0, 4], pos[0, 5]]).as_matrix()
    rotated_offsets = rot_mat @ np.array(offsets).T
    pos[0, :3] += np.array(rotated_offsets[:, 0])
    pos[1, :3] += np.array(rotated_offsets[:, 1])
    pos[2, :3] += np.array(rotated_offsets[:, 2])
    return pos, rotated_offsets


def full_test_3_sensors():
    tracker = GuidanceTracker.GuidanceTracker()
    arm = Xarm.Xarm()
    arm.configure_motion()
    img_sensors = np.array(ORIGIN_IMAGE)[None,:] + np.array(IMAGE_SENSOR_COORDS)
    ref_points_robot, offsets_robot = get_robot_test_points(arm)
    ref_points_image = np.array(TEST_POINTS_IMAGE)
    # arm.rotate([0,0,np.pi])
    record_arr = track(tracker)
    fixed_patient = record_arr[:3, :3]
    moving_patient = img_sensors
    d, z_patient, tdict_patient = procrustes(fixed_patient, moving_patient)
    R_image, b_image, T_image = tdict_patient["rotation"], tdict_patient["scale"], tdict_patient["translation"]

    image_points_patient = b_image*(ref_points_image @ R_image) + T_image

    input("Press Enter to continue...")
    print('continue')
    # arm.reset()
    record_arr = track(tracker)
    fixed_robot, mid_points = arm.get_sensors_position()
    moving_robot = record_arr[:3, :3]
    robot_angles = record_arr[:3,3:]
    print(robot_angles)
    robot_angles = np.deg2rad(robot_angles)
    d, robot_patient, tdict_robot = procrustes(fixed_robot[:, :3], moving_robot)
    R_robot, b_robot, T_robot = tdict_robot["rotation"], tdict_robot["scale"], tdict_robot["translation"]

    d_inv, robot_patient_inv, tdict_robot_inv = procrustes(moving_robot, fixed_robot[:, :3])
    R_robot_inv, b_robot_inv, T_robot_inv = tdict_robot_inv["rotation"], tdict_robot_inv["scale"], tdict_robot_inv["translation"]

    offsets = np.array([A1_offset,E1_deep_offset,C6_middle_offset])
    ref_points_robot = fixed_robot[:,:3] + offsets
    ref_points_patient_inv = b_robot_inv*(ref_points_robot[:,:3] @ R_robot_inv) + T_robot_inv
    ref_points_patient = (ref_points_robot[:,:3] @ np.linalg.inv(R_robot))/b_robot - (T_robot @ np.linalg.inv(R_robot))/b_robot
    offsets_robot_inv = (offsets @ R_robot_inv)/b_robot_inv
    offsets_patient = (offsets @ R_robot.T)/b_robot

    target_patient = image_points_patient - offsets_patient

    target_robot = b_robot*(target_patient @ R_robot) + T_robot

    return target_robot


def create_grid_offsets():
    grid_locations = ['o16', 'o14', 'm14', 'm12', 'k12', 'k10', 'i10', 'i8', 'g8', 'g6', 'e6', 'e4', 'c4', 'c2', 'a2', 'A1']
    base_grid_offsets = np.zeros((len(grid_locations), 3))
    base_grid_offsets[0] = np.array([-23, 0, 0])
    for i in range(1, len(base_grid_offsets) - 1):
        if i % 2:
            base_grid_offsets[i] = base_grid_offsets[i -1] + [-8,0,0]
        else:
            base_grid_offsets[i] = base_grid_offsets[i - 1] + [0, -4*np.sqrt(3), 0]
    base_grid_offsets[-1] = base_grid_offsets[-2] + [-4, -2*np.sqrt(3),0]
    return base_grid_offsets, grid_locations


def create_true_points(length, tracker):
    true_points = np.zeros((length,3))
    for i in range(length):
        input('hit enter when you are ready')
        record_arr = track(tracker)
        true_points[i] = record_arr[0,:3]
    input("return sensor and hit enter")
    return true_points

def offset_error():
    tracker = GuidanceTracker.GuidanceTracker()
    base_offsets, gris_locations = create_grid_offsets()
    # true_points = create_true_points(len(base_offsets), tracker)
    true_points = np.array(TRUE_POINTS)
    arm = Xarm.Xarm()
    arm.configure_motion()
    print(true_points)
    rotated_offsets = rotate_offsets(arm, base_offsets)

    record_arr = track(tracker)
    patient = record_arr[:3, :3]
    mid_patient = add_points(record_arr)
    patient = np.concatenate((patient, mid_patient))
    robot,mid = arm.get_sensors_position()
    robot = np.concatenate((robot[:,:3], mid))
    offset_postions = robot[0] + rotated_offsets
    d, z, tdict = procrustes(patient, robot)
    r, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]
    base_error = np.sqrt(np.sum((patient - z) ** 2, axis=1))
    print('base error is ', base_error)

    dist_from_sensor = np.sqrt(np.sum(rotated_offsets**2, axis=1))
    dist_from_sensor_2 = np.sqrt(np.sum(base_offsets**2, axis=1))
    print('checking unitary :\n', dist_from_sensor - dist_from_sensor_2)

    patient_hat = b*(offset_postions @ r) + T
    diff = np.abs(patient_hat - true_points)
    plt.scatter(dist_from_sensor, diff[:,0], label='x')
    plt.scatter(dist_from_sensor, diff[:,1], label='y')
    plt.scatter(dist_from_sensor, diff[:,2], label='z')
    plt.plot(dist_from_sensor, np.sqrt(np.sum(diff, axis=1)),  label='sum sqaured distance')
    plt.title('coordinates error vs distance from registration sensor')
    plt.hlines(np.mean(base_error), dist_from_sensor[0], dist_from_sensor[-1], color='black', label=
        'Average registration error: {:.2f}'.format(np.mean(base_error)))
    plt.legend()
    plt.xlabel("distance from registration sensor")
    plt.ylabel("coordinate difference (mm)")
    # plt.savefig('./tests/error_vs_offset_dist_7.png')
    plt.show()


def rotate_offsets(arm, base_offsets):
    pos = np.repeat(np.array(arm.get_position())[:, None].T, 3, axis=0)
    rot_mat = R.from_euler('xyz', [pos[0, 3], pos[0, 4], pos[0, 5]]).as_matrix()
    rotated_offsets = (rot_mat @ np.array(base_offsets).T).T
    return rotated_offsets


def move_step(arm, current, step):
    arm.move([current[0] + step[0], current[1] + step[1], current[2] + step[2]], [current[3], current[4], current[5]])


def test_cube(iter):
    arm = Xarm.Xarm()
    arm.configure_motion()
    tracker = GuidanceTracker.GuidanceTracker()

    img_sensors = np.array(ORIGIN_IMAGE) + np.array(IMAGE_SENSOR_COORDS)
    tracked_image = track(tracker)
    d_img, z_img, tdict_img = procrustes(tracked_image[:3,:3], img_sensors)
    r_img, b_img, T_img = tdict_img["rotation"], tdict_img["scale"], tdict_img["translation"]

    img_registration_error = np.mean(np.sqrt(np.sum((z_img - tracked_image[:3,:3])**2, axis=1)))

    img_grid_points = np.array(ORIGIN_IMAGE) + np.array(IMAGE_OFFSETS)
    img_grid_patient = b_img*(img_grid_points @ r_img) + T_img
    # img_grid_patient[:,2] *= -1

    robot_points = np.zeros((iter,3,3))
    patient_points = np.zeros((iter,3,3))
    target_at_robot = np.zeros((iter,3,3))

    robot_sensor_1 = np.zeros((iter, 3))
    robot_registration_errors = np.zeros((iter,1))
    input("move sensors and press enter")
    i = 0
    while i < iter:
        # input("move robot and press enter")

        robot_sensors,_ = arm.get_sensors_position()
        robot_sensor_1[i,:] = robot_sensors[0, :3]
        tracked_robot = track(tracker)
        d_robot, z_robot, tdict_robot = procrustes(tracked_robot[:3, :3], robot_sensors[:3,:3])
        r_robot, b_robot, T_robot = tdict_robot["rotation"], tdict_robot["scale"], tdict_robot["translation"]
        robot_registration_errors[i] = np.mean(np.sqrt(np.sum((z_robot - tracked_robot[:3,:3])**2, axis=1)))

        rotated_offsets = rotate_offsets(arm, ROBOT_OFFSETS)

        robot_grid_points = robot_sensors[0,:3] + rotated_offsets
        robot_grid_patient = b_robot*(robot_grid_points @ r_robot) + T_robot
        robot_points[i,...] = robot_grid_points
        patient_points[i,...] = robot_grid_patient

        img_grid_robot = (img_grid_patient @ r_robot.T - T_robot @ r_robot.T) / b_robot
        target_at_robot[i,...] = img_grid_robot

        print("moving one step")
        # move_step(arm, robot_sensors[0], [1,1,-1])
        send_servos(arm, [0,0,0,0,0,-np.deg2rad(2)])
        i+=1

    diff_patient = img_grid_patient[None,...] - patient_points
    diff_robot = target_at_robot - robot_points
    g_truth = np.array(G_TRUTH_ROBOT)
    robot_real_dists = np.sqrt(np.sum((g_truth[None,:] - robot_sensor_1)**2,axis=1))
    dist_patient = np.mean(np.sqrt(np.sum(diff_patient**2, axis=2)),axis=1)
    dist_robot = np.mean(np.sqrt(np.sum(diff_robot**2, axis=2)),axis=1)
    mean_diff_patient = np.mean(diff_patient, axis=1)
    mean_diff_robot = np.mean(diff_robot, axis=1)

    plt.scatter(robot_real_dists, dist_patient, label='NDI coordinate system')
    plt.scatter(robot_real_dists, dist_robot, label='Robot coordinate system')
    # plt.scatter(robot_real_dists, robot_registration_errors, label='Robot registration error')
    plt.hlines(img_registration_error, robot_real_dists[0], robot_real_dists[-1], color='black', label=
    'Image registration error: {:.2f}'.format(img_registration_error))
    # plt.xticks(robot_real_dists[::-1])
    plt.xlabel("Real distance from target (mm)")
    plt.ylabel("Mean Squared Error (mm)")
    plt.title("Target Errors")
    plt.legend(loc='center left')
    plt.savefig('./tests/cube/target_error_dists.png')
    plt.show()

    handles = []
    markers = ["o", "^"]
    colors = ["orange", "blue", "green"]
    plt.scatter(robot_real_dists, mean_diff_patient[:,0], marker="o", color='orange')
    plt.scatter(robot_real_dists, mean_diff_patient[:,1], marker="o", color='blue')
    plt.scatter(robot_real_dists, mean_diff_patient[:,2], marker="o", color='green')

    plt.scatter(robot_real_dists, mean_diff_robot[:, 0], marker="^", color='orange')
    plt.scatter(robot_real_dists, mean_diff_robot[:, 1], marker="^", color='blue')
    plt.scatter(robot_real_dists, mean_diff_robot[:, 2], marker="^", color='green')

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(3)]
    handles += [f(markers[i], "k") for i in range(2)]

    labels = ['x','y','z', "NDI", "Robot"]
    plt.legend(handles, labels)
    plt.xlabel("Real distance from target (mm)")
    plt.ylabel("Average diff (mm)")
    plt.title("Coordinates differnece")
    plt.savefig('./tests/cube/target_error_mean_diff')
    plt.show()


def test_cube_rotation(iter):
    arm = Xarm.Xarm()
    arm.configure_motion()
    tracker = GuidanceTracker.GuidanceTracker()

    img_sensors = np.array(ORIGIN_IMAGE) + np.array(IMAGE_SENSOR_COORDS)
    tracked_image = track(tracker)
    d_img, z_img, tdict_img = procrustes(tracked_image[:3,:3], img_sensors)
    r_img, b_img, T_img = tdict_img["rotation"], tdict_img["scale"], tdict_img["translation"]

    img_registration_error = np.mean(np.sqrt(np.sum((z_img - tracked_image[:3,:3])**2, axis=1)))

    img_grid_points = np.array(ORIGIN_IMAGE) + np.array(IMAGE_OFFSETS)
    img_grid_patient = b_img*(img_grid_points @ r_img) + T_img
    # img_grid_patient[:,2] *= -1

    robot_angles = np.zeros((iter,3))
    patient_angles = np.zeros((iter,3))
    target_at_robot = np.zeros((iter,3,3))

    robot_real_angles = np.zeros((iter,3))
    robot_registration_errors = np.zeros((iter,1))
    input("move sensors and press enter")
    i = 0
    while i < iter:
        # input("move robot and press enter")

        robot_sensors,_ = arm.get_sensors_position()
        robot_real_angles[i,:] = np.rad2deg(G_TRUTH_ROBOT_ANGLE - robot_sensors[0, 3:])
        tracked_robot = track(tracker)
        d_robot, z_robot, tdict_robot = procrustes(tracked_robot[:3, :3], robot_sensors[:3,:3])
        r_robot, b_robot, T_robot = tdict_robot["rotation"], tdict_robot["scale"], tdict_robot["translation"]
        robot_registration_errors[i] = np.mean(np.sqrt(np.sum((z_robot - tracked_robot[:3,:3])**2, axis=1)))

        rotated_offsets = rotate_offsets(arm, ROBOT_OFFSETS)

        robot_grid_points = robot_sensors[0,:3] + rotated_offsets
        robot_grid_patient = b_robot*(robot_grid_points @ r_robot) + T_robot


        alignment_data_patient, rmsd = R.align_vectors(img_grid_patient, robot_grid_patient)
        patient_angles[i] = np.rad2deg(alignment_data_patient.as_euler('xyz') % np.pi)


        # robot_angles[i,...] = robot_grid_points
        # patient_angles[i,...] = robot_grid_patient

        img_grid_robot = (img_grid_patient @ r_robot.T - T_robot @ r_robot.T) / b_robot
        # target_at_robot[i,...] = img_grid_robot

        alignment_data_robot, rmsd = R.align_vectors(img_grid_robot, robot_grid_points)
        robot_angles[i] = np.rad2deg(alignment_data_robot.as_euler('zyx') % np.pi)
        angles = np.rad2deg(get_rotation_angles(img_grid_robot, robot_grid_points))

        print("moving one step")
        # move_step(arm, robot_sensors[0], [1,1,-1])
        send_servos(arm, [0,0,0,0,0,-np.deg2rad(2)])
        i+=1

    # patient_angles[:,:2] = 180 - patient_angles[:,:2]
    # robot_angles[:, [2,0]] = robot_angles[:,[0,2]]
    diff_patient = robot_real_angles - patient_angles
    flip_idx = np.where(robot_angles > 100)
    robot_angles[flip_idx] = 180 - robot_angles[flip_idx]
    diff_robot = np.abs(robot_real_angles - robot_angles)
    # flip_idx = np.where(diff_robot > 100)
    # diff_robot[flip_idx] = 180 - diff_robot[flip_idx]
    # diff_robot = (180 - diff_robot) % 180

    plt.scatter(robot_real_angles[:, 2], diff_robot[:, 0], label='x')
    plt.scatter(robot_real_angles[:,2], diff_robot[:,1], label='y')
    plt.scatter(robot_real_angles[:,2], diff_robot[:,2], label='z')
    plt.legend()
    plt.xlabel("Real Z angle from target (mm)")
    plt.ylabel("Angle diff (degrees)")
    plt.title("Angle difference")
    plt.savefig('./tests/cube/target_error_angles_diff')
    plt.show()


def get_rotation_angles(p1, p2):
    d, z, tdict = procrustes(p1, p2)
    r = tdict["rotation"]
    angles = R.from_matrix(r).as_euler('xyz')
    return angles

test_cube_rotation(20)

    # robot_grid_patient[:, 2] *= -1

    # print(robot_grid_patient)

    # alignment_data, rmsd = R.align_vectors(img_grid_patient, robot_grid_patient)
    # alignment_angles = alignment_data.as_euler('xyz') % np.pi

    # alignment_angles_robot = alignment_angles @ r_robot.T

    # target_coords_robot[:,2] *= -1

    # current_position = arm.get_position()
    # target_angles_robot = np.array(current_position[3:]) + alignment_angles_robot
    # print("targets ", target_coords_robot, target_angles_robot)
    # arm.move(target_coords_robot[0], target_angles_robot)






# test_cube()
# img_sensors = np.array(ORIGIN_IMAGE)+ np.array(IMAGE_SENSOR_COORDS)
# print("patient robot error" , test_patient_robot(0))
# print("image_patietn error", test_image_patient(img_sensors))


# offset_error()
# full_test_3_sensors()
    # target_transformed = b_robot*(offset_patient[0] @ R_robot) + T_robot
    # return target_transformed

    # grid_patient_high = grid_patient
    # grid_patient_high[:,2] = moving_robot[:,2]
    # _, z_final, tdict_final = procrustes(grid_patient, moving_robot)
    # R_align_robot = tdict_final['rotation']
    # alignment_data, rmsd = R.align_vectors(grid_patient, moving_robot)
    # R_align_robot_scipy = alignment_data.as_matrix()
    # alignment_angles = alignment_data.as_euler('xyz')
    # # new_angles = robot_angles + alignment_angles
    # distance = grid_patient - moving_robot

    # angles_transformed = b_robot*(alignment_angles @ R_robot)
    # robot_position = np.array(arm.get_position())
    # new_angles = robot_position[3:] + angles_transformed

    # print(target_transformed, new_angles)
    # arm.move_circle(moving_robot, )
    # arm.move(target_transformed, robot_position[3:])


# tracker = GuidanceTracker.GuidanceTracker()
# arm = Xarm.Xarm()
# arm.configure_motion()
# pos,_ = arm.get_sensors_position()
# offsets = np.array([A1_offset, E1_deep_offset, C6_middle_offset])
# ref_points = pos[0,:3] + offsets
#
# record_arr = track(tracker)
# robot_patient = record_arr[:3,:3]
# d, z, tdict_robot = procrustes(pos[:,:3], robot_patient)
# R_robot, b_robot, T_robot = tdict_robot["rotation"], tdict_robot["scale"], tdict_robot["translation"]
# R_2, b_2 = orthogonal_procrustes(robot_patient, pos[:,:3])
#
# offsets_rotated = (offsets @ np.linalg.inv(R_robot))/b_robot
# offsets_rotated_2 = offsets @ np.linalg.inv(R_2)
# ref_points_patient = robot_patient[0,:] + offsets_rotated
# print(ref_points_patient)

# robot_test_points = get_robot_test_points(arm)
# print(robot_test_points)
# full_test_3_sensors()

# start_time = time.time()
# err, moves, moves_tracking = test_patient_robot(30)
# plot_errors(err, moves_tracking[:,:,:])
# print(moves_tracking[0,0,:])
# print(moves_tracking[1,0,:])
# print(moves_tracking[-1,0,:])

# print(np.mean(err), np.mean(err_fixed))

# tracker = GuidanceTracker.GuidanceTracker()
# arm = Xarm.Xarm()
# arm.configure_motion()
# arm.reset()
# arm.reset()
# record_arr = track(tracker)
# print(record_arr)
# added_points = add_points(record_arr)
# fixed, add_fixed = arm.get_sensors_position()
# print(fixed)
# print(added_points)

