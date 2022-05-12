
import GuidanceTracker
import Xarm
import server
import keyboard
import numpy as np
import time
from sksurgerycore.algorithms.procrustes import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R
from algorithms import procrustes


SUCCESS = 0
FAILURE = 1
STATE_ACTIVE = "Active"
STATE_NON_ACTIVE = "Not Active"
STATE_STAGING = "In Staging Position"
STATE_MOVE = "Movement State"
STATE_ALIGNED = "Aligned"
DEFAULT_MASSAGE = "No massages to show"
MODE_TRACKING = "Tracking Mode"
DIRECTION_MASSAGE = "Sending movement directions to Xarm..."
NO_MAT_MASSSAGE = "ERROR: No transformation matrix from patient to robot exist. Did you calibrate?"
NO_TRACK_MASSAGE = "No tracking data. did you start tracking?"
# MOVE_SUCCEED_MASSAGE = "Movement Succeed!"
MIN_RADIUS = 10

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


class GridTracker:
    def __init__(self):
        print("*****3D Guidance******")
        self.Tracker = None
        self.error_msg = ""
        self.Tracker = GuidanceTracker.GuidanceTracker()
        self.Tracker_status = self.Tracker.getTrackingState()
        if self.Tracker_status == STATE_ACTIVE:
            self.Tracker_config = self.Tracker.getSystemConfig()
            print(f"number boards          = {self.Tracker_config.numberBoards}\n")
            print(f"number sensors         = {self.Tracker_config.numberSensors}\n")
            print(f"number transmitters    = {self.Tracker_config.numberTransmitters}\n")
            # print("system agc mode	       = %d\n", self.Tracker_config.agcMode)
            print(f"maximum range          = {self.Tracker_config.maximumRange}\n")
            print(f"measurement rate       = {self.Tracker_config.measurementRate}\n")
            print(f"metric mode            = {self.Tracker_config.metric}\n")
            print(f"line frequency         = {self.Tracker_config.powerLineFrequency}\n")
            print(f"transmitter id running = {self.Tracker_config.transmitterIDRunning}\n")
            print(f"Tracker state is {self.Tracker_status}\n\n")
            print("*****X-arm******")
        else:
            self.Tracker_status = STATE_NON_ACTIVE
            self.Tracker_config=None
            self.error_msg = "Initialization of Tracking system failed - No system found"
        self.Xarm = Xarm.Xarm()
        if self.Xarm.get_arm() is not None:
            self.Xarm_status = STATE_ACTIVE
            self.Xarm.configure_motion()
            self.Xarm.reset()
        else:
            self.Xarm_status = STATE_NON_ACTIVE
            self.error_msg = self.error_msg + "\nInitialization of X-arm robot failed - No system found"
        print(f"Robot state is {self.Xarm_status}\n\n")

        self._record = None  # live data. index 0-3 -> grid, index 3-6 -> patient
        self.image_marker_points = None  # patient markers. should be at the same place as the sensors
        self.virtual_grid_points = None  # virtual grid points. should match sensors (relative) position on the the grid
        self.robot_points = self.Xarm.get_offsets()  # grid sensors in robot reference frame
        self.image_to_patient_mat = None  # image reference frame to patient reference frame transformation matrix
        self.patient_to_robot_mat = None  # patient reference frame to robot reference frame transformation matrix
        self.patient_virtual_grid_mat = None  # patient to virtual grid transformation matrix (patient reference frame)
        self.image_patient_fre = None  # Fiducial Registration Error image to patient transformation
        self.patient_robot_fre = None  # Fiducial Registration Error patient to robot transformation
        self.patient_virtual_grid_fre = None  # Fiducial Registration Error virtual grid transformation matrix
        self.alignment_angles = None  # euler angles between grid and virtual grid (patient reference frame)
        self.target = None  # virtual grid location (patient reference frame)
        self.tracking_mode = None
        self.calibration_mode = None



    @staticmethod
    def run_server():
        server.run()

    @staticmethod
    def record_to_np(record):
        # TODO switch to zeros when we will have 6 sensors
        arr = np.random.randn(len(record), 6)
        for i in range(len(record)):
            arr[i, 0] = record[i].x
            arr[i, 1] = record[i].y
            arr[i, 2] = record[i].z
            arr[i, 3] = record[i].r
            arr[i, 4] = record[i].e
            arr[i, 5] = record[i].a
        return arr

    @staticmethod
    def get_homogenous_point(point):
        return np.vstack((point, np.ones((1, point.shape[1]))))

    # @staticmethod
    # def get_procrustes_transformation_mat(fixed, moving):
    #     return orthogonal_procrustes(fixed, moving)
    #
    @staticmethod
    def get_transformation_mat(fixed, moving):
        R, T, fre = orthogonal_procrustes(fixed, moving)
        M = np.hstack((R, T))
        return M, fre

    @staticmethod
    def get_sphere_points(r, center):
        print(f"center shape {center.shape}")
        phi = np.linspace(0, np.pi, 40)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 40)
        x = center[0] + r * np.outer(np.sin(theta), np.cos(phi))
        y = center[1] + r * np.outer(np.sin(theta), np.sin(phi))
        z = center[2] + r * np.outer(np.cos(theta), np.ones_like(phi))
        return x, y, z

    @staticmethod
    def path_on_sphere(sphere_arr, start):
        distance_to_start = np.sqrt(np.sum((sphere_arr - start[None, ...]) ** 2, axis=1))
        return sphere_arr[np.argmin(distance_to_start)]

    @staticmethod
    def create_random_movement(current_pos):
        position = current_pos[:3] + (np.random.rand(3) - 0.5) * 30
        print(f"current position - {current_pos}. moving to - {position}")
        angles = current_pos[3:]
        return position, angles


    def get_tracker_status(self):
        return self.Tracker_status

    def get_xarm_status(self):
        return self.Xarm_status

    def get_num_active_sensors(self):
        if self.Tracker is not None:
            return self.Tracker.getNumActiveSensors()
        return None

    def get_metric(self):
        if self.Tracker_config is not None:
            return "mm" if self.Tracker_config.metric else "Inch"
        return None

    def get_tracker_config(self):
        return self.Tracker_config

    def get_record(self):
        return self.record_to_np(self._record) if self._record is not None else None

    def get_distance(self):
        if self.target is not None:
            return self.target[0] - self.record_to_np(self._record)[0, :3]
        return None

    def get_alignment(self):
        return self.alignment_angles

    def get_mean_fre(self):
        if self.patient_virtual_grid_fre is not None:
            if self.patient_robot_fre is not None:
                return np.mean([self.patient_robot_fre, self.patient_virtual_grid_fre, self.image_patient_fre])
            return np.mean([self.patient_virtual_grid_fre, self.image_patient_fre])
        return self.image_patient_fre

    def set_image_marker_points(self, points):
        self.image_marker_points = points

    def get_image_marker_points(self):
        return self.image_marker_points

    def set_virtual_grid_points(self, points):
        self.virtual_grid_points = points

    def get_virtual_grid_points(self):
        return self.virtual_grid_points

    def get_tracking_mode(self):
        return self.tracking_mode

    def get_calibration_mode(self):
        return self.calibration_mode

    def get_error_msg(self):
        return self.error_msg

    def calibrate(self, t):
        self.calibration_mode = STATE_NON_ACTIVE
        t_end = time.time() + t
        record_arr = self.track_and_average(t_end)
        if len(record_arr) >= 3:
            if self.image_marker_points is not None:
                # TODO switch when there are 6 sensors
                # M_patient = self.get_transformation_mat(self.image_marker_points, record_arr[3:, :3])
                self.image_to_patient_mat, self.image_patient_fre = self.get_transformation_mat(self.image_marker_points, record_arr[:3, :3])
                # print(f"Image To Patient transformation matrix shape {self.image_to_patient_mat.shape}. values:\n {self.image_to_patient_mat}")
                if self.virtual_grid_points is not None:
                    self.virtual_grid_offsets = self.virtual_grid_points - self.image_marker_points[0,:]
            #         patient_marker_points = self.image_to_patient_mat @ self.get_homogenous_point(self.image_marker_points)
            #         patient_virtual_grid_points = self.image_to_patient_mat @ self.get_homogenous_point(self.virtual_grid_points)
            #         self.patient_virtual_grid_mat, self.patient_virtual_grid_fre = self.get_transformation_mat(patient_marker_points,
            #                                                                           patient_virtual_grid_points)
            #         # self.error_msg = f"Patient To Virtual Grid transformation matrix shape {self.patient_virtual_grid_mat.shape}. values:\n {self.patient_virtual_grid_mat}"
            #     else:
            #         self.error_msg = "ERROR: Patient To Virtual Grid calibration failed: No virtual grid points were provided"
            # else:
            #     print("ERROR: Image To Patient calibration failed: No image points were provided")
            # if self.robot_points is not None:
            #     self.patient_to_robot_mat, self.patient_robot_fre = self.get_transformation_mat(np.array(self.robot_points), record_arr[:3,:3])
            #     # print(f"Patient To Robot transformation matrix shape {self.patient_to_robot_mat.shape}. values:\n {self.patient_to_robot_mat}")
            # else:
            #     self.error_msg = "ERROR: Patient To Robot calibration failed: No robot points were provided. maybe check robot.conf file"
            # if self.image_to_patient_mat is not None and self.patient_virtual_grid_mat is not None\
            #     and self.patient_to_robot_mat is not None:
            #     self.error_msg = "Calibration succeed!"
                    self.calibration_mode = STATE_ACTIVE
                    self.Xarm.staging()
                    self.Xarm_status = STATE_STAGING
        else:
            num_active_sensors = self.get_num_active_sensors()
            if num_active_sensors >= len(record_arr):
                self.error_msg = "Calibration failed! looks like some of the sensors are out of motion box..."
            else:
                self.error_msg = f"Calibration failed! looks like there are not enough active sensors - " \
                                 f"you have {num_active_sensors} active sensors and at least 3 are needed for" \
                                 f" calibration"

    def track_and_average(self, t_end):
        arr = []
        while time.time() < t_end:
            self.Tracker.trackSensors()
            record = self.Tracker.getRecords()
            arr.append(self.record_to_np(record))
        arr = np.array(arr)
        print(f"total points arr shape {arr.shape}")
        average_arr = np.mean(arr, axis=0)
        print(f"average points arr shape {average_arr.shape}")
        return average_arr

    def track(self):
        if self.tracking_mode != STATE_ACTIVE:
            self.tracking_mode = STATE_ACTIVE
        self.Tracker.trackSensors()
        self._record = self.Tracker.getRecords()

    def stop_track(self):
        self.tracking_mode = STATE_NON_ACTIVE

    def close(self):
        self.Xarm.reset()

    def calc_movement(self):
        """
        order of sensors on grid - 1-----------2
                                   |
                                   |
                                   |
                                   |
                                   3
        :return:
        """
        current_robot = self.record_to_np(self._record[:3])

        # TODO switch when there are 6 sensors
        current_patient = self.record_to_np(self._record[:3])

        d_image_patient, marker_patient, t_image_patient = procrustes(current_patient[:,:3],
                                                                      self.image_marker_points)
        R_image_patient = t_image_patient["rotation"]
        target_offsets = self.virtual_grid_offsets @ R_image_patient
        self.target = marker_patient[0] + target_offsets
            # current_patient = self.record_to_np(self._record[3:])
        alignment_data, rmsd = R.align_vectors(self.target, current_robot[:,:3])
        self.alignment_angles = alignment_data.as_euler('xyz')
        return current_patient, current_robot

    def directions_to_robot(self):
        self.track()
        current_patient, current_robot = self.calc_movement()
        fixed,_,_ = self.Xarm.get_sensors_position()
        d_patient_robot, sensors_robot, tdict = procrustes(fixed[:,:3],
                                                                      current_robot[:,:3])
        R, b, T = tdict["rotation"], tdict["scale"], tdict["translation"]
        target_robot = b * (self.target @ R) + T
        alignment_robot = b * (self.alignment_angles @ R) + T

        #         # TODO ********erase when there is real data*******
        target_robot, alignment_robot = self.create_random_movement(fixed[0])
        # move = np.conctenate((target_robot[0], alignment_robot[0]))
        code = self.Xarm.move(target_robot, alignment_robot)
        return code

        # if self.patient_to_robot_mat is not None:
        #     # code, cur_location = self.Xarm.get_position()
        #     if self.target is not None and self.alignment_angles is not None:
        #         self.error_msg = DIRECTION_MASSAGE
        #         current_robot = self.record_to_np(self._record)[:3,:3]
        #         # TODO ********erase when there is real data*******
        #         self.target = np.repeat(np.array([current_robot[0] + 20,
        #                                           current_robot[1]+10, current_robot[2]-10]), 3, axis=0)
        #         print(f"target shape {self.target.shape}")
        #         # TODO ********erase when there is real data*******
        #         alignment_robot_frame = self.patient_to_robot_mat @ self.get_homogenous_point(self.alignment_angles[...,None])
        #         r_to_target = np.sqrt(np.sum((current_robot[0] - self.target[0])**2))
        #         if r_to_target > MIN_RADIUS:
        #             x,y,z = self.get_sphere_points(r_to_target/3, self.target[0])
        #             mid_point = self.path_on_sphere(np.array([x.flatten(),y.flatten(),z.flatten()]).T, current_robot[0])
        #             print(f"radius to target {r_to_target}, mid point shape {mid_point.shape}, target shape {self.target[0].shape}")
        #             path = np.hstack((mid_point[...,None], self.target[0][...,None]))
        #             print(f'mid point {mid_point}, target {self.target}, path {path}')
        #             path_robot_frame = self.patient_to_robot_mat @ self.get_homogenous_point(path)
        #             code = self.Xarm.move_circle(path_robot_frame[:, 0],path_robot_frame[:, 1])
        #             if code == 0:
        #                 code = self.Xarm.align(alignment_robot_frame)
        #                 self.error_msg = self.error_msg + "\n" + self.Xarm.get_err()
        #                 return code
        #             self.error_msg = self.error_msg + "\n" + self.Xarm.get_err()
        #         else:
        #             path_robot_frame = self.patient_to_robot_mat @ self.get_homogenous_point(self.target[0])
        #             code = self.Xarm.move(path_robot_frame, alignment_robot_frame)
        #             self.error_msg = self.error_msg + "\n" + self.Xarm.get_err()
        #         return code
        #     else:
        #         self.error_msg = self.error_msg + "\n" + NO_TRACK_MASSAGE
        #         return FAILURE
        # self.error_msg = self.error_msg + "\n" + NO_MAT_MASSSAGE
        # return FAILURE


if __name__ == "__main__":
    # server.run()
    image_points = np.random.randn(3,3)
    virtual_grid = np.random.randn(3,3)

    app = GridTracker()
    app.set_image_marker_points(image_points)
    app.set_virtual_grid_points(virtual_grid)

    app.calibrate(5)
    i = 0
    print("tracking...")
    start_time = time.time()
    while not keyboard.is_pressed('q'):
        app.track()
        app.calc_movement()
        i += 1
    end_time = time.time()
    print(f"total {i} iteration. average time per iteration : {(end_time - start_time) / i}")


