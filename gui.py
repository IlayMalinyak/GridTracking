# __author:IlayK
# data:17/03/2022
import PySimpleGUI as sg
import core
import numpy as np
from PIL import Image
import depth_table
from multiprocessing import Process
import server
import time


font_header = ("Arial, 18")
font_massage = ("Ariel, 14")
DATA_HEADER = "Tracking Data:"
DATA_TEXT = "Position: {}\nDistance to target: {}\nAlignment with target: {}\n" \
            "Fiducial Registration Error (FRE): {}\n"
SYSTEM_HEADER = "System Data:"
SYSTEM_TEXT = "Tracking - {}\nXarm - {}\nNumber Active Sensors - {}\nMetric - {}"
MASSAGE_HEADER = "Massages Box:\n"
MASSAGE_BODY = "{}"
IMG_PATH = ".\static\img\map.png"
RAND_IMG_PTS = np.random.randn(3, 3)
RAND_GRID_PTS = np.random.randn(3, 3)

im = Image.open(IMG_PATH)
data_column = [[sg.Button("Initialize", size=(40,3),key="-INIT-"), sg.Button("Calibrate", size=(40,3), key="-CALIB-")],
               [sg.Button("Start Tracking", size=(82,3), key="-TRACK-")],
                [sg.Text(DATA_HEADER, font=font_header)],
               [sg.Text(DATA_TEXT, font=font_massage, key="-DATA_TEXT-")],
               [sg.Button("Confirm Movement", size=(82, 3), key="-CONFIRM-")]
               , [sg.Button("Stop Tracking", size=(82,3), key="-STOP-")],
                [sg.Text(SYSTEM_HEADER, font=font_header)],
               [sg.Text(SYSTEM_TEXT, font=font_massage, key="-SYSTEM_TEXT-")]
               , [sg.Text(MASSAGE_HEADER, font=font_header)],
               [sg.Text(MASSAGE_BODY, font=font_massage,size=(60,5), key="-MSG-", background_color="white", text_color='red')],
               [sg.Button("Quit", size=(82, 3), key="-QUIT-")]]
img_column = [[sg.Image(IMG_PATH, key='-IMAGE-')]]
layout = [
    [sg.Column(data_column),
     sg.VSeperator(),
     sg.Column(img_column)]]

window = sg.Window("GuidanceTracker Demo", layout, size=(1200,800))


def initialize_app():
    app = core.GridTracker()
    return app


def calibrate_app(app):
    app.set_image_marker_points(RAND_IMG_PTS)
    app.set_virtual_grid_points(RAND_GRID_PTS)
    app.calibrate(3)


def disable_buttons(window, keys):
    for key in keys:
        window[key].update(disabled=True, button_color=('black','gray'))


def enable_buttons(window, keys):
    for key in keys:
        window[key].update(disabled=False, button_color=('black', 'bisque2'))


def initialize_window(window):
    disable_buttons(window, ["-CALIB-", "-TRACK-", "-STOP-", "-CONFIRM-"])
    enable_buttons(window, ["-INIT-"])
    update_window_text(None, window)


def arr_string_repr(arr):
    str_arr = "{:.2f}, {:.2f}, {:.2f}".format(arr[0], arr[1], arr[2])
    return str_arr


def update_window_text(app, window):
    if app is not None:
        pos, dist, align, fre = app.get_record(), app.get_distance(), app.get_alignment(), app.get_mean_fre()
        tracker_state, xarm_state = app.get_tracker_status(), app.get_xarm_status()
        num_sensors, metric = app.get_num_active_sensors(), app.get_metric()
        sensor_pos = pos[0][:3] if pos is not None else None
        if sensor_pos is not None:
            sensor_pos, dist, align = arr_string_repr(sensor_pos), arr_string_repr(dist), arr_string_repr(align)
        data_text = DATA_TEXT.format(sensor_pos, dist,align,fre)
        sys_text = SYSTEM_TEXT.format(tracker_state, xarm_state, num_sensors, metric)
        msg_text = app.get_error_msg()
    else:
        data_text = DATA_TEXT.format(None,None,None,None)
        sys_text = SYSTEM_TEXT.format("Not Active", "Not Active", None, None)
        msg_text = core.DEFAULT_MASSAGE
    window["-DATA_TEXT-"].update(data_text)
    window["-SYSTEM_TEXT-"].update(sys_text)
    window["-MSG-"].update(msg_text)


def run_app():
    app = None
    event, values = window.read(timeout=100)
    initialize_window(window)
    while True:
        event, values = window.read(timeout=100)
        try:
            depth_table.mark_grid()
        except Exception as e:
            print("IMAGE ERROR")
            print(e)
        window['-IMAGE-'].update(IMG_PATH)
        if event == "-INIT-":
            window['-MSG-'].update("Initializing...")
            app = initialize_app()
            print("finished initializing")
            update_window_text(app, window)
            print("finished updating")
            if app.get_tracker_status() == core.STATE_ACTIVE:
                window['-INIT-'].update(disabled=True, button_color=('black', 'spring green'))
                enable_buttons(window, ["-CALIB-"])
            else:
                window['-INIT-'].update(button_color='red')
        if event == "-CALIB-":
            window['-MSG-'].update("Calibrating...")
            calibrate_app(app)
            update_window_text(app, window)
            if app is not None and app.get_calibration_mode() == core.STATE_ACTIVE:
                enable_buttons(window, ['-TRACK-', '-STOP-', '-CONFIRM-'])
                window['-CALIB-'].update(disabled=True, button_color=('black', 'spring green'))
        if event == "-TRACK-":
            print("tracking")
            app.track()
            app.calc_movement()
            window['-TRACK-'].update(disabled=True, button_color=('black', 'spring green'))
            update_window_text(app, window)
        if event == "-STOP-":
            app.stop_track()
            enable_buttons(window, ["-TRACK-"])
        if app is not None and app.get_tracking_mode() == core.STATE_ACTIVE:
            app.track()
            app.calc_movement()
            update_window_text(app, window)
        if event == "-CONFIRM-":
            err_code = app.directions_to_robot()
            if err_code != 0:
                print("-----ERROR------")
        if event == "-QUIT-" or event == sg.WIN_CLOSED:
            app.close()
            break
        update_window_text(app, window)


# Run the Event Loop
if __name__ == "__main__":
    run_app()


