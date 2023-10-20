
"""
This script takes as input raw imu and vicon csv files from the OxIOD dataset.

output --> csv formatted .txt files:

- - `my_timestamps_p.txt` VIO timestamps.
  - [t]
  - Note: single column, skipped first 20 frames
- `imu_measurements.txt` raw and calibrated IMU data
  - [t, acc_raw (3), acc_cal (3), gyr_raw (3), gyr_cal (3), has_vio] 
 - Note: calibration through VIO calibration states. 
    The data has been interpolated evenly between images around 1000Hz. 
    Every timestamp in my_timestamps_p.txt will have a corresponding timestamp in this file (has_vio==1). 
- `evolving_state.txt` ground truth (VIO) states at IMU rate.
  - [t, q_wxyz (4), p (3), v (3)]
  - Note: VIO state estimates with IMU integration. Timestamps are from raw IMU measurements.
- `calib_state.txt` VIO calibration states at image rate (used in `data_io.py`)
  - [t, acc_scale_inv (9), gyr_scale_inv (9), gyro_g_sense (9), b_acc (3), b_gyr (3)]
- `attitude.txt` AHRS attitude from IMU
  - [t, qw, qx, qy, qz]

"""

import glob
import csv
import os
import shutil
import bisect
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample
from pyquaternion import Quaternion
import csv


def main():
    imu_file_pattern = f'C:\\Users\\Admin\\Documents\\GitHub\\OxIOD\\OxIOD Dataset\\*\\data*\\raw\\imu*.csv'
    imu_files = glob.glob(imu_file_pattern)

    for seq_number, imu_file in enumerate(imu_files):
        vicon_file = imu_file.replace('imu', 'vi')
        output_path = f'C:\\Users\\Admin\\Documents\\GitHub\\TLIO\\Input Data for gen_fb_data.py\\Dataset\\Sequence_{seq_number}'

        # if output_path exists, clear it
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            os.mkdir(output_path)
        else:
            os.mkdir(output_path)
        
        print(f"Sequence: {seq_number}")

      
        ox_imu_data = _parse_imu_file(imu_file) 

        ox_vicon_data = _parse_vicon_file(vicon_file)
            
    
        clipped_ox_vicon_data = _clip_ox_vicon_timestamps_to_fit_ox_imu(ox_imu_data, ox_vicon_data)

        
        # try:
            #ox_imu_calibrated_data = _add_calibration_to_ox_imu_data(ox_imu_data)
            
        #     print(f"Ox IMU acc_cal_x (first value): {ox_imu_calibrated_data['acc_cal_x'][0]}")
        # except Exception as e:
        #     print("error with calibrating:", e)

        # try:
        #     ox_interpolated_vicon_data = _interpolate_ox_data_upscaling_vicon_ts(ox_imu_data, ox_vicon_data)
        # except Exception as e:
        #     print("Error with interpolation function:", e)

        ox_imu_upsampled_data = _upsample_ox_imu_data_interp1d(ox_imu_data)

        quaternions = _create_list_of_ox_vicon_quaternions(ox_vicon_data, ox_imu_data, ox_imu_upsampled_data, clipped_ox_vicon_data)

        ox_interpolated_vicon_data = _interpolate_ox_vicon_quaternions(ox_imu_upsampled_data, quaternions, clipped_ox_vicon_data)


        ox_imu_upsampled_data = _add_simple_calibration(ox_imu_upsampled_data, ox_interpolated_vicon_data)
                
        ox_interpolated_vicon_data = _linearly_interpolate_vicon_non_quaternions(ox_interpolated_vicon_data, ox_vicon_data, 
                                                                    clipped_ox_vicon_data, ox_imu_upsampled_data)
        
        
         
        # for key in ox_vicon_data.keys():
        #     print(key, ox_vicon_data[key][-1])

        # # creates new dictionary with calibrated imu data
        # ox_imu_calibrated_data = _add_calibration_to_ox_imu_data(ox_imu_data)

        velocity_midpoints_x, velocity_midpoints_y, velocity_midpoints_z = _assign_and_calculate_velocity_midpoints_for_vicon_vel_est(ox_interpolated_vicon_data)
        vicon_vel_est_x, vicon_vel_est_y, vicon_vel_est_z = _calculate_velocities_from_midpoint_velocities_for_vicon_vel_est(velocity_midpoints_x,
                                                                                                                            velocity_midpoints_y,
                                                                                                                            velocity_midpoints_z,)
        
        ox_interpolated_vicon_data = _add_vicon_vel_est_to_ox_vicon_vel_data(ox_interpolated_vicon_data, 
                                            vicon_vel_est_x, vicon_vel_est_y, 
                                            vicon_vel_est_z)



        
        _write_tlio_imu_measurements_file(ox_imu_upsampled_data, output_path)
        _write_tlio_evolving_state_file(ox_interpolated_vicon_data, output_path)

        _write_tlio_my_timestamps_p(ox_imu_upsampled_data, output_path)

        write_tlio_calib_state_files(ox_imu_upsampled_data, output_path)

        q_w_imu, q_x_imu, q_y_imu, q_z_imu =  _convert_ox_imu_attitude_to_quaternions(ox_imu_upsampled_data)

        write_tlio_attitude_files(q_w_imu, q_x_imu, q_y_imu, q_z_imu, output_path, ox_imu_upsampled_data)

         # _write_info_file(imu_file, vicon_file, output_path)



def _parse_imu_file(imu_file):
    """Parse IMU CSV file and return a dictionary of lists with IMU data points."""

    ox_imu_data = {
        'timestamps': [],
        'attitude_roll': [],
        'attitude_pitch': [],
        'attitude_yaw': [],
        'gyr_raw_x': [],
        'gyr_raw_y': [],
        'gyr_raw_z': [],
        'gravity_x': [],
        'gravity_y': [],
        'gravity_z': [],
        'user_acc_x': [],
        'user_acc_y': [],
        'user_acc_z': [],
       
            }

    with open(imu_file) as f:
        csv_reader = csv.reader(f, delimiter=',')

        for row_num, row in enumerate(csv_reader):
            if row_num < 20:
                continue

            try:
                ox_imu_data['timestamps'].append(float(row[0]) * 1e6)
                ox_imu_data['attitude_roll'].append(float(row[1]))
                ox_imu_data['attitude_pitch'].append(float(row[2]))
                ox_imu_data['attitude_yaw'].append(float(row[3]))
                ox_imu_data['gyr_raw_x'].append(float(row[4]))
                ox_imu_data['gyr_raw_y'].append(float(row[5]))
                ox_imu_data['gyr_raw_z'].append(float(row[6]))
                ox_imu_data['gravity_x'].append(float(row[7]))
                ox_imu_data['gravity_y'].append(float(row[8]))
                ox_imu_data['gravity_z'].append(float(row[9]))
                ox_imu_data['user_acc_x'].append(float(row[10]))
                ox_imu_data['user_acc_y'].append(float(row[11]))
                ox_imu_data['user_acc_z'].append(float(row[12]))
                
            except ValueError as e:
                print(f"Error on line {row_num}: {e}")

        first_row = [lst[0] for lst in ox_imu_data.values() if lst]
        print("IMU data parsed")

    return ox_imu_data


def _parse_vicon_file(vicon_file):
    """Parse vicon CSV file and return a dictionary of lists of vicon data points."""
    
    ox_vicon_data = {
        'timestamps': [],
        'translation_x': [],
        'translation_y': [],
        'translation_z': [],
        'rotation_x': [],
        'rotation_y': [],
        'rotation_z': [],
        'rotation_w': []
    }

    with open(vicon_file) as f:
        csv_reader = csv.reader(f, delimiter=',')

        for row_num, row in enumerate(csv_reader):  # Starts numbering from 1 for clarity
            if row_num < 20:
                continue

            try:
                ox_vicon_data['timestamps'].append(float(row[0]) / 1e3)
                ox_vicon_data['translation_x'].append(float(row[1]))
                ox_vicon_data['translation_y'].append(float(row[2]))
                ox_vicon_data['translation_z'].append(float(row[3]))
                ox_vicon_data['rotation_x'].append(float(row[4]))
                ox_vicon_data['rotation_y'].append(float(row[5]))
                ox_vicon_data['rotation_z'].append(float(row[6]))
                ox_vicon_data['rotation_w'].append(float(row[7]))
            except ValueError as e:
                print(f"Error on line {row_num}: {e}")


    first_row = [lst[0] for lst in ox_vicon_data.values() if lst]
    #print(f"first_row of ox_vicon_data dictionary: {first_row}")
    print("vicon data parsed")

    return ox_vicon_data



def _clip_ox_vicon_timestamps_to_fit_ox_imu(ox_imu_data, ox_vicon_data):

    ox_imu_ts_start = ox_imu_data['timestamps'][0]
    ox_imu_ts_end = ox_imu_data['timestamps'][-1]


    clipped_ox_vicon_data = {'timestamps': [], 'rotation_w': [], 'rotation_x': [], 'rotation_y': [], 'rotation_z': [], 
                    'translation_x': [], 'translation_y': [],'translation_z': []}


    for idx, timestamp in enumerate(ox_vicon_data['timestamps']):
        if ox_imu_ts_start <= timestamp <= ox_imu_ts_end:

            try:
                clipped_ox_vicon_data['timestamps'].append(timestamp)
                clipped_ox_vicon_data['rotation_w'].append(ox_vicon_data['rotation_w'][idx])
                clipped_ox_vicon_data['rotation_x'].append(ox_vicon_data['rotation_x'][idx])
                clipped_ox_vicon_data['rotation_y'].append(ox_vicon_data['rotation_y'][idx])
                clipped_ox_vicon_data['rotation_z'].append(ox_vicon_data['rotation_z'][idx])
                clipped_ox_vicon_data['translation_x'].append(ox_vicon_data['translation_x'][idx])
                clipped_ox_vicon_data['translation_y'].append(ox_vicon_data['translation_y'][idx])
                clipped_ox_vicon_data['translation_z'].append(ox_vicon_data['translation_z'][idx])
            except Exception as e:
                print("Error with appending clipped_ox_vicon_data:", e)

    return clipped_ox_vicon_data


def _checking_imu_vs_vicon_clipped_timestamps(ox_imu_data, clipped_ox_vicon_data):
    print("IMU First 10:", ox_imu_data['timestamps'][:10])
    print("IMU Last 10:", ox_imu_data['timestamps'][-10:])
    print("Clipped Vicon First 10:", clipped_ox_vicon_data['timestamps'][:10])
    print("Clipped Vicon Last 10:", clipped_ox_vicon_data['timestamps'][-10:])


def _upsample_ox_imu_data_interp1d(ox_imu_data):    
    """Upscales imu data by factor of 10 to acheive 1000Hz for TLIO
    Then the vicon data will be interpolated against the imu timestamps
    """
   
    original_timestamps = ox_imu_data['timestamps']
    num_new_points = len(original_timestamps) * 10
    new_timestamps = np.linspace(original_timestamps[0], original_timestamps[-1], num=num_new_points)
    # print(f"The length of original_timestamps is:{len(original_timestamps)} ")
    # print(f"The length of new_timestamps is:{len(new_timestamps)} ")

    ox_imu_upsampled_data = {'timestamps': new_timestamps}
    #print(f"Length of ox_imu_data[timestamps]: {len((ox_imu_data)['timestamps'])}")
    #print(f"Length of ox_imu_upsampled_data[timestamps]: {len((ox_imu_upsampled_data)['timestamps'])}")

    # checks if all lists in the ox_imu dictionary have the same length 
    lengths = [len(v) for k, v in ox_imu_data.items() if k != 'timestamps']
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All lists in the ox_imu dictionary must have the same length.")
               
                            
    for key in ox_imu_data:
        if key != 'timestamps':
            interpolator = interp1d(original_timestamps, ox_imu_data[key], kind='linear', fill_value="extrapolate")
            ox_imu_upsampled_data[key] = interpolator(new_timestamps)

    print("imu upsampled")
        
    return ox_imu_upsampled_data


def _add_simple_calibration(ox_imu_upsampled_data, ox_interpolated_vicon_data):
    
    calibration_keys_to_fill = ['user_acc_x_cal', 'user_acc_y_cal','user_acc_z_cal', 
                                'gyr_cal_x','gyr_cal_y', 'gyr_cal_z']

    for key in calibration_keys_to_fill:

        ox_imu_upsampled_data[key] = [0] * len(ox_interpolated_vicon_data['timestamps'])
        first_row = [lst[0] for lst in ox_imu_upsampled_data.values() if len(lst) > 0]
    #print(f"first row of ox_interpolated_vicon_data dictionary, with added calibration: {first_row}")
    
    return ox_imu_upsampled_data
 


def _create_list_of_ox_vicon_quaternions(ox_vicon_data, ox_imu_data, ox_imu_upsampled_data, clipped_ox_vicon_data):

   # print(ox_vicon_data['timestamps'][:20]) # checking ox_vicon_data is not empty

    # clipped vicon data to fit within imu timestamps
    ox_clipped_vicon_data = _clip_ox_vicon_timestamps_to_fit_ox_imu(ox_imu_data, ox_vicon_data) 


    vicon_clipped_timestamps = clipped_ox_vicon_data['timestamps']
    vicon_desired_timestamps = ox_imu_upsampled_data['timestamps']   

    quaternions = []
    # q_w = []
    # q_x = []
    # q_y = []
    # q_z = []
    # q_w_int = []
    # q_x_int = []
    # q_y_int = []
    # q_z_int = []
           
    #new_length_lists = len(ox_vicon_data['timestamps']) * 10

    for i in range(len(vicon_clipped_timestamps)):

        q_w = ox_vicon_data['rotation_w'][i]
        q_x = ox_vicon_data['rotation_x'][i]
        q_y = ox_vicon_data['rotation_y'][i]
        q_z = ox_vicon_data['rotation_z'][i]

        quaternion = Quaternion(q_w, q_x, q_y, q_z)
        #print(quaternion)
        quaternions.append(quaternion)
    #print(quaternions)
    #print(f"The length of quaternions list: {len(quaternions)}")

    return quaternions

def _interpolate_ox_vicon_quaternions(ox_imu_upsampled_data, quaternions, clipped_ox_vicon_data):
    """ - Interpolating vicon quaternions using speherical linear interpolation
        - creating dictonary: ox_interpolated_vicon_data and adding components from interpolated_quaternions
          list into separate lists inside ox_interpolated_vicon_data
    """
    ox_interpolated_vicon_data = {'timestamps': ox_imu_upsampled_data['timestamps'], 
                                'rotation_w': [], 'rotation_x': [], 'rotation_y': [], 'rotation_z': [],  
                                'translation_x': [], 'translation_y': [],'translation_z': []}


    vicon_clipped_timestamps = clipped_ox_vicon_data['timestamps']
    
    for timestamp in ox_imu_upsampled_data['timestamps']:
        original_index = bisect.bisect(vicon_clipped_timestamps, timestamp) - 1

        if original_index >= len(vicon_clipped_timestamps) - 1:
            original_index = len(vicon_clipped_timestamps) - 2


        if original_index < 0:
            original_index = 0


        fraction = ((timestamp - vicon_clipped_timestamps[original_index]) /
                   (vicon_clipped_timestamps[original_index + 1] - vicon_clipped_timestamps[original_index]))

        slerped_quat = Quaternion.slerp(quaternions[original_index], quaternions[original_index + 1], fraction)

        ox_interpolated_vicon_data['rotation_w'].append(slerped_quat.w)
        ox_interpolated_vicon_data['rotation_x'].append(slerped_quat.x)
        ox_interpolated_vicon_data['rotation_y'].append(slerped_quat.y)
        ox_interpolated_vicon_data['rotation_z'].append(slerped_quat.z)

    print("quaternions interpolated")

    return ox_interpolated_vicon_data



def _linearly_interpolate_vicon_non_quaternions(ox_interpolated_vicon_data, ox_vicon_data, 
                                                                    clipped_ox_vicon_data, ox_imu_upsampled_data):
    
    #print(ox_interpolated_vicon_data)

    original_timestamps = clipped_ox_vicon_data['timestamps']
    new_timestamps = ox_imu_upsampled_data['timestamps']
    

    for key in ox_vicon_data.keys():
        if key not in ['rotation_w', 'rotation_x', 'rotation_y', 'rotation_z']:

            interpolator = interp1d(original_timestamps, clipped_ox_vicon_data[key], kind='linear', fill_value="extrapolate")
            ox_interpolated_vicon_data[key] = interpolator(new_timestamps)

    #print(f"Length of vicon linearly interpolated translation_x key: {len(ox_interpolated_vicon_data['translation_x'])}")
        

    return ox_interpolated_vicon_data


def _assign_and_calculate_velocity_midpoints_for_vicon_vel_est(ox_interpolated_vicon_data):  


#converting units of translation for vicon 

    distance_travelled_x = [value / 100 for value in ox_interpolated_vicon_data['translation_x']]
    #print(f"Length of distance_travelled_x is: {len(distance_travelled_x)}")
    distance_travelled_y = [value / 100 for value in ox_interpolated_vicon_data['translation_y']]
    distance_travelled_z = [value / 100 for value in ox_interpolated_vicon_data['translation_z']]
    timestamps = ox_interpolated_vicon_data['timestamps']

    velocity_midpoints_x = [] 
    velocity_midpoints_y = []  
    velocity_midpoints_z = []  
   
    for i_curr in range((len(distance_travelled_x)) - 1):
        #print(f"Current index: {i_curr}, List length: {len(distance_travelled_x)}")
        try:

            delta_distance_x = float(distance_travelled_x[i_curr + 1]) - float(distance_travelled_x[i_curr])
            delta_distance_y = float(distance_travelled_y[i_curr + 1]) - float(distance_travelled_y[i_curr])
            delta_distance_z = float(distance_travelled_z[i_curr + 1]) - float(distance_travelled_z[i_curr])
            delta_time = float(timestamps[i_curr + 1]) - float(timestamps[i_curr])

        except IndexError:
            print(f"Row {i_curr} does not have enough data.")
                
        

        try:

            velocity_midpoints_x.append(delta_distance_x / delta_time)
            velocity_midpoints_y.append(delta_distance_y / delta_time)
            velocity_midpoints_z.append(delta_distance_z / delta_time)

        except ZeroDivisionError:

                velocity_midpoints_x.append(velocity_midpoints_x[-1] if velocity_midpoints_x else 0)
                velocity_midpoints_y.append(velocity_midpoints_y[-1] if velocity_midpoints_y else 0)
                velocity_midpoints_z.append(velocity_midpoints_z[-1] if velocity_midpoints_z else 0)
                
    assert len(velocity_midpoints_x) == len(velocity_midpoints_y) == len(velocity_midpoints_z) 
    #print(f"The length of velocity_midpoints_x is: {len(velocity_midpoints_x)}")   

    return(velocity_midpoints_x, velocity_midpoints_y, velocity_midpoints_z)


def _calculate_velocities_from_midpoint_velocities_for_vicon_vel_est(velocity_midpoints_x,
                                                                    velocity_midpoints_y,
                                                                    velocity_midpoints_z,):                                    
                                                                    
    vicon_vel_est_x = []
    vicon_vel_est_y = []
    vicon_vel_est_z = []  

    for  i_curr in range((len(velocity_midpoints_x) - 1)):
        
        try:

            if i_curr == 0:

                vicon_vel_est_x.append(velocity_midpoints_x[i_curr])
                vicon_vel_est_y.append(velocity_midpoints_y[i_curr])
                vicon_vel_est_z.append(velocity_midpoints_z[i_curr])

            elif i_curr == len(velocity_midpoints_x) - 1:

                vicon_vel_est_x.append(velocity_midpoints_x[i_curr])
                vicon_vel_est_y.append(velocity_midpoints_y[i_curr])
                vicon_vel_est_z.append(velocity_midpoints_z[i_curr])

            else:

                vicon_vel_est_x.append(((float(velocity_midpoints_x[i_curr + 1]) + float(velocity_midpoints_x[i_curr])) / 2))                            
                vicon_vel_est_y.append(((float(velocity_midpoints_y[i_curr + 1]) + float(velocity_midpoints_y[i_curr])) / 2))
                vicon_vel_est_z.append(((float(velocity_midpoints_z[i_curr + 1]) + float(velocity_midpoints_z[i_curr])) / 2))
                
        except IndexError:
                print(f"Row {i_curr} does not have enough data.")
                if len(vicon_vel_est_x) > 0:
                        vicon_vel_est_x.append(vicon_vel_est_x[-1]) 
                        vicon_vel_est_y.append(vicon_vel_est_y[-1]) 
                        vicon_vel_est_z.append(vicon_vel_est_z[-1])
                else:
                    print("The vicon_vel_est_x list is emtpy")


    # Append the last value of each list to make the last three values duplicates
    vicon_vel_est_x += [vicon_vel_est_x[-1]] * 2
    vicon_vel_est_y += [vicon_vel_est_y[-1]] * 2
    vicon_vel_est_z += [vicon_vel_est_z[-1]] * 2
        #assert((len(vicon_vel_est_x)) == len(velocity_midpoints_x))
        #assert(len(timestamps) == len(vicon_vel_est_x))
    #print(f"The length of vicon_vel_est_x is:{len(vicon_vel_est_x)}")
    #print(f"The length of vicon_vel_est_y is:{len(vicon_vel_est_y)}")
    #print(f"The length of vicon_vel_est_z is:{len(vicon_vel_est_z)}")


    return(vicon_vel_est_x, vicon_vel_est_y, vicon_vel_est_z)

def _add_vicon_vel_est_to_ox_vicon_vel_data(ox_interpolated_vicon_data, 
                                            vicon_vel_est_x, vicon_vel_est_y, 
                                            vicon_vel_est_z):
    """ Adding vicon velocity estimation to interpolated vicon dictionary of OxIOD data
    """

    ox_interpolated_vicon_data['v_x'] = vicon_vel_est_x
    ox_interpolated_vicon_data['v_y'] = vicon_vel_est_y
    ox_interpolated_vicon_data['v_z'] = vicon_vel_est_z

    #print(f"First ten v_estim values for ox_interpolated_vicon_data['v_x']: {ox_interpolated_vicon_data['v_x'][:10]}")
    print("velocity estimates appended")
    return ox_interpolated_vicon_data



def _write_tlio_evolving_state_file(ox_interpolated_vicon_data, output_path):

    evolving_state_dictionary = ox_interpolated_vicon_data
    selected_keys = ['timestamps', 'rotation_w','rotation_x','rotation_y','rotation_z',
                   'translation_x','translation_y','translation_z', 'v_x','v_y','v_z',]
    df = pd.DataFrame({key: evolving_state_dictionary[key] for key in selected_keys})
    output_file = f'{output_path}\\evolving_state.txt'

    df.to_csv(output_file, index=False, header=False) 

    

def _write_tlio_imu_measurements_file(ox_imu_upsampled_data, output_path):

    ox_imu_upsampled_data['has_vio'] = [1] * len(ox_imu_upsampled_data['timestamps'])
    imu_measurement_file_dictionary = ox_imu_upsampled_data
    selected_keys = ['timestamps', 'user_acc_x','user_acc_y','user_acc_z', 
                     'user_acc_x_cal','user_acc_y_cal','user_acc_z_cal', 
                     'gyr_raw_x','gyr_raw_y', 'gyr_raw_z', 
                     'gyr_cal_x','gyr_cal_y', 'gyr_cal_z',
                     'has_vio']
    
    df = pd.DataFrame({key: imu_measurement_file_dictionary[key] for key in selected_keys})
    output_file = f'{output_path}\\imu_measurements.txt'
    df.to_csv(output_file, index=False, header=False) 

    print(f'Saved {output_file}')


def _write_tlio_my_timestamps_p(ox_imu_upsampled_data, output_path):

    timestamps = ox_imu_upsampled_data['timestamps']
    df = pd.DataFrame({'timestamps': timestamps})
    output_file = f'{output_path}\\my_timestamps_p.txt'
    df.to_csv(output_file, index=False, header=False) 

    print(f'Saved {output_file}')


def write_tlio_calib_state_files(ox_imu_upsampled_data, output_path):

    timestamps = ox_imu_upsampled_data['timestamps']

    acc_scale_inv = np.array([[1/((14.2 /1e9))**2, 0.0, 0.0],[0.0, 1/(15.2 / 1e9)*2, 0.0],[0.0, 0.0, 1/(25.1 / 1e9)**2]])
    gyro_scale_inv = np.array([[1/(13.6 / 1e6)**2, 0.0, 0.0],[0.0, 1/(12.1 / 1e6)*2, 0.0],[0.0, 0.0, 1/(8.7 / 1e6)**2]])
    gyro_g_sense = np.array([[(195 * 9.81) * 1e-9, 0, 0],[0, (195 * 9.81) * 1e-9, 0],[0, 0, (195 * 9.81) * 1e-9]])  

  

    df = pd.DataFrame({'timestamps': timestamps, 
                       'acc_scale_inv_1': [acc_scale_inv[0, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_2': [acc_scale_inv[0, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_3': [acc_scale_inv[0, 2]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_4': [acc_scale_inv[1, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_5': [acc_scale_inv[1, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_6': [acc_scale_inv[1, 2]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_7': [acc_scale_inv[2, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_8': [acc_scale_inv[2, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'acc_scale_inv_9': [acc_scale_inv[2, 2]] * len(ox_imu_upsampled_data['timestamps']),

                       'gyro_scale_inv_1': [gyro_scale_inv[0, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_2': [gyro_scale_inv[0, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_3': [gyro_scale_inv[0, 2]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_4': [gyro_scale_inv[1, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_5': [gyro_scale_inv[1, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_6': [gyro_scale_inv[1, 2]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_7': [gyro_scale_inv[2, 0]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_8': [gyro_scale_inv[2, 1]] * len(ox_imu_upsampled_data['timestamps']),
                       'gyro_scale_inv_9': [gyro_scale_inv[2, 2]] * len(ox_imu_upsampled_data['timestamps']),


                        'gyro_g_sense_1': gyro_g_sense[0,0] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_2': gyro_g_sense[0,1] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_3': gyro_g_sense[0,2] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_4': gyro_g_sense[1,0] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_5': gyro_g_sense[1,1] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_6': gyro_g_sense[1,2] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_7': gyro_g_sense[2,0] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_8': gyro_g_sense[2,1] * len(ox_imu_upsampled_data['timestamps']),
                        'gyro_g_sense_9': gyro_g_sense[2,2] * len(ox_imu_upsampled_data['timestamps']),
                        

                       'b_acc_1': [0] * len(ox_imu_upsampled_data['timestamps']),
                        'b_acc_2': [0] * len(ox_imu_upsampled_data['timestamps']),
                         'b_acc_3': [0] * len(ox_imu_upsampled_data['timestamps']),

                          'b_gyr_1': [0] * len(ox_imu_upsampled_data['timestamps']),
                          'b_gyr_2': [0] * len(ox_imu_upsampled_data['timestamps']),
                          'b_gyr_3': [0] * len(ox_imu_upsampled_data['timestamps'])

                       })
    
    output_file = f'{output_path}\\calib_state.txt'
    df.to_csv(output_file, index=False, header=False) 

    print(f'Saved {output_file}')


def _convert_ox_imu_attitude_to_quaternions(ox_imu_upsampled_data):

    q_w_imu = []
    q_x_imu = []
    q_y_imu = []
    q_z_imu = []

    roll = np.array(ox_imu_upsampled_data['attitude_roll'])
    pitch = np.array(ox_imu_upsampled_data['attitude_pitch'])
    yaw = np.array(ox_imu_upsampled_data['attitude_yaw'])

    
    for i in range(len(ox_imu_upsampled_data['timestamps'])):
    
        cy = np.cos(yaw[i] * 0.5)
        sy = np.sin(yaw[i] * 0.5)
        cp = np.cos(pitch[i] * 0.5)
        sp = np.sin(pitch[i] * 0.5)
        cr = np.cos(roll[i] * 0.5)
        sr = np.sin(roll[i] * 0.5)

        q_w_imu.append(cy * cp * cr + sy * sp * sr)
        q_x_imu.append(cy * cp * sr - sy * sp * cr)
        q_y_imu.append(sy * cp * sr + cy * sp * cr)
        q_z_imu.append(sy * cp * cr - cy * sp * sr)

    return q_w_imu, q_x_imu, q_y_imu, q_z_imu


def write_tlio_attitude_files(q_w_imu, q_x_imu, q_y_imu, q_z_imu, output_path, ox_imu_upsampled_data):

    timestamps = ox_imu_upsampled_data['timestamps']

    df = pd.DataFrame({'timestamps': timestamps,
                        'q_w_imu': q_w_imu,
                       'q_x_imu': q_x_imu,
                       'q_y_imu': q_y_imu,
                       'q_z_imu': q_z_imu
                       })
    

    output_file = f'{output_path}\\attitude.txt'
    df.to_csv(output_file, index=False, header=False) 

    print(f'Saved {output_file}')
    

if __name__ == '__main__':
    main()








# def _add_calibration_to_ox_imu_data(ox_imu_data):
#     """
#     calibrate raw IMU with fixed calibration.
#     - Uses covariance matrices of inverse scale factors for iPhone IMU accel and gyro
#     - Units of accel are converted to micro g's and r/s, to match IMU time stamps
#     """
#     for key in ox_imu_data:
#         ox_imu_data[key] = np.array(ox_imu_data[key])


#     ox_imu_calibrated_data = {}

#     accel_calib = []
#     gyro_calib = []


#     # covariance matrices of inverse scale factors for iPhone IMU accel and gyro
#     #The units of accel are converted to micro g's and r/s, to match IMU time stamps
#     acc_scale_inv = np.array([[1/(14.2 * 10**-3)**2, 0.0, 0.0],[0.0, 1/(15.2 * 10**-3)*2, 0.0],[0.0, 0.0, 1/(25.1 * 10**-3)**2]])
#     #gyro_scale_inv = np.array([[1/(13.6)**2, 0.0, 0.0],[0.0, 1/(12.1)*2, 0.0],[0.0, 0.0, 1/(8.7)**2]])

#     # g force sensitivity in units of m/s^2 (assuming worst sensitivity of iPhone 5s gyro)
#     gyro_g_sense = np.array([[(195 * 9.81) * 1e-3, 0, 0],[0, (195 * 9.81) * 1e-3, 0],[0, 0, (195 * 9.81) * 1e-3]])   

    
#     accel_raw = np.vstack((ox_imu_data['user_acc_x'], ox_imu_data['user_acc_y'], ox_imu_data['user_acc_z']))
#     gyro_raw = np.vstack((ox_imu_data['rotation_rate_x'], ox_imu_data['rotation_rate_y'], ox_imu_data['rotation_rate_z']))

#     #assuming 0 bias for accel and gyro
#     b_acc = [0] * accel_raw.shape[1]
#     b_gyro = [0] * gyro_raw.shape[1]

#     accel_calib = (np.dot(accel_raw, acc_scale_inv)).T

#     gyro_calib = (
#     np.dot(gyro_raw, gyro_g_sense.T)
#     - np.dot(accel_raw, gyro_g_sense.T)
#     ).T


#     print(f"the shape of accel_calib is: {accel_calib.shape}")
#     print(f"the shape of gyro_calib is: {gyro_calib.shape}")

#     accel_calib_list = accel_calib.tolist()
#     gyro_calib_list = gyro_calib.tolist()


#     #copy original ox_imu_data over 
#     for key, value in ox_imu_data.items():        
#         ox_imu_calibrated_data[key] = value

#     ox_imu_calibrated_data['acc_cal_x'] = accel_calib[:, 0]
#     ox_imu_calibrated_data['acc_cal_y'] = accel_calib[:, 1]
#     ox_imu_calibrated_data['acc_cal_z'] = accel_calib[:, 2]

#     ox_imu_calibrated_data['gyr_cal_x'] = gyro_calib[:, 0]
#     ox_imu_calibrated_data['gyr_cal_y'] = gyro_calib[:, 1]
#     ox_imu_calibrated_data['gyr_cal_z'] = gyro_calib[:, 2]

#     ox_imu_calibrated_data['b_gyr_x'] = b_gyro
#     ox_imu_calibrated_data['b_gyr_y'] = b_gyro
#     ox_imu_calibrated_data['b_gyr_z'] = b_gyro

#     ox_imu_calibrated_data['b_acc_x'] = b_acc
#     ox_imu_calibrated_data['b_acc_y'] = b_acc
#     ox_imu_calibrated_data['b_acc_z'] = b_acc


#     return ox_imu_calibrated_data