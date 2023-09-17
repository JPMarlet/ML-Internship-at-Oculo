# ML-Internship-at-Oculo

Overview of OxIOD_Data_Formatting_Script.py:


Input -->    raw imu and vicon csv files from the OxIOD dataset.

output -->   csv formatted .txt files:

- - `my_timestamps_p.txt` VIO timestamps.
  - [t]
  - Note: single column, skipped first 20 frames
- `imu_measurements.txt` raw and calibrated IMU data
  - [t, acc_raw (3), acc_cal (3), gyr_raw (3), gyr_cal (3), has_vio] 
 - Note: calibration through VIO calibration states. The data has been interpolated evenly between images around 1000Hz. Every timestamp in my_timestamps_p.txt will have a corresponding timestamp in this file (has_vio==1). 
- `evolving_state.txt` ground truth (VIO) states at IMU rate.
  - [t, q_wxyz (4), p (3), v (3)]
  - Note: VIO state estimates with IMU integration. Timestamps are from raw IMU measurements.
- `calib_state.txt` VIO calibration states at image rate.
  - [t, acc_scale_inv (9), gyr_scale_inv (9), gyro_g_sense (9), b_acc (3), b_gyr (3)]
  - Note: Changing calibration states from VIO.
- `atttitude.txt` AHRS attitude from IMU
  - [t, qw, qx, qy, qz]
