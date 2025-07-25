import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.x = np.array([[0], [0], [0], [0]]) #현재 상태 (위치 + 속도)
        self.P = np.eye(4) #오차 공분산 (추정의 신뢰도)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) #상태 전이 행렬 (현재 상태에서 다음 상태로 어떻게 변하는지)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]]) #측정 행렬 (센서 값과 상태 변수 간의 관계)
        self.Q = np.eye(4) * process_variance #프로세스 노이즈 (모델의 불확실성)
        self.R = np.eye(2) * measurement_variance #측정 노이즈 (센서 오차)
        
    def predict(self, dt, imu_acceleration):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        acceleration = np.array([[0.5 * imu_acceleration[0] * dt**2], 
                                  [0.5 * imu_acceleration[1] * dt**2],
                                  [imu_acceleration[0] * dt],
                                  [imu_acceleration[1] * dt]])
        self.x = self.F @ self.x + acceleration #현재 예측 상태 (위치 + 속도)
        self.P = self.F @ self.P @ self.F.T + self.Q #공분산(불확실성)
        
    # H @ x : 예측 위치값 
    # P : 상태 추정의 불확실성
    # R : GPS 센서 노이즈의 공분산
    def update(self, gps_measurement):
        y = gps_measurement - (self.H @ self.x) # 오차(측정값 - 예측값), 
        S = self.H @ self.P @ self.H.T + self.R # 현재 측정 오차의 신뢰도
        K = self.P @ self.H.T @ np.linalg.inv(S) #가중치!! 측정값을 신뢰할지 결정(예측이 정확할 수록 K작음)
        self.x = self.x + K @ y # 예측된 상태 x에 K @ y를 더해서 최종 보정된 상태
        self.P = (np.eye(4) - K @ self.H) @ self.P # 공분산(불확실성) 보정
        
# Create a function to run the Kalman Filter and update the plot
def run_kalman_filter(process_variance, measurement_variance):
    kf = KalmanFilter(process_variance, measurement_variance)
    estimates_x = []
    estimates_y = []

    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        imu_acceleration = [imu_acceleration_x[i], imu_acceleration_y[i]]
        gps_measurement = np.array([[gps_x[i]], [gps_y[i]]])

        kf.predict(dt, imu_acceleration)
        kf.update(gps_measurement)

        estimates_x.append(kf.x[0, 0])
        estimates_y.append(kf.x[1, 0])
    return estimates_x, estimates_y


data = pd.read_csv('IMU_GPS_sensor_data.csv')
time = data['time'].values
gps_x = data['gps_x'].values
gps_y = data['gps_y'].values
abs_x = data['absolute_x'].values
abs_y = data['absolute_y'].values
imu_acceleration_x = data['imu_acceleration_x'].values
imu_acceleration_y = data['imu_acceleration_y'].values


# visualiztion
fig, ax = plt.subplots(figsize=(10,8))
plt.subplots_adjust(bottom=0.25)

gps_plot, = ax.plot(gps_x, gps_y, 'g.', alpha=0.5, label = 'GPS measured')
abs_plot, = ax.plot(abs_x, abs_y, 'r.', alpha=0.3,label='absolute_path(GPS)')
kf_plot, = ax.plot([], [], 'b.', markersize = 3 , label='Kalman Filter predicted(GPS+IMU)')

ax.set_xlabel('X[m]')
ax.set_ylabel('Y[m]')
ax.legend()
ax.grid(True)
ax.axis('equal')

ax_measure = plt.axes([0.25, 0.12, 0.65, 0.03])
ax_process = plt.axes([0.25, 0.07, 0.65, 0.03])

slider_measure = Slider(ax_measure, 'Measurement Variance', 0.1, 10.0, valinit=1.0, valstep=0.1)
slider_process = Slider(ax_process, 'Process Variancd', 0.001, 10.0, valinit=1.0, valstep=0.001)

# update plot
def update(val):
    process_var = slider_process.val
    measurement_var = slider_measure.val
    est_x, est_y = run_kalman_filter(process_var, measurement_var)
    kf_plot.set_data(est_x, est_y)
    ax.set_title(f'GPS-IMU Data Fusion Path Estimate (Process Var: {process_var:.2f}, Measurement Var: {measurement_var:.2f})')
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_process.on_changed(update)
slider_measure.on_changed(update)

# ===== 초기 실행 =====
update(None)
plt.show()


