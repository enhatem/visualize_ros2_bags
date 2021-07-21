from ls2n_tools import rosbag_extract
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
# import casadi as cs


# reading trajectory
ref_traj, ref_U = readTrajectory()

x_ref = ref_traj[:,0]
y_ref = ref_traj[:,1]
z_ref = ref_traj[:,2]

vx_ref = ref_traj[:,7]
vy_ref = ref_traj[:,8]
vz_ref = ref_traj[:,9]



angles_ref = np.zeros((ref_traj.shape[0], 3))

for i in range(ref_traj.shape[0]):
    angles_ref[i, :] = quaternion_to_euler(ref_traj[i, 3:7])

angles_ref = R2D(angles_ref)

phi_ref     = angles_ref[:,0]     # roll
theta_ref   = angles_ref[:,1]     # pitch
psi_ref     = angles_ref[:,2]     # yaw



time = np.empty(0)
x  = np.empty(0)
y  = np.empty(0)
z  = np.empty(0)
vx = np.empty(0)
vy = np.empty(0)
vz = np.empty(0)
qw = np.empty(0)
qx = np.empty(0)
qy = np.empty(0)
qz = np.empty(0)


topics = rosbag_extract.deserialize_rosbag('ros_bags/rosbag2_2021_07_21-18_11_48.bag/rosbag2_2021_07_21-18_11_48_0.db3')
print(f'length of msg: {len(topics)}')
print(f'msg keys: {topics.keys()}')


# print(topics.keys())

time_ = np.empty(0)
starting_command = np.empty(0)
for timestamp, msg in (topics["/Drone1/hover"]):
    time_ = np.append(time_, timestamp)
    starting_command = np.append(starting_command, msg)
# print(msg['/Drone1/RatesThrustSetPoint'])

start_time = -1
for i in range( len(starting_command) ):
    #print(starting_command[i])
    if starting_command[i].data == True :
        start_time = time_[i]
        break



for timestamp, msg in (topics["/Drone1/EKF/odom"]):
        if timestamp >= start_time:
            time = np.append(time, timestamp)
            x  = np.append(x, msg.pose.pose.position.x)
            y  = np.append(y, msg.pose.pose.position.y)
            z  = np.append(z, msg.pose.pose.position.z)

            qx = np.append(qx, msg.pose.pose.orientation.x)
            qy = np.append(qy, msg.pose.pose.orientation.y) 
            qz = np.append(qz, msg.pose.pose.orientation.z)
            qw = np.append(qw, msg.pose.pose.orientation.w)

            vx = np.append(vx, msg.twist.twist.linear.x)
            vy = np.append(vy, msg.twist.twist.linear.y)
            vz = np.append(vz, msg.twist.twist.linear.z)

# converting quaterions to Euler angles
angles_real = np.zeros((time.shape[0], 3))
quat = np.array([qw,qx,qy,qz]).T
for i in range(quat.shape[0]):
    angles_real[i, :] = quaternion_to_euler(quat[i,:])

angles_real = R2D(angles_real)

phi_real     = angles_real[:,0]     # roll
theta_real   = angles_real[:,1]     # pitch
psi_real     = angles_real[:,2]     # yaw


time = time[:len(x_ref)]
x = x[:len(x_ref)]
y = y[:len(y_ref)]
z = z[:len(z_ref)]



phi_real   = phi_real[:len(x_ref)]
theta_real = theta_real[:len(x_ref)]
psi_real   = psi_real[:len(x_ref)]

vx = vx[:len(vx_ref)]
vy = vy[:len(vy_ref)]
vz = vz[:len(vz_ref)]

time = (time -time[0])* 1e-9
# plt.style.use('seaborn')
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.plot(time, x, label='x')
ax1.plot(time, x_ref, label='x_ref')

ax1.set_title('Position States')
ax1.set_ylabel('x [m]')

ax2.plot(time, y, label='y')
ax2.plot(time, y_ref, label='y_ref')

ax2.set_ylabel('y [m]')

ax3.plot(time, z, label='z')
ax3.plot(time, z_ref, label='z_ref')

ax3.set_xlabel('Time [s]')
ax3.set_ylabel('z [m]')

ax1.legend()
ax2.legend()
ax3.legend()

fig2, (ax4, ax5, ax6) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax4.plot(time, vx, label='vx')
ax4.plot(time, vx_ref, label='vx_ref')

ax4.set_title('Velocity States')
ax4.set_ylabel('vx [m/s]')

ax5.plot(time, vy, label='vy')
ax5.plot(time, vy_ref, label='vy_ref')

ax5.set_ylabel('vy [m/s]')

ax6.plot(time, vz, label='vz')
ax6.plot(time, vz_ref, label='vz_ref')

ax6.set_xlabel('Time [s]')
ax6.set_ylabel('vz [m]')

ax4.legend()
ax5.legend()
ax6.legend()

fig3, (ax7, ax8, ax9) = plt.subplots(nrows=3, ncols=1, sharex=True)

ax7.plot(time, phi_real, label='phi')
ax7.plot(time, phi_ref, label='phi_ref')
ax7.set_title('Angles States')
ax7.set_ylabel('$\Phi$ [$^{\circ}$]')

ax8.plot(time, theta_real, label='theta')
ax8.plot(time, theta_ref, label='theta_ref')
ax8.set_ylabel('$\Theta$ [$^{\circ}$]')

ax9.plot(time, psi_real, label='psi')
ax9.plot(time, psi_ref, label='psi_ref')
ax9.set_ylabel('$\Psi$ [$^{\circ}$]')

ax7.legend()
ax8.legend()
ax9.legend()


# plot 3D simulation
fig4, ax10 = plt.subplots()
plt.title('Reference trajectory')
ax10 = plt.axes(projection = "3d")
ax10.plot3D(x, y, z, label='traj')
ax10.plot3D(x_ref, y_ref, z_ref, label='ref_traj')
ax10.set_xlabel("x[m]")
ax10.set_ylabel("y[m]")
ax10.set_zlabel("z[m]")

ax10.legend()


NUM_STEPS = ref_traj.shape[0]
MEAS_EVERY_STEPS = 20

X0 = [x[0], y[0], z[0]]
q0 = [qw[0], qx[0], qy[0], qz[0]]
plotDrone3D(ax10,X0,q0)

for step in range(NUM_STEPS):
        if step !=0 and step % MEAS_EVERY_STEPS ==0:
            X = [x[step], y[step], z[step]]
            q = [qw[step], qx[step], qy[step], qz[step]]
            plotDrone3D(ax10,X,q)




axisEqual3D(ax10)


plt.show()

# print(time)









