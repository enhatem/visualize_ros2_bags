from ls2n_tools import rosbag_extract
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import casadi as cs


def plotDrone3D(ax,X,q):
    
    l= 0.046 # arm length
    r = 0.02 # rotor length
    # l = 0.1
    # r = 0.04

    x = X[0]
    y = X[1]
    z = X[2]

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R = np.array([[1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]])

    c1 = np.array([x,y,z]) + R @ np.array([r,0,0])
    q1 = np.array([x,y,z]) + R @ np.array([l,l,0])
    q2 = np.array([x,y,z]) + R @ np.array([-l,-l,0])
    q3 = np.array([x,y,z]) + R @ np.array([l,-l,0])
    q4 = np.array([x,y,z]) + R @ np.array([-l,l,0])

    r1 = q1 + R @ np.array([0,0,r])
    r2 = q2 + R @ np.array([0,0,r])
    r3 = q3 + R @ np.array([0,0,r])
    r4 = q4 + R @ np.array([0,0,r])

    ax.plot3D([q1[0], q2[0]], [q1[1], q2[1]], [q1[2], q2[2]], 'k')
    ax.plot3D([q3[0], q4[0]], [q3[1], q4[1]], [q3[2], q4[2]], 'k')
    ax.plot3D([q1[0], r1[0]], [q1[1], r1[1]], [q1[2], r1[2]], 'r')
    ax.plot3D([q2[0], r2[0]], [q2[1], r2[1]], [q2[2], r2[2]], 'r')
    ax.plot3D([q3[0], r3[0]], [q3[1], r3[1]], [q3[2], r3[2]], 'r')
    ax.plot3D([q4[0], r4[0]], [q4[1], r4[1]], [q4[2], r4[2]], 'r')
    ax.plot3D([x, c1[0]], [y, c1[1]], [z, c1[2]], '-', color='orange', label='heading')

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def readTrajectory():
        
        # import csv file of measX and simU (noisy measurement)
        ref_traj = pd.read_csv('/home/elie/ros_ws/src/ls2n_drone_ros2/ls2n_crazyflie_nmpc/traj/pol_9/J=u1/measX.csv')
        ref_U = pd.read_csv('/home/elie/ros_ws/src/ls2n_drone_ros2/ls2n_crazyflie_nmpc/traj/pol_9/J=u1/simU.csv')

        # ref_traj = pd.read_csv('/home/elie/ros_ws/src/ls2n_drone_ros2/ls2n_crazyflie_nmpc/traj/pol_9/patternsearch/measX.csv')
        # ref_U = pd.read_csv('/home/elie/ros_ws/src/ls2n_drone_ros2/ls2n_crazyflie_nmpc/traj/pol_9/patternsearch/simU.csv')


        # hovering time in the beginning of the trajectory
        T_hover = 2
        N = 50

        # create references to add for the hovering time
        ref_traj_x0 = ref_traj.iloc[[0]*N*T_hover]
        ref_u0 = ref_U.iloc[[0]*N*T_hover]

        # insert hovering references and inputs into their respective dataframes
        ref_traj = pd.concat([pd.DataFrame(ref_traj_x0), ref_traj], ignore_index=True)
        ref_U = pd.concat([pd.DataFrame(ref_u0), ref_U], ignore_index=True)

        # append last reference point 3*N times to increase the length of the trajectory
        ref_traj = ref_traj.append( ref_traj.iloc[[-1]*N*3] )
        ref_U = ref_U.append( ref_U.iloc[[-1]*N*3] )

        # convert data frames to numpy arrays
        ref_traj = ref_traj[['y', 'z', 'phi', 'vy', 'vz', 'phi_dot']].to_numpy()
        ref_U = ref_U[['Thrust', 'Torque']].to_numpy()

        Thrust_ref = ref_U[:,0] * 2 # multiplied by 2 since the trajectory was based on a planar drone (mass/2)
        Torque_ref = ref_U[:,1] * 2 # not used but multiplied by 2 for consistency

        ref_U = np.array([Thrust_ref, Torque_ref]).T

        # extract each element of the trajectory
        x_ref     = np.zeros_like(ref_traj[:,0])
        y_ref     = ref_traj[:,0]
        z_ref     = ref_traj[:,1]
        phi_ref   = ref_traj[:,2]
        theta_ref = np.zeros_like(phi_ref)
        psi_ref   = np.zeros_like(phi_ref)
        vx_ref    = np.zeros_like(ref_traj[:,0])
        vy_ref    = ref_traj[:,3]
        vz_ref    = ref_traj[:,4]

        quat_ref = euler_to_quaternion(phi_ref,theta_ref,psi_ref).T

        # Number of rows in the trajectory
        rows = phi_ref.shape[0]

        # Ensure the unit quaternion at each iteration
        for i in range(rows):
            quat_ref[i] = unit_quat(quat_ref[i])

        # Extract the elements of the quaternion
        qw_ref = quat_ref[:,0]
        qx_ref = quat_ref[:,1]
        qy_ref = quat_ref[:,2]
        qz_ref = quat_ref[:,3]

        ref_traj = np.array([x_ref, y_ref, z_ref, qw_ref, qx_ref, qy_ref, qz_ref, vx_ref, vy_ref, vz_ref]).T

        return ref_traj , ref_U

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def R2D(rad):
    return rad*180 / np.pi

def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q




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

# msg = deserialize_rosbag('rosbag2_2021-07-12-15-24-56.bag/rosbag2_2021-07-12-15-24-56.bag_0.db3')

topics = rosbag_extract.deserialize_rosbag('/home/elie/ros_ws/src/ls2n_drone_ros2/ls2n_tools/ls2n_tools/ros_bags/rosbag2_2021_07_21-18_11_48.bag/rosbag2_2021_07_21-18_11_48_0.db3')
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









