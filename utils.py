import numpy as np
import pandas as pd

from pyquaternion import Quaternion

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
        ref_traj = pd.read_csv('/home/elie/CORO-IMARO/M2/Semester2/Coding/my_master_thesis/visualize_ros2_bags/reference_trajectories/globalsearch_1/measX.csv')
        ref_U = pd.read_csv('/home/elie/CORO-IMARO/M2/Semester2/Coding/my_master_thesis/visualize_ros2_bags/reference_trajectories/globalsearch_1/simU.csv')

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