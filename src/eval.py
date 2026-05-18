"""
eval.py — Performance Evaluation.

Evaluates the performance of the algorithm against ground truth.
Visualizes and presents data for inspection and reporting.
"""

from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jaxlie


log = logging.getLogger(__name__)

def _plot_state_diagnostics(results: pd.DataFrame, output_path: Path) -> None:
    """Estimator state diagnostics: gravity, disturbance, point counts."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    t_s = results['timestamp_s']
    g_norm_nominal = 9.7935   # from ekf_meas.g_norm; consistent with calib

    # Gravity vector — per-axis + magnitude
    ax = axes[0]
    for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
        ax.plot(t_s, results[f'g{axis}'], color=color, linewidth=1.0,
                label=f'g{axis}', alpha=0.85)
    g_mag = np.sqrt(results['gx']**2 + results['gy']**2 + results['gz']**2)
    ax.plot(t_s, g_mag, 'k-', linewidth=1.5, label='‖g‖', alpha=0.9)
    ax.axhline(y=g_norm_nominal, color='gray', linestyle='--', alpha=0.5,
               label=f'nominal ({g_norm_nominal:.4f})')
    ax.set_ylabel('Gravity [m/s²]')
    ax.set_title('Body-frame gravity estimate')
    ax.legend(ncol=5, fontsize=8); ax.grid(True, alpha=0.3)

    # Disturbance
    ax = axes[1]
    for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
        ax.plot(t_s, results[f'd{axis}'], color=color, linewidth=1.0,
                label=f'd{axis}', alpha=0.85)
    d_mag = np.sqrt(results['dx']**2 + results['dy']**2 + results['dz']**2)
    ax.plot(t_s, d_mag, 'k-', linewidth=1.5, label='‖d‖', alpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Disturbance [m/s²]')
    ax.set_title('Body-frame disturbance estimate')
    ax.legend(ncol=4, fontsize=8); ax.grid(True, alpha=0.3)

    # Point counts
    ax = axes[2]
    ax.plot(t_s, results['n_F'],     'b-', linewidth=1.2, label='F (EKF features)')
    ax.plot(t_s, results['n_F_pre'], 'g-', linewidth=1.2, label='F_pre (pre-admission)')
    ax.plot(t_s, results['n_I'],     'r-', linewidth=1.2, label='I (interest)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Count')
    ax.set_title('Point-set sizes per frame')
    ax.legend(ncol=3, fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _plot_trajectory(merged: pd.DataFrame, output_path: Path) -> None:
    """Side-by-side trajectory plot: estimated vs ground truth."""
    fig = plt.figure(figsize=(16, 6))
    
    # 3D trajectory
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(merged['x_est'], merged['y_est'], merged['z_est'],
            'b-', linewidth=1.5, label='Estimated', alpha=0.85)
    ax.plot(merged['x_gt'],  merged['y_gt'],  merged['z_gt'],
            'g--', linewidth=1.5, label='Ground truth', alpha=0.85)
    ax.scatter(merged['x_est'].iloc[0], merged['y_est'].iloc[0],
               merged['z_est'].iloc[0], c='blue', s=80, marker='o', label='Start')
    ax.scatter(merged['x_est'].iloc[-1], merged['y_est'].iloc[-1],
               merged['z_est'].iloc[-1], c='red',  s=80, marker='x', label='End')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    # Per-axis time series
    ax2 = fig.add_subplot(322)
    t_s = merged['timestamp_s_est']
    for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
        ax2.plot(t_s, merged[f'{axis}_est'], color=color, linewidth=1.0,
                 label=f'{axis} est', alpha=0.85)
        ax2.plot(t_s, merged[f'{axis}_gt'],  color=color, linewidth=1.0,
                 linestyle='--', alpha=0.7)
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Per-axis position (solid=est, dashed=gt)')
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Velocity time series  
    ax3 = fig.add_subplot(324)
    for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
        ax3.plot(t_s, merged[f'v{axis}_est'], color=color, linewidth=1.0,
                 label=f'v{axis} est', alpha=0.85)
        ax3.plot(t_s, merged[f'v{axis}_gt'],  color=color, linewidth=1.0,
                 linestyle='--', alpha=0.7)
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_title('Body-frame velocity (solid=est, dashed=gt)')
    ax3.legend(ncol=3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Angular velocity time series
    ax4 = fig.add_subplot(326)
    for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
        ax4.plot(t_s, merged[f'w{axis}_est'], color=color, linewidth=1.0,
                 label=f'w{axis} est', alpha=0.85)
        ax4.plot(t_s, merged[f'w{axis}_gt'],  color=color, linewidth=1.0,
                 linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular velocity [rad/s]')
    ax4.set_title('Body-frame angular velocity')
    ax4.legend(ncol=3, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate(results_path: Path,
             ground_truth_path: Path) -> dict:
    """Compare estimator output against Vicon ground truth.

    Reads the per-frame results written by main.py and the time-aligned
    ground-truth pose/velocity table from preprocessing. Aligns the two
    by camera timestamp, computes error metrics, and emits plots.

    Args:
        results_path      : CSV from main.py with per-frame
                            estimator output (timestamp_ns, pose, v_B,
                            ω_B, plus optional diagnostics).
        ground_truth_path : body_pose.csv from fetch_vid.py with
                            time-aligned ground truth.


    Notes:
        - Suggestions: translation magnitude RMSE, rotation
           magnitude RMSE (In tangent space), body-frame velocity RMSE per axis, body-frame
           angular velocity RMSE per axis. Per-frame errors and plots
           written to disk alongside results_path.

        - The vicon-marker → body offset T_{M,B} is unknown; per-axis
          comparisons in body frame are therefore biased by a constant if our guess was wrong.
          Compare rotation-invariant magnitudes (‖t‖, rotation
          angle, ‖v‖) for honest accuracy numbers; per-axis components
          are useful for trend-spotting only.
        - Filter output is in T_{B_k, B_0} convention; ground truth uses
          the same convention (preprocessing writes S_k that way).
          No conversion needed before comparison.
    """
    log.info(f"Loading results from {results_path}")
    results = pd.read_csv(results_path)
    
    log.info(f"Loading ground truth from {ground_truth_path}")
    gt = pd.read_csv(ground_truth_path)
    
    # Align by timestamp_ns
    merged = pd.merge(results, gt, on='timestamp_ns', suffixes=('_est', '_gt'))
    log.info(f"Aligned {len(merged)}/{len(results)} frames "
         f"({len(results) - len(merged)} unmatched timestamps)")
    if len(merged) == 0:
        log.error("No matching timestamps between results and ground truth")
        return {}
    
    log.info(f"Aligned {len(merged)} frames")
    
    # Initialize error arrays
    n_frames = len(merged)
    trans_errors = np.zeros(n_frames)
    rot_angle_errors = np.zeros(n_frames)
    vel_errors = np.zeros((n_frames, 3))
    vel_mag_errors = np.zeros(n_frames)
    angvel_errors = np.zeros((n_frames, 3))
    angvel_mag_errors = np.zeros(n_frames)
    
    # Compute per-frame errors
    for k in range(n_frames):
        row = merged.iloc[k]
        
        # ---- Pose errors ----
        # Estimated pose
        t_est = np.array([row['x_est'], row['y_est'], row['z_est']])
        q_est = np.array([row['qx_est'], row['qy_est'], row['qz_est'], row['qw_est']])
        T_est = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(q_est),
            t_est
        )
        
        # Ground truth pose
        t_gt = np.array([row['x_gt'], row['y_gt'], row['z_gt']])
        q_gt = np.array([row['qx_gt'], row['qy_gt'], row['qz_gt'], row['qw_gt']])
        T_gt = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(q_gt),
            t_gt
        )
        
        # Relative pose error: T_error = T_est @ T_gt^{-1}
        T_error = T_est @ T_gt.inverse()
        
        # Translation error magnitude
        trans_error = np.linalg.norm(T_error.translation())
        trans_errors[k] = trans_error
        
        # Rotation error: angle of rotation (magnitude of log map)
        rot_error = np.linalg.norm(T_error.rotation().log())
        rot_angle_errors[k] = rot_error
        
        # ---- Velocity errors ----
        v_est = np.array([row['vx_est'], row['vy_est'], row['vz_est']])
        v_gt = np.array([row['vx_gt'], row['vy_gt'], row['vz_gt']])
        vel_error = v_est - v_gt
        vel_errors[k, :] = vel_error
        vel_mag_errors[k] = np.linalg.norm(vel_error)
        
        # ---- Angular velocity errors ----
        w_est = np.array([row['wx_est'], row['wy_est'], row['wz_est']])
        w_gt = np.array([row['wx_gt'], row['wy_gt'], row['wz_gt']])
        angvel_error = w_est - w_gt
        angvel_errors[k, :] = angvel_error
        angvel_mag_errors[k] = np.linalg.norm(angvel_error)
    
    # Compute RMSE metrics
    trans_rmse = np.sqrt(np.mean(trans_errors**2))
    trans_p50  = np.median(trans_errors)
    trans_p95  = np.percentile(trans_errors, 95)   
    rot_rmse = np.sqrt(np.mean(rot_angle_errors**2))
    vel_rmse_x = np.sqrt(np.mean(vel_errors[:, 0]**2))
    vel_rmse_y = np.sqrt(np.mean(vel_errors[:, 1]**2))
    vel_rmse_z = np.sqrt(np.mean(vel_errors[:, 2]**2))
    vel_mag_rmse = np.sqrt(np.mean(vel_mag_errors**2))
    angvel_rmse_x = np.sqrt(np.mean(angvel_errors[:, 0]**2))
    angvel_rmse_y = np.sqrt(np.mean(angvel_errors[:, 1]**2))
    angvel_rmse_z = np.sqrt(np.mean(angvel_errors[:, 2]**2))
    angvel_mag_rmse = np.sqrt(np.mean(angvel_mag_errors**2))
 
    # Log summary
    log.info("=== Error Metrics ===")
    log.info(f"Translation: RMSE={trans_rmse:.4f}, median={trans_p50:.4f}, p95={trans_p95:.4f} m")
    log.info(f"Rotation RMSE:           {rot_rmse:.6f} rad ({np.degrees(rot_rmse):.4f}°)")
    log.info(f"Velocity magnitude RMSE: {vel_mag_rmse:.6f} m/s")
    log.info(f"  - vx RMSE:             {vel_rmse_x:.6f} m/s")
    log.info(f"  - vy RMSE:             {vel_rmse_y:.6f} m/s")
    log.info(f"  - vz RMSE:             {vel_rmse_z:.6f} m/s")
    log.info(f"Angular velocity RMSE:   {angvel_mag_rmse:.6f} rad/s")
    log.info(f"  - wx RMSE:             {angvel_rmse_x:.6f} rad/s")
    log.info(f"  - wy RMSE:             {angvel_rmse_y:.6f} rad/s")
    log.info(f"  - wz RMSE:             {angvel_rmse_z:.6f} rad/s")
    
    # Save per-frame errors
    errors_df = pd.DataFrame({
        'timestamp_ns': merged['timestamp_ns'],
        'trans_error': trans_errors,
        'trans_err_median': trans_p50,
        'trans_err_p95': trans_p95, 
        'rot_angle_error': rot_angle_errors,
        'vel_error_x': vel_errors[:, 0],
        'vel_error_y': vel_errors[:, 1],
        'vel_error_z': vel_errors[:, 2],
        'vel_mag_error': vel_mag_errors,
        'angvel_error_x': angvel_errors[:, 0],
        'angvel_error_y': angvel_errors[:, 1],
        'angvel_error_z': angvel_errors[:, 2],
        'angvel_mag_error': angvel_mag_errors,
    })
    
    errors_path = results_path.parent / f"{results_path.stem}_errors.csv"
    errors_df.to_csv(errors_path, index=False)
    log.info(f"Saved per-frame errors to {errors_path}")
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    time_s = merged['timestamp_s_est'].values
    
    # Translation error
    axes[0, 0].plot(time_s, trans_errors, 'b-', linewidth=1)
    axes[0, 0].axhline(y=trans_rmse, color='r', linestyle='--', label=f'RMSE: {trans_rmse:.6f} m')
    axes[0, 0].set_ylabel('Translation Error (m)')
    axes[0, 0].set_title('Translation Error Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Rotation error
    axes[0, 1].plot(time_s, np.degrees(rot_angle_errors), 'g-', linewidth=1)
    axes[0, 1].axhline(y=np.degrees(rot_rmse), color='r', linestyle='--', label=f'RMSE: {np.degrees(rot_rmse):.4f}°')
    axes[0, 1].set_ylabel('Rotation Error (degrees)')
    axes[0, 1].set_title('Rotation Error Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Velocity error components
    axes[1, 0].plot(time_s, vel_errors[:, 0], label='vx error', linewidth=1)
    axes[1, 0].plot(time_s, vel_errors[:, 1], label='vy error', linewidth=1)
    axes[1, 0].plot(time_s, vel_errors[:, 2], label='vz error', linewidth=1)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_ylabel('Velocity Error (m/s)')
    axes[1, 0].set_title('Velocity Error Components')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Velocity error magnitude
    axes[1, 1].plot(time_s, vel_mag_errors, 'c-', linewidth=1)
    axes[1, 1].axhline(y=vel_mag_rmse, color='r', linestyle='--', label=f'RMSE: {vel_mag_rmse:.6f} m/s')
    axes[1, 1].set_ylabel('Velocity Error Magnitude (m/s)')
    axes[1, 1].set_title('Velocity Error Magnitude Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Angular velocity error components
    axes[2, 0].plot(time_s, angvel_errors[:, 0], label='wx error', linewidth=1)
    axes[2, 0].plot(time_s, angvel_errors[:, 1], label='wy error', linewidth=1)
    axes[2, 0].plot(time_s, angvel_errors[:, 2], label='wz error', linewidth=1)
    axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Angular Velocity Error (rad/s)')
    axes[2, 0].set_title('Angular Velocity Error Components')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Angular velocity error magnitude
    axes[2, 1].plot(time_s, angvel_mag_errors, 'm-', linewidth=1)
    axes[2, 1].axhline(y=angvel_mag_rmse, color='r', linestyle='--', label=f'RMSE: {angvel_mag_rmse:.6f} rad/s')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Angular Velocity Error Magnitude (rad/s)')
    axes[2, 1].set_title('Angular Velocity Error Magnitude Over Time')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.tight_layout()
    
    plot_path = results_path.parent / f"{results_path.stem}_errors.png"
    plt.savefig(plot_path, dpi=150)
    log.info(f"Saved plot to {plot_path}")
    plt.close()
    
    traj_path = results_path.parent / f"{results_path.stem}_trajectory.png"
    _plot_trajectory(merged, traj_path)
    log.info(f"Saved trajectory plot to {traj_path}")
    

    diag_path = results_path.parent / f"{results_path.stem}_diagnostics.png"
    _plot_state_diagnostics(results, diag_path)
    log.info(f"Saved diagnostics plot to {diag_path}")
    
    # Return summary metrics
    metrics = {
        'n_frames': n_frames,
        'trans_rmse': float(trans_rmse),
        'trans_err_median': float(trans_p50),
        'trans_err_p95': float(trans_p95), 
        'rot_rmse_rad': float(rot_rmse),
        'rot_rmse_deg': float(np.degrees(rot_rmse)),
        'vel_mag_rmse': float(vel_mag_rmse),
        'vel_rmse_x': float(vel_rmse_x),
        'vel_rmse_y': float(vel_rmse_y),
        'vel_rmse_z': float(vel_rmse_z),
        'angvel_mag_rmse': float(angvel_mag_rmse),
        'angvel_rmse_x': float(angvel_rmse_x),
        'angvel_rmse_y': float(angvel_rmse_y),
        'angvel_rmse_z': float(angvel_rmse_z),
    }
    
    return metrics