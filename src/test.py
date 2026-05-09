import jax.numpy as jnp
from modules.ekf import project_point, relative_pose
from modules.utils import load_yaml
import jaxlie
import jax 
import jax.numpy as jnp
import numpy as np

calib = load_yaml("constants/calibration.yaml")

# =============================================================================
# project_point — ad-hoc tests
# =============================================================================
print("=" * 72)
print("project_point")
print("=" * 72)
 
p = jnp.array([1.0, 0.0, 0.0])
print("L:", project_point(p, calib, "L")[0])
print("R:", project_point(p, calib, "R")[0])
 
uv_L = project_point(p, calib, "L")[0]
uv_R = project_point(p, calib, "R")[0]
disp = float(uv_L[0] - uv_R[0])
expected = 0.05 * 384.18206787109375 / (1.0 - 0.1844)
print(f"  disparity = {disp:.3f} px  (expected b·fx/Z_cam = {expected:.3f} ✓)")
 
# NOTE: body frame is FRD — x is forward, y is right, z is down. Points
# must have x > 0.1844 to be in front of the (forward-mounted) camera.
pts = jnp.array([[1.0, 0.0,  0.0],
                 [1.5, 0.1, -0.05],
                 [3.0, 0.2,  0.1]])
print("\nbatch L:")
print(project_point(pts, calib, "L"))
 
 
# =============================================================================
# relative_pose — sanity checks
# =============================================================================
print()
print("=" * 72)
print("relative_pose")
print("=" * 72)
 
print("\nTest 1: independent poses (Φ=0 ⇒ cross=0, T_a=I ⇒ Ad=I)")
n_a = 18
T_a = jaxlie.SE3.identity()
T_b = jaxlie.SE3.identity()
P_a = jnp.eye(n_a) * 1e-3
P_b = jnp.eye(n_a) * 4e-3
Phi = jnp.zeros((n_a, n_a))
dT, Sigma = relative_pose(T_a, T_b, P_a, P_b, Phi)
diag = np.diag(np.array(Sigma))
expected = 1e-3 + 4e-3
ok = np.allclose(diag, expected, atol=1e-9)
print(f"  Σ diag = {diag}")
print(f"  expected {expected} on every entry  {'✓' if ok else '✗'}")
 
print("\nTest 2: perfectly correlated poses (Φ=I, identical covars)")
P_a = jnp.eye(n_a) * 1e-3
P_b = jnp.eye(n_a) * 1e-3
Phi = jnp.eye(n_a)
dT, Sigma = relative_pose(T_a, T_b, P_a, P_b, Phi)
norm = float(jnp.linalg.norm(Sigma))
print(f"  ||Σ||_F = {norm:.2e}  (should be ≈ 0  {'✓' if norm < 1e-9 else '✗'})")
 
print("\nTest 3: ΔT for pose change (T_a=I, T_b=Tx(1)·Rz(90°))")
T_a = jaxlie.SE3.identity()
R = jaxlie.SO3.from_z_radians(jnp.pi / 2)
T_b = jaxlie.SE3.from_rotation_and_translation(R, jnp.array([1.0, 0.0, 0.0]))
dT, Sigma = relative_pose(T_a, T_b,
                           jnp.eye(n_a)*1e-6, jnp.eye(n_a)*1e-6,
                           jnp.eye(n_a))
t = np.array(dT.translation())
phi = np.array(dT.rotation().log())
print(f"  ΔT t = {t}  (expected [1, 0, 0])")
print(f"  ΔT log(R) = {phi}  (expected [0, 0, π/2 ≈ {np.pi/2:.4f}])")
 
print("\nTest 4: post-update path, n_a=24 (2 features), n_b=21 (id 42 marginalised)")
n_a, n_b = 24, 21
# Use a self-consistent flow so the joint covariance is genuinely PSD:
#   Φ = I, Q = 0  ⇒  P_k^- = P_a
#   marginalise id 42  ⇒  P_marg = P_a restricted to surviving rows = 1e-3 I_{21}
#   (I - K H) = 0.9 I  ⇒  P_b = 0.9 · 1e-3 = 9e-4 I_{21}
P_a = jnp.eye(n_a) * 1e-3
P_b = jnp.eye(n_b) * 9e-4
Phi = jnp.eye(n_a)
IKH = jnp.eye(n_b) * 0.9
a_ids = [42, 99]
b_ids = [99]                                    # id 42 marginalised
T_a = jaxlie.SE3.from_rotation_and_translation(
    jaxlie.SO3.identity(), jnp.array([0.5, 0.0, 0.0]))
T_b = jaxlie.SE3.from_rotation_and_translation(
    jaxlie.SO3.identity(), jnp.array([0.7, 0.0, 0.0]))
dT, Sigma = relative_pose(T_a, T_b, P_a, P_b, Phi, IKH, a_ids, b_ids)
print(f"  ΔT t = {np.array(dT.translation())}  (expected [0.2, 0, 0])")
print(f"  Σ shape = {tuple(Sigma.shape)}  (expected (6, 6))")
sym_err = float(jnp.linalg.norm(Sigma - Sigma.T))
eigs = jnp.linalg.eigvalsh(Sigma)
print(f"  symmetric: ||Σ-Σᵀ|| = {sym_err:.2e}")
print(f"  eigenvalues: {np.array(eigs)}")
print(f"  positive semidefinite: {bool(jnp.all(eigs > -1e-9))}")
 
print("\nTest 5: post-update raises on missing id in a")
try:
    _ = relative_pose(T_a, T_b, P_a, P_b, Phi, IKH, [42], [99])
    print("  ✗ should have raised")
except ValueError as e:
    print(f"  ✓ raised: {e}")