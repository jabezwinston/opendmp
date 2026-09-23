"""dmp.quat: quaternion helpers, written in the DMP language (they are inlined like any helper).

A quaternion is a list [w, x, y, z].  The functions assume a unit quaternion, and say so with
`dmp.assume` where interval arithmetic alone could not prove that a value stays in range; the
simulator checks those assumptions on every pass.
"""
import opendmp as dmp


def gravity(q):
    """The gravity direction in the body frame (the rotation matrix's third row) for a unit
    quaternion q: three components in [-1, 1]."""
    w, x, y, z = q
    gx = 2 * dmp.assume(x * z - w * y, -0.5, 0.5)
    gy = 2 * dmp.assume(w * x + y * z, -0.5, 0.5)
    gz = dmp.assume(w * w - x * x - y * y + z * z, -1.0, 1.0)
    return gx, gy, gz


def rate_step(q, h):
    """One integration step, q + q (x) (0, h): h is half the rotation angle this pass (rad)."""
    w, x, y, z = q
    hx, hy, hz = h
    return [w - x * hx - y * hy - z * hz,
            x + w * hx + y * hz - z * hy,
            y + w * hy - x * hz + z * hx,
            z + w * hz + x * hy - y * hx]


def normalize(q):
    """One Newton step toward |q| = 1, q * (1.5 - 0.5 |q|^2): enough for a quaternion that
    drifts only slightly per pass."""
    n = dmp.assume(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 0.9, 1.1)
    k = 1.5 - 0.5 * n
    # |q[i]| <= sqrt(n) and sqrt(n) * (1.5 - n/2) peaks at exactly 1 (n = 1): unit components
    return [dmp.assume(q[0] * k, -1.0, 1.0), dmp.assume(q[1] * k, -1.0, 1.0),
            dmp.assume(q[2] * k, -1.0, 1.0), dmp.assume(q[3] * k, -1.0, 1.0)]
