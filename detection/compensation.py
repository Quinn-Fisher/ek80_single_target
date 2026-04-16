"""Beam compensation utilities for split-beam single target detection."""


def compute_beam_compensation(
    angle_along: float,
    angle_athwart: float,
    bw_along: float,
    bw_athwart: float,
    max_gain_db: float,
) -> float:
    """
    Compute two-way Simrad beam pattern compensation in dB.

    Parameters are mechanical angles and beam widths in degrees.
    """
    compensation_db = 6.0206 * ((angle_along / bw_along) ** 2 + (angle_athwart / bw_athwart) ** 2)
    return min(float(compensation_db), float(max_gain_db))

