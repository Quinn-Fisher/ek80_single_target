"""Beam compensation utilities for split-beam single target detection."""


def compute_beam_compensation(
    angle_along: float,
    angle_athwart: float,
    bw_along: float,
    bw_athwart: float,
    max_gain_db: float,
) -> float:
    """
    Compute Simrad-style beam pattern compensation in dB.

    Parameters are mechanical angles in degrees and EK80 two-way beam widths in degrees.
    """
    # Soule/Simrad compensation uses one-way beamwidth in the denominator.
    # EK80 metadata is commonly stored as two-way beamwidth, so convert here.
    bw_along_oneway = float(bw_along) / 2.0
    bw_athwart_oneway = float(bw_athwart) / 2.0
    compensation_db = 6.0206 * ((angle_along / bw_along_oneway) ** 2 + (angle_athwart / bw_athwart_oneway) ** 2)
    return min(float(compensation_db), float(max_gain_db))

