"""
Travel time calculation module for Traffic-Based Route Guidance System (TBRGS).

This module provides:
  - A function to convert traffic flow and segment distance into travel time
    using the PDF v1.0 procedure (free-flow and congested conditions).
"""

import math


def flow_to_time(
    flow: float,
    distance_km: float,
    a: float = 1.4648375,
    b: float = 93.75,
    speed_limit: float = 60.0,
    capacity_flow: float = 351.0,
    delay_s: float = 30.0,
) -> float:
    """
    Convert traffic flow (veh/h) and segment length (km) to travel time (seconds).

    The method follows PDF v1.0:
      1. If flow <= capacity_flow, use free-flow speed = speed_limit.
      2. Otherwise, solve the quadratic equation a*s^2 - b*s + flow = 0
         and choose the smaller root for congested speed.
      3. Compute travel time: (distance_km / speed_kmh) * 3600 + delay_s.

    Args:
        flow (float): Traffic flow in vehicles per hour.
        distance_km (float): Length of the segment in kilometers.
        a (float): Quadratic coefficient for congested speed model.
        b (float): Linear coefficient for congested speed model.
        speed_limit (float): Free-flow speed limit in km/h.
        capacity_flow (float): Flow threshold for free-flow condition (veh/h).
        delay_s (float): Fixed delay added to travel time in seconds.

    Returns:
        float: Estimated travel time for the segment in seconds.
    """
    # Determine speed based on free-flow or congested conditions
    if flow <= capacity_flow:
        speed = speed_limit
    else:
        # Calculate discriminant for quadratic formula
        disc = b * b - 4 * a * flow
        sqrt_disc = math.sqrt(max(disc, 0.0))
        # Compute both roots
        s1 = (b - sqrt_disc) / (2 * a)
        s2 = (b + sqrt_disc) / (2 * a)
        # Choose the smaller root as congested speed
        speed = min(s1, s2)

    # Convert speed (km/h) to travel time: hours → seconds, then add fixed delay
    travel_time_s = (distance_km / speed) * 3600.0 + delay_s
    return travel_time_s
