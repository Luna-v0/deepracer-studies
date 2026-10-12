"""
All the function bellow where re implemented on torch from the original library:

https://github.com/Machine-Learning-Labs/DeepRacerRewardFunctionsCollection

Each one returns the step reward itself as an (N,) tensor, so they are wired
with ``RewardShaping(fn=..., scales={})`` (ADR 0002).
"""

import math

import torch

from deepracer_genesis.envs.rules import is_off_track

#: reward bounds the upstream progress-scaled functions clamp to
REWARD_MAX = 1e5
REWARD_MIN = -1e5


def lap_bonus_baseline(env) -> torch.Tensor:
    """Sparse baseline: pays 1.0 only on the step the car crosses the line.

    ``progress_m`` wraps at the finish line, so a crossing shows up as the
    previous position floor-dividing to -1 while the current one is 0.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    lap_length = env.track.total_len_env
    previous = env.progress_m - env.d_progress

    crossed = (env.progress_m // lap_length) > (previous // lap_length)

    return torch.where(crossed, 1.0, 0.0)


def single_aws_reward(env) -> torch.Tensor:
    """Rewards driving close to the center line, in three fixed metre bands.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    distance_from_center = env.lateral.abs()

    reward = torch.full_like(distance_from_center, 1e-3)

    further_distance = distance_from_center <= 0.05
    middle_distance = distance_from_center <= 0.03
    closest_distance = distance_from_center <= 0.02

    # widest band first: each assignment overwrites the previous one
    reward[further_distance] = 0.1
    reward[middle_distance] = 0.3
    reward[closest_distance] = 1.0

    return reward


def sparsed_wheels_on_track(env) -> torch.Tensor:
    """Pays a flat reward whenever every wheel is still on the track.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)

    return torch.where(all_wheels_on_track, 1.0, 1e-3)


def classic_center_line(env) -> torch.Tensor:
    """The classic AWS center line reward, in bands relative to track width.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()

    reward = torch.full_like(distance_from_center, 1e-3)

    marker1 = distance_from_center < (0.1 * track_width)
    marker2 = distance_from_center < (0.25 * track_width)
    marker3 = distance_from_center < (0.5 * track_width)

    # widest marker first, otherwise marker3 would flatten every band to 0.1
    reward[marker3] = 0.1
    reward[marker2] = 0.5
    reward[marker1] = 1.0

    return reward


def inside_two_borders(env) -> torch.Tensor:
    """Rewards keeping every wheel on track and clear of either border.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    distance_from_center = env.lateral.abs()
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)

    overall_distance = (env.half_width - distance_from_center) > 0.05

    between_borders = all_wheels_on_track & overall_distance

    return torch.where(between_borders, 1.0, 1e-3)


def mitigate_zig_zag(env) -> torch.Tensor:
    """Center line reward with a flat penalty for steering past 15 degrees.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    steering = env.actions[:, 0].abs() * 30.0
    reward = classic_center_line(env)

    steering_too_much = steering > 15.0

    return torch.where(steering_too_much, reward * 0.8, reward)


def mitigating_zig_zag_slow_speed_edging(env) -> torch.Tensor:
    """Rewards the inner third of the track, penalising hard steering and crawling.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    throttle = env.v_forward
    distance_from_center = env.lateral.abs()
    third_of_track = env.half_width * (2 / 3)
    steering = env.actions[:, 0].abs() * 30.0

    reward = torch.full_like(distance_from_center, 1e-3)

    distance_of_third = distance_from_center <= third_of_track

    slow_speed = throttle < 0.5

    steering_too_much = steering > 15.0

    reward[distance_of_third] = 1.0
    reward[steering_too_much] *= 0.8
    reward[slow_speed] *= 0.8

    return reward


def mitigate_zig_zag_with_granular(env) -> torch.Tensor:
    """Banded center line reward with graded steering and a slow-speed penalty.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    throttle = env.v_forward
    steering = env.actions[:, 0].abs()

    reward = torch.full_like(distance_from_center, 1e-3)

    marker1 = distance_from_center < (0.25 * track_width)
    marker2 = distance_from_center < (0.3 * track_width)
    marker3 = distance_from_center < (0.5 * track_width)

    steering_a_bit = steering > 0.3
    steering_too_much = steering > 0.5

    slow_speed = throttle < 0.5

    # widest marker first, otherwise marker3 would flatten every band to 0.1
    reward[marker3] = 0.1
    reward[marker2] = 0.2
    reward[marker1] = 1.0

    # strictest steering band first: > 0.5 is also > 0.3, so testing the loose
    # one first would make the 0.6 penalty unreachable
    reward = torch.where(
        steering_too_much,
        reward * 0.6,
        torch.where(
            steering_a_bit,
            reward * 0.8,
            reward
        )
    )

    reward[slow_speed] *= 0.8

    return reward


def mitigate_zig_zag_with_progress(env) -> torch.Tensor:
    """Center line reward with a steering penalty, scaled by lap progress.

    ``progress_m`` is the position on the lap, so this matches AWS's progress
    only when the env spawns at the start line (``random_start=False``).

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    reward = mitigate_zig_zag(env)
    progress = env.progress_m / env.track.total_len_env * 100.0

    started = progress >= 1.0
    lap_done = progress >= 99.0

    reward = torch.where(started, reward * REWARD_MAX * (progress / 100.0), reward)
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)

    return reward.clamp(REWARD_MIN, REWARD_MAX)


def progress_tiers_inside_borders(env) -> torch.Tensor:
    """Border keeping paid in progress tiers, with a hard penalty for going off.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    distance_from_center = env.lateral.abs()
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env * 100.0
    speed = env.v_forward

    reward = torch.full_like(distance_from_center, 1e-3)

    between_borders = all_wheels_on_track & (
        (env.half_width - distance_from_center) >= 0.05)

    reward[between_borders] = 1.0

    # lowest tier first: each tier overwrites the one below it
    reward = torch.where(progress < 10.0, reward * (speed / 5.0), reward)
    for threshold, divisor in ((10.0, 16.0), (25.0, 8.0), (50.0, 4.0), (75.0, 2.0)):
        tier = progress >= threshold
        reward = torch.where(tier, (REWARD_MAX / divisor) * (progress / 100.0),
                             reward)

    lap_done = progress >= 99.0
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)

    # off track overrides every tier above
    return torch.where(all_wheels_on_track, reward,
                       torch.full_like(reward, REWARD_MIN))


def center_line_with_direction_guard(env) -> torch.Tensor:
    """Center line bands, voided when driving reversed or off track.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)
    steering = env.actions[:, 0].abs() * 30.0

    reward = torch.full_like(distance_from_center, 1e-3)

    marker1 = distance_from_center <= (0.1 * track_width)
    marker2 = distance_from_center <= (0.25 * track_width)
    marker3 = distance_from_center <= (0.5 * track_width)

    reward[marker3] = 0.1
    reward[marker2] = 0.5
    reward[marker1] = 1.0

    steering_too_much = steering > 30.0
    slow_speed = env.v_forward < 4.5

    reward[steering_too_much] *= 0.5
    reward[slow_speed] *= 0.5

    is_reversed = env.dir_sign < 0
    voided = is_reversed | ~all_wheels_on_track

    return torch.where(voided, torch.full_like(reward, 1e-3), reward)


def speed_scaled_center_line(env) -> torch.Tensor:
    """Center line bands multiplied by speed, with a bonus for the left side.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)
    speed = env.v_forward
    steering = env.actions[:, 0].abs() * 30.0

    reward = torch.full_like(distance_from_center, 1e-3)

    marker1 = distance_from_center <= (0.1 * track_width)
    marker2 = distance_from_center <= (0.15 * track_width)
    marker3 = distance_from_center <= (0.25 * track_width)
    marker4 = distance_from_center <= (0.5 * track_width)

    reward[marker4] = (0.1 * speed)[marker4]
    reward[marker3] = (0.3 * speed)[marker3]
    reward[marker2] = (0.8 * speed)[marker2]
    reward[marker1] = (1.0 * speed)[marker1]

    # env.lateral is signed against the left normal, so positive is left
    is_left_of_center = env.lateral > 0
    banded = marker4 & is_left_of_center
    reward[banded] += 0.1

    steering_too_much = steering > 15.0
    reward[steering_too_much] *= 0.8

    return torch.where(all_wheels_on_track, reward,
                       torch.full_like(reward, 1e-3))


def progress_tiers_with_speed(env) -> torch.Tensor:
    """Progress tiers scaled by speed-weighted center line bands.

    The upstream original multiplies by ``reward`` again inside its
    is_left_of_center branches; that self-reference is dropped as a typo.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    all_wheels_on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env * 100.0
    speed = env.v_forward
    steering = env.actions[:, 0].abs() * 30.0

    reward = torch.ones_like(distance_from_center)

    # lowest tier first
    for threshold, divisor in ((1.0, 10.0), (25.0, 8.0), (50.0, 4.0), (75.0, 2.0)):
        tier = progress >= threshold
        reward = torch.where(tier, (REWARD_MAX / divisor) * (progress / 100.0),
                             reward)

    marker1 = distance_from_center <= (0.1 * track_width)
    marker2 = distance_from_center <= (0.15 * track_width)
    marker3 = distance_from_center <= (0.25 * track_width)
    marker4 = distance_from_center <= (0.5 * track_width)

    band = torch.zeros_like(reward)
    band[marker4] = (0.1 * speed)[marker4]
    band[marker3] = (0.3 * speed)[marker3]
    band[marker2] = (0.8 * speed)[marker2]
    band[marker1] = (1.0 * speed)[marker1]

    is_left_of_center = env.lateral > 0
    band[marker4 & is_left_of_center] += 0.1

    reward = reward * band

    steering_too_much = steering > 15.0
    reward[steering_too_much] *= 0.8

    lap_done = progress >= 99.0
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)

    off_road = ~all_wheels_on_track | ~marker4
    reward = torch.where(off_road, torch.full_like(reward, REWARD_MIN), reward)

    return reward.clamp(REWARD_MIN, REWARD_MAX)


#: waypoint index -> which side the car should be on to earn the racing-line
#: bonus (True = left of center). Indices are the upstream author's own track.
_RACING_LINE = {8: False, 17: True, 33: False, 36: False, 40: True,
                51: True, 59: False, 64: True}


def center_line_with_waypoint_targets(env) -> torch.Tensor:
    """Center line bands plus a racing-line bonus at hand-picked waypoints.

    The waypoint indices come from the upstream author's own track and carry no
    meaning on another one.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    steering = env.actions[:, 0].abs() * 30.0

    reward = torch.full_like(distance_from_center, 1e-3)

    marker1 = distance_from_center <= (0.1 * track_width)
    marker2 = distance_from_center <= (0.25 * track_width)
    marker3 = distance_from_center <= (0.5 * track_width)

    reward[marker3] = 0.1
    reward[marker2] = 0.5
    reward[marker1] = 1.0

    is_left_of_center = env.lateral > 0
    for waypoint, wants_left in _RACING_LINE.items():
        on_target = (env.wp_idx == waypoint) & (is_left_of_center == wants_left)
        reward[on_target] = 5.0

    steering_too_much = steering > 30.0
    slow_speed = env.v_forward < 4.0

    reward[steering_too_much] *= 0.5
    reward[slow_speed] *= 0.5

    is_reversed = env.dir_sign < 0
    voided = is_reversed | ~marker3

    return torch.where(voided, torch.full_like(reward, 1e-3), reward)


def exponential_center_with_heading(env) -> torch.Tensor:
    """Exponential centering scaled by progress, penalising heading and throttle.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    distance_from_center = env.lateral.abs()
    on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env
    steering = env.actions[:, 0].abs()
    throttle = (env.actions[:, 1] + 1.0) / 2.0

    reward = torch.exp(-6.0 * distance_from_center) * progress

    heading_off = env.heading_err.abs() >= math.radians(10.0)
    steering_too_much = steering > 0.85
    throttle_too_high = throttle > (1.0 - 0.4 * steering)

    reward[heading_off] *= 0.25
    reward[steering_too_much] *= 0.75
    reward = reward * (0.5 + throttle)
    reward[throttle_too_high] *= 0.8

    lap_done = progress >= 0.99
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)
    reward = torch.where(on_track, reward, torch.full_like(reward, REWARD_MIN))

    return reward.clamp(REWARD_MIN, REWARD_MAX)


def progress_scaled_lane_keeping(env) -> torch.Tensor:
    """Progress-scaled reward tapering outside the center lane, heading aware.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    track_width = 2.0 * env.half_width
    distance_from_center = env.lateral.abs()
    on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env
    steering = env.actions[:, 0].abs()
    throttle = (env.actions[:, 1] + 1.0) / 2.0

    center_lane = track_width * 0.25
    heading = env.heading_err.abs()

    reward = REWARD_MAX * progress

    outside_lane = distance_from_center > center_lane
    reward = torch.where(outside_lane,
                         reward * (1.0 - distance_from_center / env.half_width),
                         reward)

    heading_off = heading >= math.radians(10.0)
    steering_too_much = ~heading_off & (steering > 0.85)
    aligned = ~heading_off & ~steering_too_much

    reward = torch.where(heading_off, reward * (1.0 - heading / 180.0), reward)
    reward = torch.where(steering_too_much, reward * (0.85 / steering.clamp(min=1e-6)),
                         reward)
    reward = torch.where(aligned, reward * (1.0 + (10.0 - heading / 10.0)), reward)

    steering_dominates = (steering > 0.5) & (steering > throttle)
    reward = torch.where(steering_dominates, reward * (1.0 - (steering - throttle)),
                         reward * (1.0 + throttle))

    lap_done = progress >= 0.99
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)
    reward = torch.where(on_track, reward, torch.full_like(reward, REWARD_MIN))

    return reward.clamp(REWARD_MIN, REWARD_MAX)


def exponential_center_first_model(env) -> torch.Tensor:
    """Gentle exponential centering with heading, steering and throttle penalties.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    distance_from_center = env.lateral.abs()
    on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env
    steering = env.actions[:, 0].abs()
    throttle = (env.actions[:, 1] + 1.0) / 2.0

    reward = torch.exp(-1.0 * distance_from_center) * progress

    heading_off = env.heading_err.abs() >= math.radians(5.0)
    steering_too_much = steering > 0.7
    slow_speed = throttle < 0.35
    throttle_too_high = throttle > (1.0 - 0.4 * steering)

    reward[heading_off] *= 0.5
    reward[steering_too_much] *= 0.75
    reward = torch.where(slow_speed, reward * (1.0 - 0.5 * throttle), reward)
    reward[throttle_too_high] *= 0.8

    lap_done = progress >= 0.99
    reward = torch.where(lap_done, torch.full_like(reward, 1e3), reward)
    reward = torch.where(on_track, reward, torch.full_like(reward, -1.0))

    return reward.clamp(1e-5, REWARD_MAX)


def exponential_center_second_model(env) -> torch.Tensor:
    """Sharper exponential centering, the tidied second iteration of the first.

    Args:
        env: The live DeepRacerEnv.

    Returns:
        The (N,) step reward.
    """
    margin = env.cfg["termination"]["wheel_margin"]
    distance_from_center = env.lateral.abs()
    on_track = ~is_off_track(env.lateral, env.half_width, margin)
    progress = env.progress_m / env.track.total_len_env
    steering = env.actions[:, 0].abs()
    throttle = (env.actions[:, 1] + 1.0) / 2.0

    reward = torch.exp(-6.0 * distance_from_center) * progress

    heading_off = env.heading_err.abs() >= math.radians(10.0)
    steering_too_much = steering > 0.85
    throttle_too_high = throttle > (1.0 - 0.4 * steering)

    reward[heading_off] *= 0.25
    reward[steering_too_much] *= 0.75
    reward = reward * (0.5 + throttle)
    reward[throttle_too_high] *= 0.8

    lap_done = progress >= 0.99
    reward = torch.where(lap_done, torch.full_like(reward, REWARD_MAX), reward)
    reward = torch.where(on_track, reward, torch.full_like(reward, REWARD_MIN))

    return reward.clamp(REWARD_MIN, REWARD_MAX)
