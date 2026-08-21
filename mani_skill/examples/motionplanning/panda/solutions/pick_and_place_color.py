import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PickAndPlaceColorEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.utils.structs.pose import to_sapien_pose

FINGER_LENGTH = 0.025
# height the gripper travels at between the cubes and the trays
CARRY_HEIGHT = 0.16

# mplib does NOT read the velocity/acceleration limits out of the URDF -- it defaults
# every joint to 1.0 rad/s and 1.0 rad/s^2, which the solver then scales by these
# fractions. The Panda is rated 2.175 rad/s (joints 1-4) / 2.61 rad/s (joints 5-7), so
# the stock 0.9 runs the arm at roughly 40% of its capability and the acceleration cap
# is what really stretches the trajectories out. These stay under the URDF ceiling.
JOINT_VEL_LIMIT = 1.8   # rad/s
JOINT_ACC_LIMIT = 4.0   # rad/s^2

# Frames spent holding still are frames a policy learns nothing from, so the dwell times
# are only as long as the physics needs.
GRIPPER_CLOSE_STEPS = 6  # the fingers must actually clamp before the lift
GRIPPER_OPEN_STEPS = 3
RELEASE_SETTLE_STEPS = 2  # kill residual arm motion before letting go, so the cube
                          # drops straight down instead of skidding off-centre. With the
                          # arm holding still for this plus GRIPPER_OPEN_STEPS, this is
                          # the whole pause between placing a cube and leaving for the
                          # next one, so it is kept as short as the placement tolerates.
SETTLE_STEPS = 6         # let the cube drop the last 1.5 cm onto the tray floor
FINAL_SETTLE_STEPS = 12  # success needs every cube static on the last frame


def _chain(planner, poses, refine_steps=0):
    """Plan a sequence of poses and execute them as ONE time-parameterised trajectory.

    Every move_to_pose_* call plans a trajectory that both starts and ends at zero
    velocity, so running them back to back brings the arm to a full stop at each
    intermediate waypoint -- most visibly at the pre-grasp pose. Planning the segments up
    front, concatenating their joint paths and re-timing the whole path with TOPP keeps
    the same geometry (the path is densely sampled, so the spline passes through it) but
    decelerates only once, at the very end. Corners are rounded in time rather than cut in
    space: TOPP slows the arm through them to respect the acceleration limit.

    Returns None if any segment fails to plan, so the caller can fall back to stepping
    through the waypoints one at a time.
    """
    mp = planner.planner
    dof = len(mp.joint_vel_limits)
    qpos = planner.robot.get_qpos().cpu().numpy()[0]
    segments = []
    for pose in poses:
        pose = to_sapien_pose(pose)
        result = mp.plan_screw(
            np.concatenate([pose.p, pose.q]),
            qpos,
            time_step=planner.base_env.control_timestep,
            use_point_cloud=planner.use_point_cloud,
        )
        if result["status"] != "Success":
            return None
        segments.append(result["position"])
        # the next segment starts where this one ended
        qpos = np.concatenate([result["position"][-1], qpos[dof:]])

    path = np.vstack(segments)
    # consecutive segments share an endpoint; a duplicate point is a zero-length step,
    # which TOPP reads as a singular path derivative and crawls through
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path = path[np.concatenate([[True], step > 1e-6])]
    if len(path) < 3:  # TOPP needs a few samples to fit a spline
        return None

    _, qs, qds, _, _ = mp.TOPP(path, planner.base_env.control_timestep)
    return planner.follow_path(
        {"position": qs, "velocity": qds}, refine_steps=refine_steps
    )


def _transit(planner, pose):
    """Long horizontal move. Screw planning is a straight line so it can clip the other
    cubes/trays. Fall back on RRTConnect when it fails."""
    res = planner.move_to_pose_with_screw(pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(pose)
    return res


def _plan_grasp_pose(env, planner, cube):
    """Top down grasp pose for the cube. The cubes are randomly yawed, so a few yaw
    offsets are tried until the planner finds one it can actually reach."""
    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        candidate = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
        if planner.move_to_pose_with_screw(candidate, dry_run=True) != -1:
            return candidate
    print("Fail to find a valid grasp pose")
    return grasp_pose


def solve(env: PickAndPlaceColorEnv, seed=None, debug=False, vis=False, options=None):
    # options is forwarded to reset, so a caller can pin the starting stage with
    # options={"start_stage": 3}. Left as None the env decides, which is either the full
    # task or a draw from its start_stage_probs.
    env.reset(seed=seed, options=options)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=JOINT_VEL_LIMIT,
        joint_acc_limits=JOINT_ACC_LIMIT,
    )
    env = env.unwrapped

    # releasing here drops the cube a short way onto the tray floor, while keeping the
    # fingers clear of the tray walls
    release_z = 2 * env.tray_wall_thickness + env.cube_half_size + 0.015

    # Grasp poses for all three cubes are planned up front, while the arm is still at
    # its rest pose. The cubes do not move until they are picked, so the poses are valid
    # for the whole episode, and having the next cube's grasp in hand before the retreat
    # starts is what lets the retreat and the next approach be flown as one motion.
    # Episodes can start part-way through the task, with the first cubes already sitting
    # in their trays (see the env's start_stage_probs). Those are left alone.
    info = env.evaluate()
    todo = [
        i
        for i, color_name in enumerate(env.COLORS)
        if not bool(info[f"is_{color_name}_placed"][0])
    ]
    if not todo:  # already solved at reset; nothing to do
        planner.close()
        return env.step(env.agent.robot.get_qpos()[0, :-1].cpu().numpy())

    grasp_poses = {i: _plan_grasp_pose(env, planner, env.cubes[i]) for i in todo}
    above_cubes = {
        i: sapien.Pose([g.p[0], g.p[1], CARRY_HEIGHT], g.q)
        for i, g in grasp_poses.items()
    }

    def approach(idx):
        """Travel to cube idx and descend onto it as one continuous motion. The old
        intermediate pre-grasp pose was redundant -- it sat on the same vertical line as
        the other two, so it only added a dead stop half way down."""
        if _chain(planner, [above_cubes[idx], grasp_poses[idx]]) is None:
            # screw planning failed somewhere in the chain; fall back to separate moves,
            # which can route around obstacles with RRTConnect
            _transit(planner, above_cubes[idx])
            _transit(planner, grasp_poses[idx] * sapien.Pose([0, 0, -0.05]))
            planner.move_to_pose_with_screw(grasp_poses[idx])

    res = None
    approach(todo[0])
    # env.cubes and env.trays are both ordered red, green, blue, so index i pairs each
    # cube with the tray of its own color
    for n, i in enumerate(todo):
        tray = env.trays[i]
        is_last = n == len(todo) - 1
        planner.close_gripper(t=GRIPPER_CLOSE_STEPS)

        # Lift clear of the other cubes, carry to the matching tray and lower in, all as
        # one motion. The waypoints still force the arm up to CARRY_HEIGHT before it
        # travels sideways; it just no longer stops at the corners.
        grasp_pose = grasp_poses[i]
        tray_x, tray_y = tray.pose.p.cpu().numpy()[0][:2]
        above_tray = sapien.Pose([tray_x, tray_y, CARRY_HEIGHT], grasp_pose.q)
        lower_into_tray = sapien.Pose([tray_x, tray_y, release_z], grasp_pose.q)
        if (
            _chain(
                planner,
                [above_cubes[i], above_tray, lower_into_tray],
                refine_steps=RELEASE_SETTLE_STEPS,
            )
            is None
        ):
            planner.move_to_pose_with_screw(above_cubes[i])
            _transit(planner, above_tray)
            planner.move_to_pose_with_screw(
                lower_into_tray, refine_steps=RELEASE_SETTLE_STEPS
            )
        res = planner.open_gripper(t=GRIPPER_OPEN_STEPS)

        # Retreat out of the tray. For every cube but the last that retreat runs straight
        # on into the next cube's approach as a single trajectory, so the arm does not
        # stop once the cube is placed. Only the last cube needs a hold at the end, since
        # the success check reads the final frame and wants every cube at rest.
        if is_last:
            res = planner.move_to_pose_with_screw(
                above_tray, refine_steps=FINAL_SETTLE_STEPS
            )
        else:
            nxt = todo[n + 1]
            if (
                _chain(planner, [above_tray, above_cubes[nxt], grasp_poses[nxt]])
                is None
            ):
                res = planner.move_to_pose_with_screw(
                    above_tray, refine_steps=SETTLE_STEPS
                )
                approach(nxt)

    planner.close()
    return res
