from typing import Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("PickAndPlaceColor-v1", max_episode_steps=1000)
class PickAndPlaceColorEnv(BaseEnv):
    """
    **Task Description:**
    - Three cubes (red, green, blue) and three trays of the matching colors are placed on the
      table. The goal is to pick up each cube one by one and place it inside the tray of the
      same color.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions randomized in the half of the workspace nearest the
      robot. The positions are sampled such that the cubes do not collide with each other
    - the three trays are randomly assigned to three jittered slots in the half of the
      workspace furthest from the robot, so their left/right ordering changes every episode
    - every sampled position is kept inside the robot's reachable workspace

    **Rewards:**
    - the sparse reward counts completed stages, six in all, worth one point each: pick red,
      place red, pick green, place green, pick blue, place blue. It ranges from 0 to 6 and
      steps up by 1 per stage. A colour's pick stage stays earned once its cube is in the
      tray, so a finished colour is always worth 2 points

    **Success Conditions:**
    - each cube lies inside its matching tray (within the tray's inner walls and resting on
      the tray floor)
    - all three cubes are static
    - none of the cubes are grasped by the robot (the robot must let go of every cube)

    **Cameras:**
    - the default robot is `panda_wristcam`, so observations carry both `base_camera` and
      the wrist-mounted `hand_camera`, each at `sensor_camera_resolution`. Pass
      ``robot_uids="panda"`` for a single-camera setup.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    agent: Union[Panda, Fetch]

    # colors of the three cube/tray pairs. Trays use a darkened version of the cube color so
    # the two are still visually distinguishable while keeping the same hue.
    COLORS = dict(
        red=([1.0, 0.0, 0.0, 1.0], [0.7, 0.06, 0.06, 1.0]),
        green=([0.0, 1.0, 0.0, 1.0], [0.05, 0.4, 0.05, 1.0]),
        blue=([0.0, 0.0, 1.0, 1.0], [0.05, 0.08, 0.45, 1.0]),
    )

    cube_half_size = 0.02
    # how far a cube must be off the table before it counts as picked, rather than merely
    # gripped while still resting on the surface. This has to stay below the height at
    # which a cube rests in a tray (2 * tray_wall_thickness = 1cm above the table) so that
    # a delivered cube still reads as picked -- see _cube_picked.
    pick_lift_height = 0.005
    # tray geometry (a shallow open-top box)
    tray_inner_half_size = 0.03  # half length of the tray's inner square (1 cm around the cube)
    tray_wall_thickness = 0.005
    tray_wall_height = 0.012  # height of the walls above the tray floor
    tray_outer_half_size = tray_inner_half_size + tray_wall_thickness

    # workspace bands. The panda base sits at x=-0.615, so smaller x is closer to the robot.
    cube_region = [[-0.24, -0.22], [-0.15, 0.22]]  # [[x_min, y_min], [x_max, y_max]]
    tray_slot_ys = [-0.2, 0.0, 0.2]
    tray_slot_y_jitter = 0.04
    tray_x_range = [-0.03, 0.05]

    # Resolution of base_camera, matching the ReCAST PickCube-v1 setup.
    sensor_camera_resolution = (480, 480)

    #: the stages an episode may start from, as stage numbers in the 6-stage reward.
    #: Stage 1 is the full task; stage 3 starts with red already delivered; stage 5 with
    #: red and green delivered. Index i pre-fills i cubes.
    START_STAGES = (1, 3, 5)

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        start_stage_probs=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Mix of starting stages, as weights over START_STAGES. None (the default) means
        # every episode is the full three-cube task. The stage is drawn per episode from
        # the episode RNG, so a given seed always yields the same stage and the mix is
        # reproducible. reset(options={"start_stage": k}) overrides it for one episode.
        if start_stage_probs is not None:
            probs = np.asarray(start_stage_probs, dtype=np.float64)
            if probs.shape != (len(self.START_STAGES),):
                raise ValueError(
                    f"start_stage_probs must have {len(self.START_STAGES)} entries, one "
                    f"per stage in {self.START_STAGES}, got {start_stage_probs}"
                )
            if (probs < 0).any() or probs.sum() <= 0:
                raise ValueError(
                    f"start_stage_probs must be non-negative and sum to more than 0, "
                    f"got {start_stage_probs}"
                )
            probs = probs / probs.sum()
        else:
            probs = None
        self.start_stage_probs = probs

        # The wrist camera is defined by the agent, not the task, so its resolution is
        # defaulted here to match base_camera. Explicit caller settings take precedence.
        if robot_uids == "panda_wristcam":
            sensor_configs = dict(kwargs.pop("sensor_configs", None) or {})
            width, height = self.sensor_camera_resolution
            sensor_configs.setdefault("hand_camera", dict(width=width, height=height))
            kwargs["sensor_configs"] = sensor_configs

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        # angled so the whole workspace (both cubes and trays) stays unoccluded by the arm
        pose = sapien_utils.look_at(eye=[0.3, 0.35, 0.4], target=[-0.12, 0, 0.03])
        width, height = self.sensor_camera_resolution
        return [CameraConfig("base_camera", pose, width, height, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.55, 0.55, 0.55], [-0.1, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _build_tray(self, color, name: str, initial_pose: sapien.Pose):
        """Builds a shallow open-top tray whose local origin sits at the center of its
        bottom face, so a pose with z=0 places it flat on the table."""
        builder = self.scene.create_actor_builder()
        outer = self.tray_outer_half_size
        floor_half = self.tray_wall_thickness
        wall_half_h = self.tray_wall_height / 2
        wall_half_t = self.tray_wall_thickness / 2
        # distance from the tray center to the center of a wall
        d = self.tray_inner_half_size + wall_half_t
        wall_z = 2 * floor_half + wall_half_h

        poses = [
            sapien.Pose([0, 0, floor_half]),  # floor
            sapien.Pose([-d, 0, wall_z]),
            sapien.Pose([d, 0, wall_z]),
            sapien.Pose([0, -d, wall_z]),
            sapien.Pose([0, d, wall_z]),
        ]
        half_sizes = [
            [outer, outer, floor_half],
            [wall_half_t, outer, wall_half_h],
            [wall_half_t, outer, wall_half_h],
            [outer, wall_half_t, wall_half_h],
            [outer, wall_half_t, wall_half_h],
        ]
        material = sapien.render.RenderMaterial(base_color=color)
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=material)
        builder.set_initial_pose(initial_pose)
        return builder.build_kinematic(name=name)

    def _cube_resting_pose_in_tray(self, tray, b: int):
        """Pose for a cube already delivered to its tray: on the tray floor, jittered
        inside the walls but comfortably within the placement tolerance."""
        jitter = self.tray_inner_half_size - self.cube_half_size - 0.003
        xyz = tray.pose.p.clone()
        xyz[:, :2] += torch.rand((b, 2), device=self.device) * 2 * jitter - jitter
        xyz[:, 2] = 2 * self.tray_wall_thickness + self.cube_half_size
        return xyz

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = []
        self.trays = []
        for i, (color_name, (cube_color, tray_color)) in enumerate(self.COLORS.items()):
            self.cubes.append(
                actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=cube_color,
                    name=f"cube_{color_name}",
                    initial_pose=sapien.Pose(p=[i - 1, -0.3, 0.2]),
                )
            )
            self.trays.append(
                self._build_tray(
                    tray_color,
                    name=f"tray_{color_name}",
                    initial_pose=sapien.Pose(p=[i - 1, 0.3, 0.2]),
                )
            )
        # convenience aliases
        self.cube_red, self.cube_green, self.cube_blue = self.cubes
        self.tray_red, self.tray_green, self.tray_blue = self.trays

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ---------------- trays: randomly permuted, jittered slots ----------------
            slot_ys = common.to_tensor(self.tray_slot_ys, device=self.device)
            # a random permutation of the 3 slots per parallel env
            perm = torch.argsort(torch.rand((b, len(self.tray_slot_ys))), dim=1)
            tray_y = slot_ys[perm] + (
                torch.rand((b, 3)) * 2 * self.tray_slot_y_jitter
                - self.tray_slot_y_jitter
            )
            x_lo, x_hi = self.tray_x_range
            tray_x = torch.rand((b, 3)) * (x_hi - x_lo) + x_lo

            for i, tray in enumerate(self.trays):
                pos = torch.zeros((b, 3))
                pos[:, 0] = tray_x[:, i]
                pos[:, 1] = tray_y[:, i]
                pos[:, 2] = 0.0  # tray origin is its bottom face
                tray.set_pose(Pose.create_from_pq(p=pos, q=[1, 0, 0, 0]))

            # ------------- how many cubes are already delivered at reset -------------
            # n_done cubes start in their trays, so the episode begins part-way through
            # the task: 0 -> stage 1 (the full task), 1 -> stage 3, 2 -> stage 5.
            start_stage = (options or {}).get("start_stage")
            if start_stage is not None:
                if start_stage not in self.START_STAGES:
                    raise ValueError(
                        f"start_stage must be one of {self.START_STAGES}, got "
                        f"{start_stage}"
                    )
                n_done = torch.full(
                    (b,), self.START_STAGES.index(start_stage), dtype=torch.long
                )
            elif self.start_stage_probs is not None:
                cdf = common.to_tensor(
                    np.cumsum(self.start_stage_probs), device=self.device
                )
                n_done = torch.searchsorted(cdf, torch.rand((b, 1))).squeeze(1)
                n_done = n_done.clamp(max=len(self.START_STAGES) - 1).to(torch.long)
            else:
                n_done = torch.zeros((b,), dtype=torch.long)

            # ---------------- cubes: collision free uniform sampling -----------------
            sampler = randomization.UniformPlacementSampler(
                bounds=self.cube_region, batch_size=b, device=self.device
            )
            # keep the cubes far enough apart that the gripper fingers always fit
            radius = 0.035
            for i, (cube, tray) in enumerate(zip(self.cubes, self.trays)):
                xyz = torch.zeros((b, 3))
                # sampled unconditionally so the table layout for a given seed does not
                # shift with the starting stage
                xyz[:, :2] = sampler.sample(radius, 100, verbose=False)
                xyz[:, 2] = self.cube_half_size
                # the first n_done cubes start delivered instead
                delivered = (n_done > i).unsqueeze(1)
                xyz = torch.where(
                    delivered, self._cube_resting_pose_in_tray(tray, b), xyz
                )
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _cube_picked(self, cube, is_grasped, is_placed):
        """A cube counts as picked once it is clear of the table: held and lifted, in
        flight after being released over a tray, or already resting in one (a tray floor
        sits above table height).

        This keys on height rather than on the grasp so the stage stays earned through the
        release itself. For a frame or two after the gripper opens the cube is falling --
        no longer held, not yet at rest in the tray -- and a grasp-based test would read
        false there, dropping the reward by one every time the robot lets go and turning
        the staircase into a sawtooth. The pick stage of a colour therefore stays earned
        once its cube is delivered, so a finished colour is always worth its full 2
        points."""
        clear_of_table = (
            cube.pose.p[..., 2] > self.cube_half_size + self.pick_lift_height
        )
        return (clear_of_table | is_grasped | is_placed).bool()

    def _cube_in_tray(self, cube, tray):
        """A cube counts as placed when it sits on the tray floor, within the tray walls,
        is at rest, and is no longer held by the robot."""
        offset = cube.pose.p - tray.pose.p
        # the cube center must stay inside the tray's inner square
        margin = self.tray_inner_half_size - self.cube_half_size
        xy_flag = torch.max(torch.abs(offset[..., :2]), dim=1).values <= margin
        # resting on the tray floor
        resting_z = 2 * self.tray_wall_thickness + self.cube_half_size
        z_flag = torch.abs(offset[..., 2] - resting_z) <= 0.01
        is_static = cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_grasped = self.agent.is_grasping(cube)
        return (xy_flag & z_flag & is_static & ~is_grasped).bool(), is_grasped

    def evaluate(self):
        info = dict()
        placed, picked = [], []
        for (color_name, _), cube, tray in zip(
            self.COLORS.items(), self.cubes, self.trays
        ):
            is_placed, is_grasped = self._cube_in_tray(cube, tray)
            is_picked = self._cube_picked(cube, is_grasped, is_placed)
            info[f"is_{color_name}_picked"] = is_picked
            info[f"is_{color_name}_placed"] = is_placed
            info[f"is_{color_name}_grasped"] = is_grasped
            placed.append(is_placed)
            picked.append(is_picked)
        info["num_placed"] = torch.stack(placed, dim=1).sum(dim=1)
        # the six stages, in order: pick red, place red, pick green, place green,
        # pick blue, place blue
        info["num_stages"] = torch.stack(picked + placed, dim=1).sum(dim=1)
        info["success"] = placed[0] & placed[1] & placed[2]
        return info

    def compute_sparse_reward(self, obs, action: torch.Tensor, info: dict):
        """The task is scored as six stages worth one point each -- pick red, place red,
        pick green, place green, pick blue, place blue -- so the reward runs 0 to 6 and
        steps up once per stage completed.

        Picking a colour stays earned once that cube is in its tray (see _cube_picked),
        so a completed colour is always worth its full 2 points and the reward is a
        staircase. It is still a function of the current state rather than a latch on
        history: knocking a placed cube back out of its tray gives the points back, which
        is what makes the reward Markovian and safe to learn from.

        Overrides the default sparse reward, which would only pay out once all three
        cubes are placed."""
        return info["num_stages"].to(torch.float)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        # tray positions describe the goal, so they are always exposed
        for color_name, tray in zip(self.COLORS.keys(), self.trays):
            obs[f"tray_{color_name}_pos"] = tray.pose.p
        if "state" in self.obs_mode:
            for color_name, cube in zip(self.COLORS.keys(), self.cubes):
                obs[f"cube_{color_name}_pose"] = cube.pose.raw_pose
                obs[f"tcp_to_cube_{color_name}_pos"] = (
                    cube.pose.p - self.agent.tcp.pose.p
                )
            for color_name, cube, tray in zip(
                self.COLORS.keys(), self.cubes, self.trays
            ):
                obs[f"cube_{color_name}_to_tray_pos"] = tray.pose.p - cube.pose.p
        return obs
