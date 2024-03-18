"""Script to visualize the cone constraints on the Reachbot cable directions"""

import time

import pybullet
import numpy as np
import numpy.typing as npt

from reachbot_manipulation.core.reachbot import Reachbot
from reachbot_manipulation.utils.math_utils import normalize
from reachbot_manipulation.utils.grasp_utils import grasp_normal_rmat
from reachbot_manipulation.utils.rotations import rmat_to_quat
from reachbot_manipulation.utils.bullet_utils import initialize_pybullet


def view_cable_cones(
    robot: Reachbot,
    cone_angle: float = np.pi / 6,
    rgba: npt.ArrayLike = (1, 0, 0, 0.5),
    scale: float = 1,
) -> list[int]:
    """Visualize the cone limits on the Reachbot cable directions

    Args:
        robot (Reachbot): Reachbot
        cone_angle (float, optional): Interior angle of the cones. Must be pi/6, pi/4, pi/3, 3*pi/4, or pi.
            Defaults to pi/6.
        rgba (npt.ArrayLike, optional): Color description. Defaults to (1, 0, 0, 0.5).
        scale (float, optional): Scaling factor on the size of the cones. Defaults to 1.

    Returns:
        list[int]: Pybullet IDs of the cones
    """
    # Round to nearest 15 degree increment and check if we have a matching mesh
    deg = np.rad2deg(cone_angle)
    deg_rounded = 15 * round(deg / 15)
    if abs(deg_rounded - deg) > 1 or deg_rounded not in {30, 45, 60, 75, 90}:
        raise ValueError(
            "Unsupported cone angle: Must be pi/6, pi/4, pi/3, 3*pi/4, or pi"
        )
    cone_positions = robot.cable_positions
    cone_normals = normalize(robot.cable_positions - robot.position)
    rmats = [grasp_normal_rmat(n) for n in cone_normals]
    orns = [rmat_to_quat(r) for r in rmats]
    mesh_file = f"reachbot_manipulation/assets/meshes/cones/{deg_rounded}_deg.obj"
    cone_ids = []
    # Create each cone body in simulation
    for i in range(robot.NUM_CABLES):
        visual_id = robot.client.createVisualShape(
            shapeType=robot.client.GEOM_MESH,
            rgbaColor=rgba,
            fileName=mesh_file,
            meshScale=[scale] * 3,
            visualFramePosition=cone_positions[i],
            visualFrameOrientation=orns[i],
        )
        cone_ids.append(
            robot.client.createMultiBody(
                baseMass=0,  # Fixed position
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_id,
            )
        )
    return cone_ids


def _view_all_cones():
    client = initialize_pybullet()
    client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    angles = [np.deg2rad(a) for a in (30, 45, 60, 75, 90)]
    for i, angle in enumerate(angles):
        robot = Reachbot(pose=(2 * i, 0, 0, 0, 0, 0, 1))
        view_cable_cones(robot, cone_angle=angle, scale=0.5)
    input("Press Enter to loop the simulation")
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


def _view_one_at_a_time():
    client = initialize_pybullet()
    client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    client.resetDebugVisualizerCamera(8.20, -61.20, 3.40, (5.64, 2.91, 0.48))
    angles = [np.deg2rad(a) for a in (30, 45, 60, 75, 90)]
    robot = Reachbot()
    for i, angle in enumerate(angles):
        print("Viewing cone angle: ", angle)
        ids = view_cable_cones(robot, cone_angle=angle, scale=0.5)
        input("Press Enter to switch to the next cone")
        for cone_id in ids:
            client.removeBody(cone_id)
    input("Press Enter to loop the simulation")
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _view_one_at_a_time()
