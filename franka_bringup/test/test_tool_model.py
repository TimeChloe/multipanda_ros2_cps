# Copyright (c) 2026 Yue
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from franka_bringup.tool_model import (
    ToolModelError,
    generate_tool_artifacts,
    load_tool_description,
)


WORKSPACE = Path(__file__).resolve().parents[2]
BRINGUP = WORKSPACE / 'franka_bringup'
DESCRIPTION = WORKSPACE / 'franka_description'


def test_metal_ball_derived_parameters_match_existing_controller_geometry():
    tool = load_tool_description(str(BRINGUP / 'config/tools/metal_ball.yaml'))
    assert tool.mass == pytest.approx(0.2)
    assert tool.com_parent == pytest.approx((0.0, 0.0, 0.03))
    assert tool.tcp['xyz'] == pytest.approx((0.0, 0.0, 0.06))
    assert tool.collision_center_from_tcp == pytest.approx((0.0, 0.0, -0.03))
    assert tool.safety_radius == pytest.approx(0.03)


def test_invalid_inertia_is_rejected(tmp_path):
    source = yaml.safe_load(
        (BRINGUP / 'config/tools/metal_ball.yaml').read_text(encoding='utf-8')
    )
    source['inertial']['inertia']['izz'] = -1.0
    invalid_path = tmp_path / 'invalid.yaml'
    invalid_path.write_text(yaml.safe_dump(source), encoding='utf-8')
    with pytest.raises(ToolModelError, match='positive definite'):
        load_tool_description(str(invalid_path))


def test_one_yaml_generates_urdf_mjcf_and_controller_parameters(tmp_path):
    artifacts = generate_tool_artifacts(
        tool_config_path=str(BRINGUP / 'config/tools/metal_ball.yaml'),
        arm_id='panda',
        robot_xacro_path=str(
            DESCRIPTION / 'robots/sim/panda_arm_sim.urdf.xacro'
        ),
        xacro_mappings={
            'arm_id': 'panda',
            'hand': 'false',
            'metal_ball': 'false',
            'initial_positions': '0.0 -0.578 0.0 -1.753 0.0 1.175 0.785',
        },
        monitor_base_urdf_path=str(WORKSPACE / 'model_urdf/panda_ng.urdf'),
        output_root=str(tmp_path),
        mujoco_base_path=str(DESCRIPTION / 'mujoco/franka/panda_ng.xml'),
        mujoco_table_path=str(DESCRIPTION / 'mujoco/franka/table.xml'),
        include_table=True,
        controller_config_path=str(
            BRINGUP / 'config/sim/single_sim_controllers.yaml'
        ),
    )

    urdf_root = ET.parse(artifacts.monitor_urdf_path).getroot()
    tool_link = urdf_root.find("link[@name='panda_metal_ball_link']")
    assert tool_link is not None
    assert float(tool_link.find('inertial/mass').attrib['value']) == pytest.approx(0.2)

    scene_root = ET.parse(artifacts.mujoco_scene_path).getroot()
    robot_include = Path(scene_root.find('include').attrib['file'])
    robot_root = ET.parse(robot_include).getroot()
    assert robot_root.find(".//body[@name='panda_metal_ball']") is not None
    assert robot_root.find("sensor/touch[@name='panda_metal_ball_touch']") is not None

    controller_config = yaml.safe_load(
        Path(artifacts.controller_config_path).read_text(encoding='utf-8')
    )
    reachable = controller_config[
        'reachable_cartesian_impedance_controller'
    ]['ros__parameters']
    assert reachable['monitor_urdf_model_path'] == artifacts.monitor_urdf_path
    assert reachable['tcp_offset'] == pytest.approx([0.0, 0.0, 0.06])
    assert reachable['ee_collision_center_offset'] == pytest.approx(
        [0.0, 0.0, -0.03]
    )
    assert reachable['ee_collision_radius'] == pytest.approx(0.03)


def test_relative_mesh_is_shared_by_urdf_and_mujoco(tmp_path):
    mesh_path = tmp_path / 'meshes' / 'tool.stl'
    mesh_path.parent.mkdir()
    mesh_path.write_text(
        'solid tool\n'
        'facet normal 0 0 1\n'
        'outer loop\n'
        'vertex 0 0 0\n'
        'vertex 0.01 0 0\n'
        'vertex 0 0.01 0\n'
        'endloop\n'
        'endfacet\n'
        'endsolid tool\n',
        encoding='utf-8',
    )
    source = yaml.safe_load(
        (BRINGUP / 'config/tools/metal_ball.yaml').read_text(encoding='utf-8')
    )
    source['name'] = 'mesh_tool'
    mesh_geometry = {
        'type': 'mesh',
        'mesh': 'meshes/tool.stl',
        'scale': [1.0, 2.0, 3.0],
    }
    source['visual'] = {**mesh_geometry, 'rgba': [0.2, 0.4, 0.8, 1.0]}
    source['collision'] = mesh_geometry
    config_path = tmp_path / 'tool.yaml'
    config_path.write_text(yaml.safe_dump(source), encoding='utf-8')

    artifacts = generate_tool_artifacts(
        tool_config_path=str(config_path),
        arm_id='panda',
        robot_xacro_path=str(
            DESCRIPTION / 'robots/sim/panda_arm_sim.urdf.xacro'
        ),
        xacro_mappings={
            'arm_id': 'panda',
            'hand': 'false',
            'metal_ball': 'false',
            'initial_positions': '0.0 -0.578 0.0 -1.753 0.0 1.175 0.785',
        },
        monitor_base_urdf_path=str(WORKSPACE / 'model_urdf/panda_ng.urdf'),
        output_root=str(tmp_path / 'generated'),
        mujoco_base_path=str(DESCRIPTION / 'mujoco/franka/panda_ng.xml'),
        mujoco_table_path=str(DESCRIPTION / 'mujoco/franka/table.xml'),
    )

    urdf_root = ET.parse(artifacts.monitor_urdf_path).getroot()
    urdf_mesh = urdf_root.find(
        "link[@name='panda_mesh_tool_link']/visual/geometry/mesh"
    )
    assert urdf_mesh.attrib['filename'] == f'file://{mesh_path}'
    assert urdf_mesh.attrib['scale'] == '1 2 3'

    scene_root = ET.parse(artifacts.mujoco_scene_path).getroot()
    robot_root = ET.parse(scene_root.find('include').attrib['file']).getroot()
    mujoco_mesh = robot_root.find(
        "asset/mesh[@name='panda_mesh_tool_visual_asset']"
    )
    assert Path(mujoco_mesh.attrib['file']) == mesh_path
    assert mujoco_mesh.attrib['scale'] == '1 2 3'
