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

"""Generate consistent URDF, MuJoCo and controller artifacts from one tool YAML."""

from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ToolModelError(ValueError):
    """Raised when a tool description cannot be used safely."""


def _finite_vector(value: Any, size: int, field: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ToolModelError(f"{field} must contain exactly {size} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ToolModelError(f"{field} must contain only finite numbers")
    return result


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise ToolModelError(f"{field} must be {qualifier}")
    return result


def _pose(data: Dict[str, Any], field: str) -> Dict[str, Tuple[float, ...]]:
    return {
        "xyz": _finite_vector(data.get("xyz", [0.0, 0.0, 0.0]), 3, f"{field}.xyz"),
        "rpy": _finite_vector(data.get("rpy", [0.0, 0.0, 0.0]), 3, f"{field}.rpy"),
    }


def _rotation_from_rpy(rpy: Sequence[float]) -> List[List[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _quaternion_wxyz_from_rpy(rpy: Sequence[float]) -> Tuple[float, ...]:
    roll, pitch, yaw = (0.5 * item for item in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, ...]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def _add(left: Sequence[float], right: Sequence[float]) -> Tuple[float, ...]:
    return tuple(left[index] + right[index] for index in range(3))


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def _mat_mul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> List[List[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _inertia_matrix(inertia: Dict[str, float]) -> List[List[float]]:
    return [
        [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
        [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
        [inertia["ixz"], inertia["iyz"], inertia["izz"]],
    ]


def _validate_inertia(inertia: Dict[str, float]) -> None:
    ixx, iyy, izz = inertia["ixx"], inertia["iyy"], inertia["izz"]
    ixy, ixz, iyz = inertia["ixy"], inertia["ixz"], inertia["iyz"]
    minor_2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    if ixx <= 0.0 or minor_2 <= 0.0 or determinant <= 0.0:
        raise ToolModelError("inertial.inertia must be symmetric positive definite")
    if ixx > iyy + izz or iyy > ixx + izz or izz > ixx + iyy:
        raise ToolModelError("inertial.inertia violates the rigid-body triangle inequalities")


def _geometry(
    value: Dict[str, Any], field: str, config_directory: Path
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolModelError(f"{field} must be a mapping")
    result: Dict[str, Any] = {"type": str(value.get("type", "")).lower()}
    result.update(_pose(value.get("origin", {}), f"{field}.origin"))
    if result["type"] == "sphere":
        result["radius"] = _number(value.get("radius"), f"{field}.radius", positive=True)
    elif result["type"] == "box":
        size = _finite_vector(value.get("size"), 3, f"{field}.size")
        if any(item <= 0.0 for item in size):
            raise ToolModelError(f"{field}.size values must be positive")
        result["size"] = size
    elif result["type"] == "cylinder":
        result["radius"] = _number(value.get("radius"), f"{field}.radius", positive=True)
        result["length"] = _number(value.get("length"), f"{field}.length", positive=True)
    elif result["type"] == "mesh":
        mesh_value = value.get("mesh")
        if not isinstance(mesh_value, str) or not mesh_value.strip():
            raise ToolModelError(f"{field}.mesh must be a non-empty path")
        mesh_path = Path(mesh_value).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = (config_directory / mesh_path).resolve()
        if not mesh_path.is_file():
            raise ToolModelError(f"{field}.mesh does not exist: {mesh_path}")
        result["mesh"] = str(mesh_path)
        scale = _finite_vector(value.get("scale", [1.0, 1.0, 1.0]), 3, f"{field}.scale")
        if any(item <= 0.0 for item in scale):
            raise ToolModelError(f"{field}.scale values must be positive")
        result["scale"] = scale
    else:
        raise ToolModelError(f"{field}.type must be sphere, box, cylinder, or mesh")
    return result


@dataclass(frozen=True)
class ToolDescription:
    name: str
    parent_frame: str
    mount: Dict[str, Tuple[float, ...]]
    tcp: Dict[str, Tuple[float, ...]]
    mass: float
    com: Tuple[float, ...]
    inertia: Dict[str, float]
    visual: Dict[str, Any]
    collision: Dict[str, Any]
    visual_rgba: Tuple[float, ...]
    safety_center_tool: Tuple[float, ...]
    safety_radius: float
    gravity_compensation: bool
    friction: Tuple[float, ...]
    touch_sensor: bool

    @property
    def mount_rotation(self) -> List[List[float]]:
        return _rotation_from_rpy(self.mount["rpy"])

    @property
    def safety_center_parent(self) -> Tuple[float, ...]:
        return _add(self.mount["xyz"], _mat_vec(self.mount_rotation, self.safety_center_tool))

    @property
    def collision_center_from_tcp(self) -> Tuple[float, ...]:
        center = self.safety_center_parent
        return tuple(center[index] - self.tcp["xyz"][index] for index in range(3))

    @property
    def com_parent(self) -> Tuple[float, ...]:
        return _add(self.mount["xyz"], _mat_vec(self.mount_rotation, self.com))

    @property
    def inertia_parent(self) -> List[List[float]]:
        rotation = self.mount_rotation
        return _mat_mul(_mat_mul(rotation, _inertia_matrix(self.inertia)), _transpose(rotation))

    @property
    def inertia_column_major(self) -> List[float]:
        matrix = self.inertia_parent
        return [matrix[row][col] for col in range(3) for row in range(3)]


@dataclass(frozen=True)
class ToolArtifacts:
    robot_description: str
    monitor_urdf_path: str
    mujoco_scene_path: Optional[str]
    controller_config_path: Optional[str]
    tool: Optional[ToolDescription]


def load_tool_description(path: str) -> ToolDescription:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ToolModelError(f"tool configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ToolModelError("tool configuration root must be a mapping")
    if int(data.get("schema_version", 1)) != 1:
        raise ToolModelError("only tool schema_version 1 is supported")

    name = str(data.get("name", ""))
    if not _NAME_PATTERN.fullmatch(name):
        raise ToolModelError(
            "name must start with a letter and contain only letters, digits, '_'"
        )
    parent_frame = str(data.get("parent_frame", "panda_link8"))
    if parent_frame != "panda_link8":
        raise ToolModelError("schema version 1 currently supports parent_frame panda_link8 only")

    mount = _pose(data.get("mount", {}), "mount")
    tcp = _pose(data.get("tcp", {}), "tcp")
    if any(abs(value) > 1.0e-12 for value in tcp["rpy"]):
        raise ToolModelError(
            "schema version 1 requires tcp.rpy=[0,0,0]; the controller currently uses a "
            "translation-only TCP offset"
        )

    inertial = data.get("inertial")
    if not isinstance(inertial, dict):
        raise ToolModelError("inertial is required and must be a mapping")
    mass = _number(inertial.get("mass"), "inertial.mass", positive=True)
    com = _finite_vector(inertial.get("com", [0.0, 0.0, 0.0]), 3, "inertial.com")
    inertia_value = inertial.get("inertia")
    if not isinstance(inertia_value, dict):
        raise ToolModelError("inertial.inertia is required and must be a mapping")
    inertia = {
        key: _number(inertia_value.get(key), f"inertial.inertia.{key}")
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    }
    _validate_inertia(inertia)

    config_directory = config_path.parent
    visual = _geometry(data.get("visual", {}), "visual", config_directory)
    collision = _geometry(data.get("collision", {}), "collision", config_directory)
    rgba = _finite_vector(
        data.get("visual", {}).get("rgba", [0.72, 0.74, 0.76, 1.0]),
        4,
        "visual.rgba",
    )
    if any(value < 0.0 or value > 1.0 for value in rgba):
        raise ToolModelError("visual.rgba values must be in [0, 1]")

    safety = data.get("safety")
    if not isinstance(safety, dict) or not isinstance(safety.get("bounding_sphere"), dict):
        raise ToolModelError("safety.bounding_sphere is required")
    safety_sphere = safety["bounding_sphere"]
    safety_center = _finite_vector(
        safety_sphere.get("center", [0.0, 0.0, 0.0]), 3, "safety.bounding_sphere.center"
    )
    safety_radius = _number(
        safety_sphere.get("radius"), "safety.bounding_sphere.radius", positive=True
    )

    mujoco = data.get("mujoco", {})
    if not isinstance(mujoco, dict):
        raise ToolModelError("mujoco must be a mapping")
    friction = _finite_vector(mujoco.get("friction", [1.0, 0.005, 0.0001]), 3, "mujoco.friction")
    if any(value < 0.0 for value in friction):
        raise ToolModelError("mujoco.friction values must be non-negative")

    return ToolDescription(
        name=name,
        parent_frame=parent_frame,
        mount=mount,
        tcp=tcp,
        mass=mass,
        com=com,
        inertia=inertia,
        visual=visual,
        collision=collision,
        visual_rgba=rgba,
        safety_center_tool=safety_center,
        safety_radius=safety_radius,
        gravity_compensation=bool(mujoco.get("gravity_compensation", True)),
        friction=friction,
        touch_sensor=bool(mujoco.get("touch_sensor", False)),
    )


def _append_urdf_geometry(parent: ET.Element, geometry: Dict[str, Any]) -> None:
    geometry_element = ET.SubElement(parent, "geometry")
    kind = geometry["type"]
    if kind == "sphere":
        ET.SubElement(geometry_element, "sphere", {"radius": _fmt([geometry["radius"]])})
    elif kind == "box":
        ET.SubElement(geometry_element, "box", {"size": _fmt(geometry["size"])})
    elif kind == "cylinder":
        ET.SubElement(
            geometry_element,
            "cylinder",
            {"radius": _fmt([geometry["radius"]]), "length": _fmt([geometry["length"]])},
        )
    else:
        ET.SubElement(
            geometry_element,
            "mesh",
            {"filename": f"file://{geometry['mesh']}", "scale": _fmt(geometry["scale"])},
        )


def _append_tool_to_urdf(root: ET.Element, tool: ToolDescription, arm_id: str) -> None:
    parent_frame = tool.parent_frame.replace("panda_", f"{arm_id}_", 1)
    link_name = f"{arm_id}_{tool.name}_link"
    joint_name = f"{arm_id}_{tool.name}_joint"
    link = ET.Element("link", {"name": link_name})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt(tool.com), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": _fmt([tool.mass])})
    ET.SubElement(
        inertial,
        "inertia",
        {key: _fmt([value]) for key, value in tool.inertia.items()},
    )

    visual = ET.SubElement(link, "visual")
    ET.SubElement(
        visual,
        "origin",
        {"xyz": _fmt(tool.visual["xyz"]), "rpy": _fmt(tool.visual["rpy"])},
    )
    _append_urdf_geometry(visual, tool.visual)
    material = ET.SubElement(visual, "material", {"name": f"{arm_id}_{tool.name}_material"})
    ET.SubElement(material, "color", {"rgba": _fmt(tool.visual_rgba)})

    collision = ET.SubElement(link, "collision")
    ET.SubElement(
        collision,
        "origin",
        {"xyz": _fmt(tool.collision["xyz"]), "rpy": _fmt(tool.collision["rpy"])},
    )
    _append_urdf_geometry(collision, tool.collision)

    joint = ET.Element("joint", {"name": joint_name, "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": _fmt(tool.mount["xyz"]), "rpy": _fmt(tool.mount["rpy"])},
    )
    ET.SubElement(joint, "parent", {"link": parent_frame})
    ET.SubElement(joint, "child", {"link": link_name})

    insert_index = next(
        (index for index, element in enumerate(root) if element.tag == "ros2_control"),
        len(root),
    )
    root.insert(insert_index, link)
    root.insert(insert_index + 1, joint)

    tcp_link_name = f"{arm_id}_{tool.name}_tcp_link"
    tcp_link = ET.Element("link", {"name": tcp_link_name})
    tcp_joint = ET.Element(
        "joint", {"name": f"{arm_id}_{tool.name}_tcp_joint", "type": "fixed"}
    )
    ET.SubElement(
        tcp_joint,
        "origin",
        {"xyz": _fmt(tool.tcp["xyz"]), "rpy": _fmt(tool.tcp["rpy"])},
    )
    ET.SubElement(tcp_joint, "parent", {"link": parent_frame})
    ET.SubElement(tcp_joint, "child", {"link": tcp_link_name})
    root.insert(insert_index + 2, tcp_link)
    root.insert(insert_index + 3, tcp_joint)


def _render_robot_description(
    xacro_path: str,
    mappings: Dict[str, str],
    tool: Optional[ToolDescription],
    arm_id: str,
) -> str:
    import xacro

    document = xacro.process_file(xacro_path, mappings=mappings)
    root = ET.fromstring(document.toxml())
    if tool is not None:
        _append_tool_to_urdf(root, tool, arm_id)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _render_monitor_urdf(
    base_urdf_path: str, tool: Optional[ToolDescription], arm_id: str
) -> str:
    root = ET.parse(base_urdf_path).getroot()
    if arm_id != "panda":
        for element in root.iter():
            for attribute in ("name", "link"):
                if attribute in element.attrib:
                    element.attrib[attribute] = element.attrib[attribute].replace(
                        "panda_", f"{arm_id}_"
                    )
    if tool is not None:
        _append_tool_to_urdf(root, tool, arm_id)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _append_mujoco_geom(
    body: ET.Element,
    geometry: Dict[str, Any],
    name: str,
    geom_class: str,
    rgba: Optional[Sequence[float]],
    friction: Optional[Sequence[float]],
    asset: ET.Element,
) -> None:
    attributes = {
        "name": name,
        "class": geom_class,
        "pos": _fmt(geometry["xyz"]),
        "quat": _fmt(_quaternion_wxyz_from_rpy(geometry["rpy"])),
    }
    kind = geometry["type"]
    if kind == "sphere":
        attributes.update({"type": "sphere", "size": _fmt([geometry["radius"]])})
    elif kind == "box":
        attributes.update({"type": "box", "size": _fmt([0.5 * item for item in geometry["size"]])})
    elif kind == "cylinder":
        attributes.update(
            {"type": "cylinder", "size": _fmt([geometry["radius"], 0.5 * geometry["length"]])}
        )
    else:
        mesh_name = f"{name}_asset"
        ET.SubElement(
            asset,
            "mesh",
            {"name": mesh_name, "file": geometry["mesh"], "scale": _fmt(geometry["scale"])},
        )
        attributes.update({"type": "mesh", "mesh": mesh_name})
    if rgba is not None:
        attributes["rgba"] = _fmt(rgba)
    if friction is not None:
        attributes["friction"] = _fmt(friction)
    ET.SubElement(body, "geom", attributes)


def _render_mujoco_robot(
    base_mjcf_path: str, tool: Optional[ToolDescription], arm_id: str
) -> str:
    root = ET.parse(base_mjcf_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((Path(base_mjcf_path).parent / "assets").resolve()))
    if arm_id != "panda":
        raise ToolModelError("the current single-arm MuJoCo model requires arm_id=panda")
    if tool is None:
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    link8 = root.find(".//body[@name='panda_link8']")
    asset = root.find("asset")
    if link8 is None or asset is None:
        raise ToolModelError("MuJoCo base model does not contain panda_link8 or asset")

    body_name = f"panda_{tool.name}"
    tool_body = ET.SubElement(
        link8,
        "body",
        {
            "name": body_name,
            "pos": _fmt(tool.mount["xyz"]),
            "quat": _fmt(_quaternion_wxyz_from_rpy(tool.mount["rpy"])),
            "gravcomp": "1" if tool.gravity_compensation else "0",
        },
    )
    inertia = tool.inertia
    ET.SubElement(
        tool_body,
        "inertial",
        {
            "mass": _fmt([tool.mass]),
            "pos": _fmt(tool.com),
            "fullinertia": _fmt(
                [
                    inertia["ixx"], inertia["iyy"], inertia["izz"],
                    inertia["ixy"], inertia["ixz"], inertia["iyz"],
                ]
            ),
        },
    )
    _append_mujoco_geom(
        tool_body,
        tool.visual,
        f"{body_name}_visual",
        "visual",
        tool.visual_rgba,
        None,
        asset,
    )
    _append_mujoco_geom(
        tool_body,
        tool.collision,
        f"{body_name}_collision",
        "collision",
        None,
        tool.friction,
        asset,
    )
    ET.SubElement(
        link8,
        "site",
        {
            "name": f"{body_name}_tcp_site",
            "pos": _fmt(tool.tcp["xyz"]),
            "size": "0.006",
            "type": "sphere",
            "rgba": "1 0.9 0.1 1",
        },
    )
    if tool.touch_sensor:
        site_name = f"{body_name}_touch_site"
        ET.SubElement(
            tool_body,
            "site",
            {
                "name": site_name,
                "pos": _fmt(tool.safety_center_tool),
                "type": "sphere",
                "size": _fmt([tool.safety_radius * 1.01]),
                "rgba": "1 0.2 0.2 0.15",
            },
        )
        sensor = root.find("sensor")
        if sensor is None:
            sensor = ET.SubElement(root, "sensor")
        ET.SubElement(sensor, "touch", {"name": f"{body_name}_touch", "site": site_name})

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _render_scene(robot_mjcf_path: str, table_path: str, include_table: bool) -> str:
    root = ET.Element("mujoco", {"model": "panda scene"})
    ET.SubElement(root, "include", {"file": str(Path(robot_mjcf_path).resolve())})
    if include_table:
        ET.SubElement(root, "include", {"file": str(Path(table_path).resolve())})
    ET.SubElement(root, "statistic", {"center": "0.3 0 0.4", "extent": "1"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "groundplane",
        },
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _controller_config(
    source_path: str,
    monitor_urdf_path: str,
    tool: Optional[ToolDescription],
) -> Dict[str, Any]:
    with Path(source_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for section in (
        "cartesian_impedance_example_controller",
        "nonlinear_impedance_controller",
    ):
        parameters = config.get(section, {}).get("ros__parameters", {})
        if "urdf_model_path" in parameters:
            parameters["urdf_model_path"] = monitor_urdf_path
    reachable = config.get("reachable_cartesian_impedance_controller", {}).get(
        "ros__parameters", {}
    )
    reachable["monitor_urdf_model_path"] = monitor_urdf_path
    if tool is not None:
        reachable["tcp_offset"] = list(tool.tcp["xyz"])
        reachable["ee_collision_center_offset"] = list(tool.collision_center_from_tcp)
        reachable["ee_collision_radius"] = tool.safety_radius
        reachable["enable_mujoco_contact_logging"] = tool.touch_sensor
        reachable["mujoco_contact_sensor_topic"] = f"/panda_{tool.name}_touch"
    return config


def generate_tool_artifacts(
    *,
    tool_config_path: str,
    arm_id: str,
    robot_xacro_path: str,
    xacro_mappings: Dict[str, str],
    monitor_base_urdf_path: str,
    output_root: str,
    mujoco_base_path: Optional[str] = None,
    mujoco_table_path: Optional[str] = None,
    include_table: bool = False,
    controller_config_path: Optional[str] = None,
) -> ToolArtifacts:
    tool = load_tool_description(tool_config_path) if tool_config_path else None
    inputs = [robot_xacro_path, monitor_base_urdf_path]
    if tool_config_path:
        inputs.append(tool_config_path)
    for optional_path in (mujoco_base_path, mujoco_table_path, controller_config_path):
        if optional_path:
            inputs.append(optional_path)
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(Path(path).resolve().read_bytes())
    digest.update(repr(sorted(xacro_mappings.items())).encode("utf-8"))
    digest.update(str(include_table).encode("ascii"))
    output_directory = Path(output_root).expanduser().resolve() / digest.hexdigest()[:16]
    output_directory.mkdir(parents=True, exist_ok=True)

    robot_description = _render_robot_description(
        robot_xacro_path, xacro_mappings, tool, arm_id
    )
    monitor_urdf = _render_monitor_urdf(monitor_base_urdf_path, tool, arm_id)
    monitor_path = output_directory / "panda_monitor_with_tool.urdf"
    monitor_path.write_text(monitor_urdf, encoding="utf-8")

    scene_path: Optional[Path] = None
    generated_controller_path: Optional[Path] = None
    if mujoco_base_path:
        robot_mjcf = _render_mujoco_robot(mujoco_base_path, tool, arm_id)
        robot_mjcf_path = output_directory / "panda_with_tool.xml"
        robot_mjcf_path.write_text(robot_mjcf, encoding="utf-8")
        if not mujoco_table_path:
            raise ToolModelError("mujoco_table_path is required when generating a scene")
        scene_path = output_directory / "scene_with_tool.xml"
        scene_path.write_text(
            _render_scene(str(robot_mjcf_path), mujoco_table_path, include_table),
            encoding="utf-8",
        )
    if controller_config_path:
        generated_controller_path = output_directory / "controllers_with_tool.yaml"
        generated_controller_path.write_text(
            yaml.safe_dump(
                _controller_config(controller_config_path, str(monitor_path), tool),
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    return ToolArtifacts(
        robot_description=robot_description,
        monitor_urdf_path=str(monitor_path),
        mujoco_scene_path=str(scene_path) if scene_path else None,
        controller_config_path=(
            str(generated_controller_path) if generated_controller_path else None
        ),
        tool=tool,
    )
