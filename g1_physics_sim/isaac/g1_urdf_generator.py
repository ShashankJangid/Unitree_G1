"""g1_urdf_generator.py – Generate standard 29-DoF URDF for NVIDIA Isaac Sim & ROS 2.

Parses kinematic joints, parent-child links, mesh references, limits, and inertial properties
from the official Unitree G1 29-DoF MuJoCo model, and exports a standardized URDF file
ready for NVIDIA Isaac Sim / Isaac Lab Omniverse importer.
"""

import os
import pathlib
import xml.etree.ElementTree as ET


def generate_g1_urdf(output_path: str = None) -> str:
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    if output_path is None:
        output_path = str(base_dir / "models" / "g1" / "g1_29dof.urdf")

    xml_path = base_dir / "models" / "g1" / "g1_29dof.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Source XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()


    urdf = ET.Element("robot", name="g1_29dof")


    mat_black = ET.SubElement(urdf, "material", name="black")
    ET.SubElement(mat_black, "color", rgba="0.1 0.1 0.1 1.0")
    mat_grey = ET.SubElement(urdf, "material", name="grey")
    ET.SubElement(mat_grey, "color", rgba="0.7 0.7 0.7 1.0")
    mat_metal = ET.SubElement(urdf, "material", name="metal")
    ET.SubElement(mat_metal, "color", rgba="0.3 0.3 0.3 1.0")


    mesh_rel_prefix = "meshes/"


    worldbody = root.find("worldbody")
    pelvis_body = worldbody.find("body")

    def process_body(mj_body, parent_link_name=None):
        link_name = mj_body.get("name")
        body_pos = mj_body.get("pos", "0 0 0")
        body_quat = mj_body.get("quat", "1 0 0 0")


        link_el = ET.SubElement(urdf, "link", name=link_name)


        inertial = mj_body.find("inertial")
        if inertial is not None:
            ine_el = ET.SubElement(link_el, "inertial")
            ET.SubElement(ine_el, "origin", xyz=inertial.get("pos", "0 0 0"), rpy="0 0 0")
            ET.SubElement(ine_el, "mass", value=inertial.get("mass", "1.0"))

            diaginertia = inertial.get("diaginertia", "0.01 0.01 0.01").split()
            ET.SubElement(
                ine_el,
                "inertia",
                ixx=diaginertia[0],
                ixy="0",
                ixz="0",
                iyy=diaginertia[1],
                iyz="0",
                izz=diaginertia[2],
            )
        else:
            ine_el = ET.SubElement(link_el, "inertial")
            ET.SubElement(ine_el, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ine_el, "mass", value="0.1")
            ET.SubElement(ine_el, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")


        for geom in mj_body.findall("geom"):
            mesh_name = geom.get("mesh")
            gtype = geom.get("type", "mesh" if mesh_name else "box")
            gpos = geom.get("pos", "0 0 0")
            gquat = geom.get("quat", "1 0 0 0")


            vis_el = ET.SubElement(link_el, "visual")
            ET.SubElement(vis_el, "origin", xyz=gpos, rpy="0 0 0")
            vis_geom = ET.SubElement(vis_el, "geometry")
            if mesh_name:
                mesh_file = f"package://g1_description/meshes/{mesh_name}.STL"
                ET.SubElement(vis_geom, "mesh", filename=mesh_file)
            else:
                ET.SubElement(vis_geom, "sphere", radius="0.02")
            ET.SubElement(vis_el, "material", name="grey")


            col_el = ET.SubElement(link_el, "collision")
            ET.SubElement(col_el, "origin", xyz=gpos, rpy="0 0 0")
            col_geom = ET.SubElement(col_el, "geometry")
            if mesh_name:
                mesh_file = f"package://g1_description/meshes/{mesh_name}.STL"
                ET.SubElement(col_geom, "mesh", filename=mesh_file)
            else:
                ET.SubElement(col_geom, "sphere", radius="0.02")


        joint = mj_body.find("joint")
        if parent_link_name is not None:
            joint_name = joint.get("name") if joint is not None else f"{parent_link_name}_to_{link_name}"
            joint_type = "revolute" if joint is not None else "fixed"
            joint_axis = joint.get("axis", "0 0 1") if joint is not None else "0 0 1"
            joint_range = joint.get("range", "-3.14 3.14").split() if joint is not None else ["-3.14", "3.14"]

            j_el = ET.SubElement(urdf, "joint", name=joint_name, type=joint_type)
            ET.SubElement(j_el, "parent", link=parent_link_name)
            ET.SubElement(j_el, "child", link=link_name)
            ET.SubElement(j_el, "origin", xyz=body_pos, rpy="0 0 0")

            if joint_type == "revolute":
                ET.SubElement(j_el, "axis", xyz=joint_axis)
                ET.SubElement(
                    j_el,
                    "limit",
                    lower=joint_range[0],
                    upper=joint_range[1],
                    effort="100.0",
                    velocity="10.0",
                )
                ET.SubElement(j_el, "dynamics", damping="1.5", friction="0.1")


        for child_body in mj_body.findall("body"):
            process_body(child_body, parent_link_name=link_name)


    process_body(pelvis_body)


    import xml.dom.minidom
    rough_string = ET.tostring(urdf, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_urdf = reparsed.toprettyxml(indent="  ")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_urdf)

    print(f"[SUCCESS] Exported 29-DoF URDF for NVIDIA Isaac Sim to:\n  {output_path}")
    return output_path


if __name__ == "__main__":
    generate_g1_urdf()
