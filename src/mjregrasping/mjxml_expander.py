from pathlib import Path
from xml.etree import ElementTree as ET


class MujocoXmlExpander:

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        for include in self.root.findall('include'):
            file = include.attrib['file']
            with open(f'models/{file}', 'r') as include_file:
                include_root = ET.fromstring(include_file.read())
                self.root.extend(include_root)

        # remove include elements now that they've been "included"
        for include in self.root.findall('include'):
            self.root.remove(include)

    def get_mesh(self, mesh_name):
        for asset in self.root.findall("asset"):
            for mesh in asset.findall("mesh"):
                name = mesh.attrib['name']
                file = mesh.attrib['file']
                if name == mesh_name:
                    return file
        return None

    # iterate over all worldbody elements
    # and all body elements within them recursively
    def get_e(self, tag: str, name: str):
        for e in self.root.iter(tag):
            if "name" in e.attrib and e.attrib["name"] == name:
                return e

    def get_vec(self, e, k: str):
        x_str = e.attrib[k]
        x = [float(x) for x in x_str.split()]
        return x

    def set_vec(self, e, x, k: str):
        x_str = " ".join([str(x) for x in x])
        e.attrib[k] = x_str

    def set_vec_i(self, e, k: str, i: int, x_i: float):
        x = self.get_vec(e, k)
        x[i] = x_i
        self.set_vec(e, x, k)

    def save_tmp(self, tmp_filename='models/tmp.xml'):
        self.tree.write(tmp_filename)
        return Path(tmp_filename)
