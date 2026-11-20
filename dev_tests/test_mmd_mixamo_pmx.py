"""在**真实 PMX 模型**上验证 MMD → Mixamo 的映射（不需要 Blender，也不装 MMD Tools）。

只解析 PMX 的骨骼表（骨架名 + 父级），然后跑插件的 ``plan_conversion``，检查：
  * 下半身 / 上半身 有没有认出来（Hips / Spine）；
  * 标准骨是否都映射到了 Mixamo 名字，且**没有重名**；
  * 头发 / 裙子骨有多少、是否一根都没被删；
  * 有哪些骨名翻不成英文（保留原样，不影响使用）。

用法：
  python dev_tests/test_mmd_mixamo_pmx.py "路径\\model.pmx"
  （不给参数就自动找仓库里那份「流萤」模型）
"""
import glob
import importlib.util
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location(
    "mmd_mixamo", os.path.join(ROOT, "pose_capture", "mmd_mixamo.py"))
mmd = importlib.util.module_from_spec(spec)
sys.modules["mmd_mixamo"] = mmd
spec.loader.exec_module(mmd)

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.encoding = "utf-16"

    def take(self, count: int) -> bytes:
        chunk = self.data[self.pos:self.pos + count]
        if len(chunk) != count:
            raise ValueError("PMX 读到文件末尾（pos=%d, 还要 %d）" % (self.pos, count))
        self.pos += count
        return chunk

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def uint8(self):
        return self.unpack("<B")[0]

    def uint16(self):
        return self.unpack("<H")[0]

    def int32(self):
        return self.unpack("<i")[0]

    def float32(self):
        return self.unpack("<f")[0]

    def floats(self, count: int):
        return self.unpack("<%df" % count)

    def text(self) -> str:
        length = self.int32()
        raw = self.take(length)
        if self.encoding == "utf-16":
            return raw.decode("utf-16-le", "replace")
        return raw.decode("utf-8", "replace")

    def index(self, size: int) -> int:
        if size == 1:
            return struct.unpack("<b", self.take(1))[0]
        if size == 2:
            return struct.unpack("<h", self.take(2))[0]
        return struct.unpack("<i", self.take(4))[0]


def parse_bones(path: str):
    """返回 (bone_names, parents, version)。只读骨骼表，其余段按规范跳过。"""
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:4] != b"PMX ":
        raise ValueError("不是 PMX 文件（magic=%r）" % data[:4])
    reader = Reader(data)
    reader.pos = 4
    version = reader.float32()
    globals_count = reader.uint8()
    globals_bytes = reader.take(globals_count)
    reader.encoding = "utf-16" if globals_bytes[0] == 0 else "utf-8"
    additional_uv = globals_bytes[1]
    vertex_index_size = globals_bytes[2]
    texture_index_size = globals_bytes[3]
    bone_index_size = globals_bytes[5]
    for _ in range(4):                      # model name / en / comment / en
        reader.text()

    vertex_count = reader.int32()
    for _ in range(vertex_count):
        reader.floats(3 + 3 + 2)            # pos / normal / uv
        if additional_uv:
            reader.floats(additional_uv * 4)
        weight_type = reader.uint8()
        if weight_type == 0:
            reader.index(bone_index_size)
        elif weight_type == 1:
            reader.index(bone_index_size)
            reader.index(bone_index_size)
            reader.float32()
        elif weight_type in (2, 4):
            for _ in range(4):
                reader.index(bone_index_size)
            reader.floats(4)
        elif weight_type == 3:
            reader.index(bone_index_size)
            reader.index(bone_index_size)
            reader.float32()
            reader.floats(9)
        else:
            raise ValueError("未知的权重类型 %d（pos=%d）" % (weight_type, reader.pos))
        reader.float32()                    # エッジ倍率（PMX 2.0 也有，不是 2.1 才加）


    reader.take(reader.int32() * vertex_index_size)     # faces
    for _ in range(reader.int32()):                      # textures
        reader.text()

    for _ in range(reader.int32()):                      # materials
        reader.text()
        reader.text()
        reader.floats(4 + 3 + 1 + 3)        # diffuse / specular / power / ambient
        reader.uint8()                       # draw flag
        reader.floats(4 + 1)                 # edge color / size
        reader.index(texture_index_size)
        reader.index(texture_index_size)
        reader.uint8()                       # sphere mode
        if reader.uint8():                   # shared toon
            reader.index(texture_index_size)
        else:
            reader.take(1)
        reader.text()                        # memo
        reader.int32()                       # surface count

    names, parents = [], []
    for _ in range(reader.int32()):
        name = reader.text()
        reader.text()                        # name en
        reader.floats(3)                     # position
        parent = reader.index(bone_index_size)
        reader.int32()                       # layer
        flags = reader.uint16()
        if flags & 0x0001:
            reader.index(bone_index_size)    # tail bone
        else:
            reader.floats(3)
        if flags & 0x0100 or flags & 0x0200:  # inherit rotation / translation
            reader.index(bone_index_size)
            reader.float32()
        if flags & 0x0400:
            reader.floats(3)                 # fixed axis
        if flags & 0x0800:
            reader.floats(6)                 # local axes
        if flags & 0x2000:
            reader.int32()                   # external parent
        if flags & 0x0020:                   # IK：目标骨 / 循环次数 / 单位角 / リンク
            reader.index(bone_index_size)
            reader.int32()
            reader.float32()
            for _ in range(reader.int32()):
                reader.index(bone_index_size)
                if reader.uint8():           # 限制角度
                    reader.floats(6)
        if version >= 2.1:
            reader.int32()                   # deform layer

        names.append(name)
        parents.append(parent)
    return names, parents, version


def default_model() -> str:
    """在仓库里随便找一个“人物” PMX（跳过武器 / 剑）。"""
    found = [path for path in glob.glob(os.path.join(ROOT, "**", "*.pmx"), recursive=True)
             if "武器" not in path and "剑" not in path]
    return found[0] if found else ""



def main():
    path = sys.argv[1] if len(sys.argv) > 1 else default_model()
    if not path or not os.path.isfile(path):
        print("跳过：没有找到 PMX（用 python dev_tests/test_mmd_mixamo_pmx.py <model.pmx>）")
        return 0
    print("真实模型：%s（%.1f MB）" % (os.path.basename(path),
                                      os.path.getsize(path) / 1048576.0))
    names, parents, version = parse_bones(path)
    print("  PMX %.1f，骨骼 %d 根" % (version, len(names)))
    check(10 <= len(names) <= 2000, "骨骼数合理", "%d" % len(names))

    parent_map = {name: (names[parents[index]] if 0 <= parents[index] < len(names)
                         and parents[index] != index else None)
                  for index, name in enumerate(names)}
    plan = mmd.plan_conversion(names, parent_map)
    rename, delete = plan["rename"], plan["delete"]

    print("\n[1] 骨架映射")
    check(plan["hips"] is not None, "找到 Hips（下半身 / センター）", str(plan["hips"]))
    check(plan["spine"] is not None, "找到 Spine（上半身）", str(plan["spine"]))
    check(rename.get(plan["hips"] or "", "").endswith("Hips"),
          "Hips 映射", str(rename.get(plan["hips"] or "")))
    check(rename.get(plan["spine"] or "", "").endswith("Spine"),
          "Spine 映射", str(rename.get(plan["spine"] or "")))
    check(plan["reparent"].get(plan["spine"] or "") == plan["hips"],
          "上半身挂到下半身下")
    mixamo = [target for target in rename.values() if target.startswith("mixamorig:")]
    check(len(mixamo) >= 20, "映射到 Mixamo 的骨数", "%d" % len(mixamo))

    print("\n[2] 没有重名 / 没有把两根骨改成同一个名字")
    targets = list(rename.values())
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    check(not duplicates, "改名目标唯一", duplicates or "%d 个" % len(targets))


    print("\n[3] 头发 / 裙子骨：一根都不删")
    hair = plan["hair"]
    skirt = plan["skirt"]
    check(len(hair) >= 1, "头发骨", "%d 根：%s" % (len(hair), ", ".join(hair[:6])))
    check(len(skirt) >= 1, "裙子骨", "%d 根：%s" % (len(skirt), ", ".join(skirt[:6])))
    killed = [name for name in hair + skirt if name in delete]
    check(not killed, "头发 / 裙子一根都没被删", killed or "无")
    kept_renamed = [(name, rename[name]) for name in hair[:6] if name in rename]
    if kept_renamed:
        print("      例：" + "，".join("%s → %s" % pair for pair in kept_renamed))

    print("\n[4] 控制骨")
    check(bool(delete), "删掉的控制骨", "%d 根：%s" % (len(delete), ", ".join(delete[:8])))

    print("\n[5] 还有哪些骨名保留原样（不影响使用，只是没翻成英文）")
    leftovers = [name for name in names
                 if name not in delete and name not in rename
                 and any(ord(char) > 0x2E80 for char in name)]

    print("      %d 根：%s" % (len(leftovers), ", ".join(leftovers[:10]) or "无"))
    check(len(leftovers) <= len(names) * 0.5, "保留原样的是少数", "%d/%d"
          % (len(leftovers), len(names)))

    print("\n[6] 汇总")
    print("      %s" % mmd.summarize(plan))
    print("      保留骨骼 %d 根（含头发 %d / 裙子 %d），删除控制骨 %d 根"
          % (len(plan["kept"]), len(hair), len(skirt), len(delete)))

    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1
    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

