"""Rigify 肢体写入的“纯逻辑”自检 —— **普通 Python 就能跑，不需要 Blender**。

覆盖的是不需要 bpy 的那部分逻辑（用假的 bpy / mathutils 载入真实插件模块）：
  1. 模式感知的自动匹配：IK 模式优先匹配 `upper_arm_ik` / `hand_ik` / `foot_ik` 这些控制器，
     FK 模式（或不指定）保持 `*_fk` 优先；没有 IK 控制器时自动退回普通骨名；
  2. IK 直写计划：链根 + 末端控制器、IK 目标骨、手腕 / 脚踝关键点角色；
  3. “这次写到了哪条肢体”的识别（写到 IK 控制器时也要认得出来）+ 直写触发条件；
  4. 关键帧要覆盖的控制器、写入方式选项解析。

用法：
  python dev_tests/test_limb_logic.py

（真骨架上的验证还是要 Blender：见 dev_tests/test_rigify_limbs.py）
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REAL = os.path.join(ROOT, "pose_capture")


class _Any:
    """万能假对象：既能当类用（继承），也能当函数 / 属性用。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Any()

    def __getattr__(self, name):
        return _Any

    def __setattr__(self, name, value):
        pass

    def __getitem__(self, key):
        return _Any()

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __contains__(self, key):
        return False

    def __add__(self, other):
        return _Any()

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __add__

    def __hash__(self):
        return id(self)


def stub(name):
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: _Any      # PEP 562
    sys.modules[name] = module
    return module


for _name in ("bpy", "mathutils", "numpy"):
    stub(_name)
sys.modules["mathutils"].Vector = _Any
sys.modules["mathutils"].Matrix = _Any

# 造一个假包，把真实文件按模块名装进去（相对 import 需要包上下文）
package = types.ModuleType("pctest")
package.__path__ = []
sys.modules["pctest"] = package
stub("pctest.utils")                 # 只有 “from . import utils” 这种引用，不会真的调用
stub("pctest.reconstruction")        # 同上，避免连累 numpy


def load(name):
    spec = importlib.util.spec_from_file_location("pctest." + name,
                                                  os.path.join(REAL, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["pctest." + name] = module
    spec.loader.exec_module(module)
    return module


rigify_map = load("rigify_map")
rigify_limbs = load("rigify_limbs")
retarget = load("retarget")

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)


class Bone:
    def __init__(self, name, z=0.0):
        self.name = name
        self.head_local = types.SimpleNamespace(z=z)


class Bones:
    def __init__(self, names):
        self._map = {}
        for index, name in enumerate(names):
            self._map[name] = Bone(name, z=0.2 * index)
        for name in self._map:                       # 肩膀高度给个高处，脊椎截断才准
            if name.startswith(("spine_fk", "neck")):
                self._map[name].head_local = types.SimpleNamespace(z=0.5)
            if name.startswith(("upper_arm", "shoulder")):
                self._map[name].head_local = types.SimpleNamespace(z=1.5)

    def __getitem__(self, name):
        return self._map[name]

    def get(self, name):
        return self._map.get(name)

    def __iter__(self):
        return iter(self._map.values())

    def __contains__(self, name):
        return name in self._map


class Arm:
    def __init__(self, names):
        self.name = "FakeRig"
        self.data = types.SimpleNamespace(bones=Bones(names))


SPINE = ["spine_fk"] + ["spine_fk.%03d" % i for i in range(1, 6)]
LIMB_COMMON = ["torso", "root", "neck", "head"]
GENERATED = SPINE + LIMB_COMMON + [
    "shoulder.L", "shoulder.R",
    "upper_arm_fk.L", "upper_arm_fk.R", "forearm_fk.L", "forearm_fk.R",
    "hand_fk.L", "hand_fk.R", "palm.01.L", "palm.01.R",
    "thigh_fk.L", "thigh_fk.R", "shin_fk.L", "shin_fk.R",
    "foot_fk.L", "foot_fk.R", "toe_fk.L", "toe_fk.R",
    "upper_arm_ik.L", "upper_arm_ik.R", "forearm_ik.L", "forearm_ik.R",
    "upper_arm_ik_target.L", "upper_arm_ik_target.R", "hand_ik.L", "hand_ik.R",
    "thigh_ik.L", "thigh_ik.R", "shin_ik.L", "shin_ik.R",
    "thigh_ik_target.L", "thigh_ik_target.R", "foot_ik.L", "foot_ik.R",
    "toe_ik.L", "toe_ik.R", "foot_heel_ik.L", "foot_heel_ik.R",
    "foot_spin_ik.L", "foot_spin_ik.R",
    "upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R",
]
METARIG = SPINE + LIMB_COMMON + ["upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
                                 "hand.L", "hand.R", "thigh.L", "thigh.R", "shin.L",
                                 "shin.R", "foot.L", "foot.R", "toe.L", "toe.R",
                                 "shoulder.L", "shoulder.R"]

print("[1] 模式感知的自动匹配")
rig = Arm(GENERATED)
bmap_fk = rigify_map.auto_detect(rig, False)
bmap_ik = rigify_map.auto_detect(rig, False, "IK")
print("    FK 映射：", {r: bmap_fk.get(r) for r in ("upper_arm.L", "forearm.R", "hand.L",
                                                   "thigh.R", "shin.L", "foot.R")})
print("    IK 映射：", {r: bmap_ik.get(r) for r in ("upper_arm.L", "forearm.R", "hand.L",
                                                   "thigh.R", "shin.L", "foot.R")})
check(bmap_fk.get("upper_arm.L") == "upper_arm_fk.L" and bmap_fk.get("hand.R") == "hand_fk.R",
      "不指定 limb_mode 时仍然是 FK 优先（老行为）", bmap_fk.get("upper_arm.L"))
check(bmap_ik.get("upper_arm.L") == "upper_arm_ik.L" and bmap_ik.get("hand.R") == "hand_ik.R",
      "IK 模式优先匹配 IK 控制器", bmap_ik.get("upper_arm.L"))
check(bmap_ik.get("forearm.L") == "forearm_ik.L" and bmap_ik.get("foot.R") == "foot_ik.R",
      "小臂 / 脚也匹配 IK 控制器", "%s / %s" % (bmap_ik.get("forearm.L"),
                                                bmap_ik.get("foot.R")))
check(bmap_ik.get("spine_00") == bmap_fk.get("spine_00") == "spine_fk",
      "脊椎映射不受写入方式影响", str(bmap_ik.get("spine_00")))
check(bmap_fk.get("shoulder.L") == "shoulder.L" and bmap_fk.get("root") == "root",
      "锁骨 / 根骨映射正常", "%s / %s" % (bmap_fk.get("shoulder.L"), bmap_fk.get("root")))

metarig = Arm(METARIG)
bmap_meta = rigify_map.auto_detect(metarig, False, "IK")
check(bmap_meta.get("upper_arm.L") == "upper_arm.L" and bmap_meta.get("hand.L") == "hand.L",
      "没有 IK 控制器时 IK 模式退回普通骨名", bmap_meta.get("upper_arm.L"))
check(rigify_map.ordered_candidates("upper_arm.{s}", "L", "IK")[0] == "upper_arm_ik.L"
      and rigify_map.ordered_candidates("upper_arm.{s}", "L", "FK")[0] == "upper_arm_fk.L"
      and rigify_map.ordered_candidates("upper_arm.{s}", "L")[0] == "upper_arm_fk.L",
      "候选顺序随写入方式变化")

print("[2] IK 直写计划")
ARM_SPEC_L = {"prop_bone": "upper_arm_parent.L", "pole_prop": "pole_vector",
              "fk_bones": ["upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"],
              "ik_bones": ["upper_arm_ik.L", "MCH-forearm_ik.L",
                           "MCH-upper_arm_ik_target.L"],
              "ctrl_bones": ["upper_arm_ik.L", "upper_arm_ik_target.L", "hand_ik.L"],
              "tail_bones": [], "extra_ctrls": [], "kind": "Arm", "side": "L"}
LEG_SPEC_R = {"prop_bone": "thigh_parent.R", "pole_prop": "pole_vector",
              "fk_bones": ["thigh_fk.R", "shin_fk.R", "foot_fk.R", "toe_fk.R"],
              "ik_bones": ["thigh_ik.R", "MCH-shin_ik.R", "MCH-thigh_ik_target.R"],
              "ctrl_bones": ["thigh_ik.R", "thigh_ik_target.R", "foot_ik.R"],
              "tail_bones": ["toe_ik.R"],
              "extra_ctrls": ["foot_heel_ik.R", "foot_spin_ik.R"],
              "kind": "Leg", "side": "R"}
plans = {"upper_arm_fk.L": ARM_SPEC_L, "thigh_fk.R": LEG_SPEC_R}
directs = {key: rigify_limbs.ik_write_plan(spec) for key, spec in plans.items()}
for key, plan in sorted(directs.items()):
    print("    %-18s -> %s  目标 %s  终点 %s" % (key, plan["controls"], plan["target"],
                                             plan["tip_role"]))
check(directs["upper_arm_fk.L"]["controls"] == {"upper_arm.L": "upper_arm_ik.L",
                                               "hand.L": "hand_ik.L"},
      "大臂只直写链根 + 末端控制器（小臂交给 IK 解算）",
      str(directs["upper_arm_fk.L"]["controls"]))
check(directs["upper_arm_fk.L"]["target"] == "upper_arm_ik_target.L"
      and directs["upper_arm_fk.L"]["tip_role"] == "forearm.L"
      and directs["upper_arm_fk.L"]["tip_fk"] == "forearm_fk.L",
      "手腕关键点 / IK 目标骨正确")
check(directs["thigh_fk.R"]["controls"] == {"thigh.R": "thigh_ik.R",
                                            "foot.R": "foot_ik.R",
                                            "toe.R": "toe_ik.R"},
      "大腿直写链根 + 脚 + 脚趾控制器", str(directs["thigh_fk.R"]["controls"]))
check(directs["thigh_fk.R"]["target"] == "thigh_ik_target.R"
      and directs["thigh_fk.R"]["tip_role"] == "shin.R"
      and directs["thigh_fk.R"]["tip_fk"] == "shin_fk.R",
      "脚踝关键点 / IK 目标骨正确")
check(rigify_limbs.ik_write_plan({"side": "L", "fk_bones": ["upper_arm_fk.L"]}) is None,
      "信息不全时直写计划返回 None")

print("[3] “写过的肢体”识别 + 直写触发")
check(rigify_limbs.mapped_limb_mode(None, bmap_fk, plans) == "FK", "FK 映射认成 FK",
      str(rigify_limbs.mapped_limb_mode(None, bmap_fk, plans)))
check(rigify_limbs.mapped_limb_mode(None, bmap_ik, plans) == "IK", "IK 映射认成 IK",
      str(rigify_limbs.mapped_limb_mode(None, bmap_ik, plans)))
check(rigify_limbs.mapped_limb_mode(None, rigify_map.BoneMap(), plans) == "",
      "空映射返回空串")

written_ik = retarget._limbs_written({"upper_arm_ik.L", "hand_ik.L"}, plans)
check(sorted(written_ik) == ["upper_arm_fk.L"],
      "写到 IK 控制器时也能认出是哪条肢体", str(sorted(written_ik)))
written_fk = retarget._limbs_written({"forearm_fk.L", "thigh_fk.R"}, plans)
check(sorted(written_fk) == ["thigh_fk.R", "upper_arm_fk.L"],
      "写到 FK 骨骼时照样认（老行为）", str(sorted(written_fk)))
check(retarget._limbs_written({"spine_fk"}, plans) == {}, "没写到的肢体不算")
check(sorted(retarget.direct_limb_plans(bmap_ik, written_ik)) == ["upper_arm_fk.L"],
      "IK 映射 -> 触发直写", str(sorted(retarget.direct_limb_plans(bmap_ik, written_ik))))
check(retarget.direct_limb_plans(bmap_fk, written_fk) == {},
      "FK 映射 -> 不直写（走老路径）")
check(retarget.direct_limb_plans(rigify_map.BoneMap(), written_ik) == {},
      "映射为空时不直写")

print("[4] 关键帧要覆盖的控制器")
kf = retarget._limb_keyframe_bones(plans, "IK")
check("hand_ik.L" in kf and "upper_arm_ik_target.L" in kf and "foot_heel_ik.R" in kf,
      "IK 控制器 / IK 目标 / 脚跟都有关键帧", str(sorted(kf)))
check(retarget._limb_keyframe_bones(plans, "FK") == [], "FK 模式不加控制器关键帧")
check(retarget.limb_mode_option({"limb_mode": "IK"}) == "IK"
      and retarget.limb_mode_option({"fk_mode": True}) == "FK"
      and retarget.limb_mode_option({}) == "IK", "写入方式选项解析正常")

print("结果：", "通过 ✓" if not failures else "未通过 ✗")
for item in failures:
    print("   -", item)
sys.exit(0 if not failures else 1)
