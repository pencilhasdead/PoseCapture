"""MMD → Mixamo 转换的纯逻辑自检（不需要 Blender）：改哪些名 / 删哪些骨 / 谁挂谁下面。

用法：
  python dev_tests/test_mmd_mixamo.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.dirname(HERE) + os.sep + "pose_capture"

spec = importlib.util.spec_from_file_location("mmd_mixamo", os.path.join(REAL, "mmd_mixamo.py"))
mmd = importlib.util.module_from_spec(spec)
sys.modules["mmd_mixamo"] = mmd
spec.loader.exec_module(mmd)

failures = []


def check(ok, label, detail=""):
    print("  %s %s %s" % ("OK " if ok else "!! ", label, detail))
    if not ok:
        failures.append(label)
    return ok


# 一棵典型 MMD 骨架：名字 -> 父级
PARENTS = {
    "全ての親": None,
    "センター": "全ての親",
    "グルーブ": "センター",
    "上半身": "センター",
    "上半身2": "上半身",
    "首": "上半身2",
    "頭": "首",
    "左肩P": "上半身2",
    "左肩": "左肩P",
    "左腕": "左肩",
    "左腕捩": "左腕",
    "左ひじ": "左腕捩",
    "左手首": "左ひじ",
    "左中指1": "左手首",
    "左中指2": "左中指1",
    "左中指3": "左中指2",
    "右肩": "上半身2",
    "右腕": "右肩",
    "右ひじ": "右腕",
    "右手首": "右ひじ",
    "下半身": "センター",
    "左足": "下半身",
    "左ひざ": "左足",
    "左足首": "左ひざ",
    "左つま先": "左足首",
    "右足": "下半身",
    "右ひざ": "右足",
    "右足首": "右ひざ",
    "右つま先": "右足首",
    "左足ＩＫ": "全ての親",
    "左つま先ＩＫ": "左足ＩＫ",
    "左足D": "左足",
    "左ひざD": "左ひざ",
    "左髪1": "頭",
    "左髪2": "左髪1",
    "右髪1": "頭",
    "左馬尾1-1": "頭",
    "中劉海1": "頭",
    "スカート_0_1": "下半身",
    "スカート_0_2": "スカート_0_1",
    "裙_1_2": "下半身",

    "胸": "上半身2",
    "謎骨": "上半身2",
}


def main():
    print("MMD → Mixamo 纯逻辑自检")
    names = list(PARENTS)
    plan = mmd.plan_conversion(names, PARENTS)

    print("\n[1] 标准骨 -> Mixamo")
    expect = {"下半身": "mixamorig:Hips", "上半身": "mixamorig:Spine",
              "上半身2": "mixamorig:Spine1", "首": "mixamorig:Neck",
              "頭": "mixamorig:Head", "左腕": "mixamorig:LeftArm",
              "左ひじ": "mixamorig:LeftForeArm", "左手首": "mixamorig:LeftHand",
              "左中指1": "mixamorig:LeftHandMiddle1", "左中指3": "mixamorig:LeftHandMiddle3",
              "右足": "mixamorig:RightUpLeg", "右ひざ": "mixamorig:RightLeg",
              "右足首": "mixamorig:RightFoot", "右つま先": "mixamorig:RightToe_End"}
    bad = {name: (plan["rename"].get(name), target) for name, target in expect.items()
           if plan["rename"].get(name) != target}
    check(not bad, "14 根关键骨名字正确", bad or "%d 根" % len(expect))
    check(plan["rename"].get("左肩") == "mixamorig:LeftShoulder", "肩 -> Shoulder")
    check(len(mmd.BONES) >= 100, "标准骨表条数", "%d" % len(mmd.BONES))

    print("\n[2] 控制骨删除 / 层级修正")
    for name in ("全ての親", "センター", "グルーブ", "左肩P", "左足ＩＫ",
                 "左つま先ＩＫ", "左足D", "左ひざD"):
        check(name in plan["delete"], "删 %s" % name)
    check(plan["reparent"].get("上半身") == "下半身", "Spine 挂到 Hips 下",
          str(plan["reparent"].get("上半身")))
    check(plan["reparent"].get("下半身") is None and "下半身" in plan["reparent"],
          "Hips 成为根骨")
    check(plan["reparent"].get("左肩") == "上半身2", "肩P 被删后肩改挂上半身2",
          str(plan["reparent"].get("左肩")))

    print("\n[3] 头发 / 裙子 / 其他骨：只改名不删")
    for name in ("左髪1", "右髪1", "スカート_0_1", "スカート_0_2", "胸", "謎骨"):
        check(name in plan["kept"], "保留 %s" % name)
    check(plan["rename"].get("左髪1") == "hair1.L", "左髪1 -> hair1.L",
          plan["rename"].get("左髪1"))
    check(plan["rename"].get("右髪1") == "hair1.R", "右髪1 -> hair1.R")
    check(plan["rename"].get("スカート_0_1") == "skirt_0_1", "スカート_0_1 -> skirt_0_1",
          plan["rename"].get("スカート_0_1"))
    check(plan["rename"].get("裙_1_2") == "skirt_1_2", "裙_1_2 -> skirt_1_2（中文裙）")
    check(plan["rename"].get("左馬尾1-1") == "ponytail1_1.L", "左馬尾1-1 -> ponytail1_1.L")
    check(plan["rename"].get("中劉海1") == "mid_bangs1", "中劉海1 -> mid_bangs1",
          plan["rename"].get("中劉海1"))


    check(plan["rename"].get("胸") == "breast", "胸 -> breast")
    check("謎骨" not in plan["rename"], "翻不了的骨名保持原样")
    check(len(plan["hair"]) == 5, "认出 5 根头发骨", str(plan["hair"]))
    check(len(plan["skirt"]) == 3, "认出 3 根裙子骨", str(plan["skirt"]))


    print("\n[4] 权重 / 顶点组去向")
    check(plan["deleted_targets"].get("左足D") == "左足", "足D 的权重并进 足",
          str(plan["deleted_targets"].get("左足D")))
    check(plan["deleted_targets"].get("左肩P") == "上半身2", "肩P 的权重并进 上半身2")
    check(plan["deleted_targets"].get("センター") is None, "センター 没有保留的祖先（权重会丢）")

    print("\n[5] 其他写法 / 幂等性")
    check(mmd.mixamo_name("arm_L") == "mixamorig:Arm".replace("mixamorig:Arm", "mixamorig:LeftArm"),
          "认得 en 写法 arm_L", str(mmd.mixamo_name("arm_L")))
    check(mmd.mixamo_name("arm.L") == "mixamorig:LeftArm", "认得 blender 写法 arm.L")
    second = mmd.plan_conversion([plan["rename"].get(n, n) for n in plan["kept"]],
                                 {plan["rename"].get(k, k): (plan["rename"].get(v, v) if v else None)
                                  for k, v in PARENTS.items() if k in plan["kept"]})
    check(not second["rename"] and not second["delete"], "对已经转换过的骨架再跑一次 = 不改",
          "改名 %d / 删 %d" % (len(second["rename"]), len(second["delete"])))

    print("\n[6] 前缀 / 说明")
    no_prefix = mmd.plan_conversion(["下半身", "上半身"], {"下半身": None, "上半身": "下半身"},
                                    prefix="")
    check(no_prefix["rename"] == {"下半身": "Hips", "上半身": "Spine"}, "无前缀模式",
          str(no_prefix["rename"]))
    text = mmd.summarize(plan)
    check("改名" in text and "头发" in text, "summarize 文案", text[:60])

    print("\n[7] Unity Humanoid 简化（simplify=True）")
    simple = mmd.plan_conversion(names, PARENTS, simplify=True)
    check("左腕捩" in simple["delete"], "腕捩 被简化掉",
          str(simple["deleted_targets"].get("左腕捩")))
    check(simple["deleted_targets"].get("左腕捩") == "左腕", "腕捩 的权重并进 左腕",
          str(simple["deleted_targets"].get("左腕捩")))
    check("謎骨" in simple["delete"], "认不出的身体杂骨被简化掉")
    check("謎骨" not in simple["rename"], "要删的骨不再改名")
    for name in ("左髪1", "右髪1", "左馬尾1-1", "中劉海1", "スカート_0_1", "裙_1_2", "胸"):
        check(name in simple["kept"], "简化后仍保留 %s" % name)
    check(simple["rename"].get("胸") == "breast", "胸 保留并改名 breast")
    for name in ("下半身", "上半身", "上半身2", "首", "頭", "左腕", "左ひじ", "左手首",
                 "右足", "右ひざ", "右足首", "左中指1"):
        check(name not in simple["delete"], "标准骨不删 %s" % name)
    unchanged = mmd.plan_conversion(names, PARENTS)
    check(len(simple["delete"]) >= len(unchanged["delete"]) + 2,
          "简化多删了几根", "%d -> %d" % (len(unchanged["delete"]), len(simple["delete"])))
    check("Unity Humanoid" in "；".join(simple["notes"]), "notes 里说明简化了多少",
          "；".join(simple["notes"])[:60])
    check("精简" in mmd.summarize(simple), "summarize 里有精简数量",
          mmd.summarize(simple)[:80])
    again = mmd.plan_conversion([simple["rename"].get(n, n) for n in simple["kept"]],
                                {simple["rename"].get(k, k):
                                 (simple["rename"].get(v, v) if v else None)
                                 for k, v in PARENTS.items() if k in simple["kept"]},
                                simplify=True)
    check(not again["delete"], "转换过的骨架再跑 simplify = 不再删",
          "删 %d" % len(again["delete"]))

    print()
    if failures:
        print("!! 失败 %d 项：%s" % (len(failures), ", ".join(failures)))
        return 1

    print("通过 (all OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
