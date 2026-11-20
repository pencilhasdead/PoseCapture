"""MMD 骨骼 -> Mixamo（``mixamorig:``）改名 + 层级整理，**保留头发 / 裙子等非标准骨骼**。

功能对应 MMD Bone Tools（`QIangIQ/MMD-bone-SDK`，MIT）的 “convert --to mixamo”，
只是搬进本插件、直接在 Blender 里改骨架：

1. 标准人体骨按对照表改名（下半身→Hips、上半身→Spine、足→UpLeg、ひざ→Leg、
   腕→Arm、ひじ→ForeArm、手首→Hand、指 1/2/3/先→HandXxx1..4…）；
2. 把「上半身」挂到「下半身(Hips)」下面 —— MMD 里它俩是兄弟，Mixamo 要求 Spine 是 Hips
   的子级；不修这一步，Mixamo 动画转 Hips 时上半身不会跟着动；
3. MMD 控制骨（全ての親 / センター / グルーブ / 肩P / 肩C / 足ＩＫ / つま先ＩＫ / 足D…）
   默认删掉，它们的权重并进父骨；
4. **其他骨骼一律保留**：头发、裙子、袖、胸、捩り、目、あご…；名字可选翻成英文
   （左髪1 → hair1.L）；
5. 同步重命名顶点组（否则蒙皮全丢），被删骨的权重按“最近保留的祖先”合并；
6. 摘掉 MMD Tools 的追加变换（驱动器 / 约束），否则改名后它们还会把骨骼拉回去。

骨骼表整理自 MMD Bone Tools（MIT License, QIangIQ/MMD-bone-SDK）公开的
jp/en/blender/unity/vrm/mixamo 对照表，只取了本插件用到的部分。

``plan_conversion`` 不 import bpy，可以脱离 Blender 单测。
"""
from __future__ import annotations

import re
import unicodedata

MIXAMO_PREFIX = "mixamorig:"
MIXAMO_PREFIXES = ("mixamorig:", "mixamorig1:", "")

# 控制骨：Mixamo 没有槽位，默认删除（权重并进父骨）
CONTROL_IDS = {
    "master", "center", "groove", "waist_cancel",
    "shoulder_p", "shoulder_c", "hand_dummy",
    "leg_ik_parent", "leg_ik", "toe_ik",
    "leg_d", "knee_d", "ankle_d",
}

# id, jp, en, blender, mixamo（{J}=左/右，{s}=L/R，{Side}=Left/Right；空 = 没有槽位）
_ROWS = (
    ("master", "全ての親", "master|mother|root|all parent", "master|root", ""),
    ("center", "センター", "center|centre", "center", ""),
    ("groove", "グルーブ", "groove", "groove", ""),
    ("waist", "腰", "waist", "waist", ""),
    ("lower_body", "下半身", "lower body|lowerbody|lower_body", "lower_body", "Hips"),
    ("upper_body", "上半身", "upper body|upperbody|upper_body", "upper_body", "Spine"),
    ("upper_body2", "上半身2", "upper body2|upper body 2|upper_body2", "upper_body2", "Spine1"),
    ("upper_body3", "上半身3", "upper body3|upper body 3|upper_body3", "upper_body3", "Spine2"),
    ("neck", "首", "neck", "neck", "Neck"),
    ("head", "頭", "head", "head", "Head"),
    ("head_tip", "頭先", "head tip|head_tip", "head_tip", "HeadTop_End"),
    ("eye", "{J}目|{J}眼", "eye_{s}", "eye.{s}", "{Side}Eye"),
    ("eyes", "両目", "eyes|both eyes", "eyes", ""),
    ("jaw", "あご|顎", "jaw", "jaw", ""),
    ("tongue", "舌", "tongue", "tongue", ""),
    ("waist_cancel", "腰キャンセル{J}", "waist cancel_{s}|waistcancel_{s}", "waist_cancel.{s}", ""),
    ("shoulder_p", "{J}肩P", "shoulderP_{s}|shoulder p_{s}", "shoulderP.{s}", ""),
    ("shoulder", "{J}肩", "shoulder_{s}", "shoulder.{s}", "{Side}Shoulder"),
    ("shoulder_c", "{J}肩C", "shoulderC_{s}|shoulder c_{s}", "shoulderC.{s}", ""),
    ("arm", "{J}腕", "arm_{s}", "arm.{s}", "{Side}Arm"),
    ("arm_twist", "{J}腕捩", "arm twist_{s}|armtwist_{s}|arm_twist_{s}", "arm_twist.{s}", ""),
    ("arm_twist1", "{J}腕捩1", "arm twist1_{s}", "arm_twist1.{s}", ""),
    ("arm_twist2", "{J}腕捩2", "arm twist2_{s}", "arm_twist2.{s}", ""),
    ("arm_twist3", "{J}腕捩3", "arm twist3_{s}", "arm_twist3.{s}", ""),
    ("elbow", "{J}ひじ|{J}肘", "elbow_{s}", "elbow.{s}", "{Side}ForeArm"),
    ("wrist_twist", "{J}手捩", "wrist twist_{s}|wristtwist_{s}|hand twist_{s}", "wrist_twist.{s}", ""),
    ("wrist_twist1", "{J}手捩1", "wrist twist1_{s}", "wrist_twist1.{s}", ""),
    ("wrist_twist2", "{J}手捩2", "wrist twist2_{s}", "wrist_twist2.{s}", ""),
    ("wrist_twist3", "{J}手捩3", "wrist twist3_{s}", "wrist_twist3.{s}", ""),
    ("wrist", "{J}手首", "wrist_{s}|hand_{s}", "wrist.{s}", "{Side}Hand"),
    ("hand_dummy", "{J}ダミー", "dummy_{s}", "dummy.{s}", ""),
    ("thumb0", "{J}親指0", "thumb0_{s}", "thumb0.{s}", "{Side}HandThumb1"),
    ("thumb1", "{J}親指1", "thumb1_{s}", "thumb1.{s}", "{Side}HandThumb2"),
    ("thumb2", "{J}親指2", "thumb2_{s}", "thumb2.{s}", "{Side}HandThumb3"),
    ("thumb_tip", "{J}親指先", "thumb tip_{s}|thumb3_{s}", "thumb_tip.{s}", "{Side}HandThumb4"),
    ("index1", "{J}人指1|{J}人差指1", "fore1_{s}|index1_{s}", "index1.{s}", "{Side}HandIndex1"),
    ("index2", "{J}人指2|{J}人差指2", "fore2_{s}|index2_{s}", "index2.{s}", "{Side}HandIndex2"),
    ("index3", "{J}人指3|{J}人差指3", "fore3_{s}|index3_{s}", "index3.{s}", "{Side}HandIndex3"),
    ("index_tip", "{J}人指先|{J}人差指先", "fore tip_{s}|index tip_{s}", "index_tip.{s}", "{Side}HandIndex4"),
    ("middle1", "{J}中指1", "middle1_{s}", "middle1.{s}", "{Side}HandMiddle1"),
    ("middle2", "{J}中指2", "middle2_{s}", "middle2.{s}", "{Side}HandMiddle2"),
    ("middle3", "{J}中指3", "middle3_{s}", "middle3.{s}", "{Side}HandMiddle3"),
    ("middle_tip", "{J}中指先", "middle tip_{s}|middle4_{s}", "middle_tip.{s}", "{Side}HandMiddle4"),
    ("ring1", "{J}薬指1", "third1_{s}|ring1_{s}", "ring1.{s}", "{Side}HandRing1"),
    ("ring2", "{J}薬指2", "third2_{s}|ring2_{s}", "ring2.{s}", "{Side}HandRing2"),
    ("ring3", "{J}薬指3", "third3_{s}|ring3_{s}", "ring3.{s}", "{Side}HandRing3"),
    ("ring_tip", "{J}薬指先", "third tip_{s}|ring tip_{s}", "ring_tip.{s}", "{Side}HandRing4"),
    ("little1", "{J}小指1", "little1_{s}|pinky1_{s}", "little1.{s}", "{Side}HandPinky1"),
    ("little2", "{J}小指2", "little2_{s}|pinky2_{s}", "little2.{s}", "{Side}HandPinky2"),
    ("little3", "{J}小指3", "little3_{s}|pinky3_{s}", "little3.{s}", "{Side}HandPinky3"),
    ("little_tip", "{J}小指先", "little tip_{s}|pinky tip_{s}", "little_tip.{s}", "{Side}HandPinky4"),
    ("leg", "{J}足", "leg_{s}", "leg.{s}", "{Side}UpLeg"),
    ("knee", "{J}ひざ|{J}膝", "knee_{s}", "knee.{s}", "{Side}Leg"),
    ("ankle", "{J}足首", "ankle_{s}", "ankle.{s}", "{Side}Foot"),
    ("toe", "{J}つま先|{J}爪先", "toe_{s}", "toe.{s}", "{Side}Toe_End"),
    ("leg_ik_parent", "{J}足IK親", "leg IK parent_{s}", "leg_IK_parent.{s}", ""),
    ("leg_ik", "{J}足IK|{J}足ＩＫ", "leg IK_{s}|legIK_{s}", "leg_IK.{s}", ""),
    ("toe_ik", "{J}つま先IK|{J}つま先ＩＫ", "toe IK_{s}|toeIK_{s}", "toe_IK.{s}", ""),
    ("leg_d", "{J}足D", "leg D_{s}|legD_{s}", "leg_D.{s}", ""),
    ("knee_d", "{J}ひざD", "knee D_{s}|kneeD_{s}", "knee_D.{s}", ""),
    ("ankle_d", "{J}足首D", "ankle D_{s}|ankleD_{s}", "ankle_D.{s}", ""),
    ("toe_ex", "{J}足先EX", "toe EX_{s}|toeEX_{s}", "toe_EX.{s}", "{Side}ToeBase"),
)

# 头发 / 裙子 / 服饰词表：非标准骨改名用（最长匹配）
_EXTRA_WORDS = (
    # 身体部位（捩り / D 骨这类非标准骨的名字里会出现）
    ("上半身", "upper_body"), ("下半身", "lower_body"), ("センター", "center"),
    ("両目", "eyes"), ("あご", "jaw"), ("顎", "jaw"), ("舌", "tongue"),
    ("手首", "wrist"), ("足首", "ankle"), ("つま先", "toe"),
    ("ひじ", "elbow"), ("肘", "elbow"), ("ひざ", "knee"), ("膝", "knee"),
    ("親指", "thumb"), ("人指", "index"), ("人差指", "index"),
    ("中指", "middle"), ("薬指", "ring"), ("小指", "little"), ("指", "finger"),
    ("肩", "shoulder"), ("腕", "arm"), ("手", "hand"), ("足", "leg"), ("脚", "leg"),
    ("首", "neck"), ("頭", "head"), ("目", "eye"),
    # 头发 / 裙子 / 服饰
    ("おっぱい", "breast"), ("胸", "breast"), ("お尻", "butt"), ("尻", "butt"),

    ("髪の毛", "hair"), ("前髪", "bangs"), ("後髪", "back_hair"), ("後ろ髪", "back_hair"),
    ("横髪", "side_hair"), ("側髪", "side_hair"), ("側髮", "side_hair"),
    ("劉海", "bangs"), ("刘海", "bangs"),
    ("馬尾", "ponytail"), ("马尾", "ponytail"),
    ("もみあげ", "sideburn"), ("アホ毛", "ahoge"), ("ツインテール", "twintail"),
    ("ポニーテール", "ponytail"), ("三つ編み", "braid"), ("おさげ", "braid"),
    ("お団子", "bun"), ("髪", "hair"), ("髮", "hair"),
    ("スカート", "skirt"), ("裙", "skirt"), ("裾", "hem"), ("摆", "hem"), ("襬", "hem"),
    ("袖", "sleeve"), ("襟", "collar"), ("領", "collar"),
    ("リボン", "ribbon"), ("ネクタイ", "necktie"), ("領結", "necktie"),
    ("結", "knot"), ("帶", "belt"), ("帯", "belt"), ("带", "belt"),
    ("マフラー", "scarf"),

    ("マント", "cape"), ("フード", "hood"), ("ベルト", "belt"), ("フリル", "frill"),
    ("手袋", "glove"), ("ブーツ", "boots"), ("ニーソ", "knee_socks"), ("靴", "shoe"),
    ("尻尾", "tail"), ("しっぽ", "tail"), ("ケモミミ", "kemomimi"), ("猫耳", "cat_ear"),
    ("翼", "wing"), ("羽", "wing"), ("角", "horn"), ("耳", "ear"),
    ("飾り", "ornament"), ("アクセ", "accessory"), ("鎖", "chain"), ("鈴", "bell"),
    ("ダミー", "dummy"), ("調整", "adjust"), ("補助", "helper"), ("操作", "control"),
    ("捩", "twist"), ("前", "front"), ("後", "back"), ("横", "side"),
    ("上", "upper"), ("下", "lower"), ("中", "mid"), ("先", "tip"), ("根元", "root"),
)


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKC", str(name)).strip().lower()


# Unity Humanoid 的骨架槽位（按 Mixamo 骨名后缀）：简化时只保留这些「身体骨」
UNITY_HUMANOID = {
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "{Side}Eye", "{Side}Shoulder", "{Side}Arm", "{Side}ForeArm", "{Side}Hand",
    # Unity 的 Toes 槽在 MMD 里可能只有「つま先」(Toe_End)，两种都留着
    "{Side}UpLeg", "{Side}Leg", "{Side}Foot", "{Side}ToeBase", "{Side}Toe_End",
    "{Side}HandThumb1", "{Side}HandThumb2", "{Side}HandThumb3",
    "{Side}HandIndex1", "{Side}HandIndex2", "{Side}HandIndex3",
    "{Side}HandMiddle1", "{Side}HandMiddle2", "{Side}HandMiddle3",
    "{Side}HandRing1", "{Side}HandRing2", "{Side}HandRing3",
    "{Side}HandPinky1", "{Side}HandPinky2", "{Side}HandPinky3",
}

# 这些“非标准骨”要保留（头发 / 裙子 / 飘带 / 胸 / 饰品 / 武器…），简化时不删
ACCESSORY_WORDS = (
    # 头发
    "髪", "髮", "hair", "bangs", "劉海", "刘海", "アホ毛", "ahoge", "ツインテ", "twintail",
    "ポニテ", "馬尾", "马尾", "ponytail", "braid", "お団子", "sideburn", "もみあげ",
    # 裙子 / 服饰
    "スカート", "skirt", "裙", "裾", "hem", "摆", "襬", "袖", "sleeve", "襟", "collar",
    "リボン", "ribbon", "ネクタイ", "necktie", "領結", "結", "紐", "ひも", "string",
    "帶", "带", "飘带", "飄帶", "sash", "マフラー", "scarf", "マント", "cape", "veil",
    # 身体上要单独动的“软”部位
    "胸", "breast", "おっぱい", "尻", "butt", "お尻", "腹", "belly",
    # 饰品 / 尾巴 / 翅膀 / 道具
    "尻尾", "しっぽ", "tail", "翼", "羽", "wing", "角", "horn", "耳", "ear",
    "飾", "ornament", "アクセ", "accessory", "鎖", "chain", "鈴", "bell",
    "武器", "剣", "刀", "sword", "gun", "杖", "staff", "傘", "umbrella", "prop",
)


def is_accessory(name: str) -> bool:
    """名字看起来是头发 / 裙子 / 飘带 / 饰品（简化时要保留）。"""
    text = unicodedata.normalize("NFKC", str(name))
    lowered = text.lower()
    return any(word in text or word.lower() in lowered for word in ACCESSORY_WORDS)


_UNITY_BASES = {template.replace("{Side}", "") for template in UNITY_HUMANOID}


def is_unity_humanoid_bone(mixamo_name: str | None) -> bool:
    """Mixamo 名字是否落在 Unity Humanoid 的骨架槽位里（每手 3 节手指、没有脚趾尖）。"""
    text = str(mixamo_name or "")
    if not text:
        return False
    for side in ("Left", "Right"):
        if text.startswith(side):
            return text[len(side):] in _UNITY_BASES
    return text in _UNITY_BASES



def _expand(template: str, side: str) -> str:
    """把 {J}/{s}/{Side} 换成对应左右；side 为空时不替换（非左右骨）。"""
    if not side:
        return template
    out = template
    for key, values in (("J", {"L": "左", "R": "右"}),
                        ("s", {"L": "L", "R": "R"}),
                        ("Side", {"L": "Left", "R": "Right"})):
        out = out.replace("{%s}" % key, values[side])
    return out


def _aliases(template: str, side: str) -> tuple[str, ...]:
    return tuple(part for part in (_expand(chunk, side) for chunk in template.split("|")) if part)


def _build() -> list[dict]:
    table = []
    for row_id, jp, en, blender, mixamo in _ROWS:
        sided = "{J}" in jp or "{s}" in en or "{Side}" in mixamo
        for side in (("L", "R") if sided else ("",)):
            table.append({
                "id": "%s.%s" % (row_id, side) if side else row_id,
                "base": row_id,
                "side": side,
                "jp": _aliases(jp, side),
                "en": _aliases(en, side),
                "blender": _aliases(blender, side),
                "mixamo": _expand(mixamo, side) if mixamo else "",
            })
    return table


BONES = _build()
_INDEX: dict[str, dict[str, dict]] = {"jp": {}, "en": {}, "blender": {}}
for _bone in BONES:
    for _conv in ("jp", "en", "blender"):
        for _alias in _bone[_conv]:
            _INDEX[_conv].setdefault(_norm(_alias), _bone)

# 反向索引：已经换成 Mixamo 命名的骨（Hips / LeftHandMiddle1 / LeftToe_End）也能认回来
_MIXAMO_INDEX: dict[str, dict] = {}
for _bone in BONES:
    if _bone["mixamo"]:
        _MIXAMO_INDEX.setdefault(_norm(_bone["mixamo"]), _bone)


def _strip_prefix(name: str) -> str:
    """去掉 ``mixamorig:`` / ``Armature|mixamorig:`` 这类前缀，只留骨名。"""
    text = unicodedata.normalize("NFKC", str(name)).strip()
    if "|" in text:
        text = text.rsplit("|", 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text


def lookup(name: str) -> dict | None:
    """按 jp → en → blender 顺序查标准骨（认得 左腕 / arm_L / arm.L 三种写法）。

    已经转换成 Mixamo 命名的骨（``Hips`` / ``LeftHandMiddle1`` / ``mixamorig:Hips``）
    也能认回来，所以同一根骨架重复转换是“什么都不改”。
    """
    key = _norm(name)
    for conv in ("jp", "en", "blender"):
        hit = _INDEX[conv].get(key)
        if hit is not None:
            return hit
    hit = _MIXAMO_INDEX.get(_norm(_strip_prefix(name)))
    if hit is not None:
        return hit
    return None




def mixamo_name(name: str, prefix: str = MIXAMO_PREFIX) -> str | None:
    """标准骨的 Mixamo 名字；非标准骨返回 None。"""
    entry = lookup(name)
    if entry is None or not entry["mixamo"]:
        return None
    return prefix + entry["mixamo"]



def classify(name: str) -> tuple[str, dict | None]:
    """返回 ('standard'|'control'|'extra', entry)。

    extra = 头发 / 裙子 / 捩り / 目 之类“要保留但没有 Mixamo 槽位”的骨骼。
    """
    entry = lookup(name)
    if entry is not None and entry["mixamo"]:
        return "standard", entry
    if entry is not None and entry["base"] in CONTROL_IDS:
        return "control", entry
    return "extra", entry


def _split_side(name: str) -> tuple[str, str]:
    text = unicodedata.normalize("NFKC", str(name)).strip()
    side = ""
    for index, char in enumerate(text):
        if char in ("左", "右"):
            side = "L" if char == "左" else "R"
            text = (text[:index] + text[index + 1:]).strip()
            break
    return side, text


def _translate_words(text: str) -> str | None:
    """按词表把日文翻成英文；还有翻不了的字符就返回 None（那就保留原名）。

    数字直接贴在前一个词后面，`-` / `_` 这类分隔符变成 `_`：
    「後髪1-1」→ ``back_hair1_1``、「スカート_0_1」→ ``skirt0_1``。
    """
    words = sorted(_EXTRA_WORDS, key=lambda item: len(item[0]), reverse=True)
    pieces: list[tuple[str, str]] = []          # (kind, text)  kind: word / num / sep
    index = 0
    while index < len(text):
        char = text[index]
        for jp, eng in words:
            if text.startswith(jp, index):
                pieces.append(("word", eng))
                index += len(jp)
                break
        else:
            if char.isdigit():
                pieces.append(("num", char))
            elif char in " _-.":
                pieces.append(("sep", ""))
            elif char.isascii() and char.isalnum():
                pieces.append(("word", char))
            else:
                return None
            index += 1
    if not pieces:
        return None
    out = ""
    for kind, piece in pieces:
        if kind == "num":
            out += piece
        elif kind == "sep":
            if out and not out.endswith("_"):
                out += "_"
        else:
            out += ("_" if out and not out.endswith("_") else "") + piece
    return out.strip("_") or None



def translate_extra_name(name: str) -> str:
    """头发 / 裙子 / 服饰骨的名字翻成英文（左髪1 → hair1.L）；翻不了就原样返回。

    纯 ASCII 的名字直接返回（保证对已经转换过的骨架再跑一次不会改动任何东西）。
    """
    if not name or not re.search(r"[^\x00-\x7f]", name):
        return str(name)
    side, stem = _split_side(name)
    if not stem:
        return name
    translated = _translate_words(stem)
    if not translated or re.fullmatch(r"[A-Za-z0-9_.+\-]+", translated) is None:
        return name
    return translated + (".%s" % side if side else "")



# --------------------------------------------------------------------------- 计划
def plan_conversion(bone_names, parents=None, prefix: str = MIXAMO_PREFIX,
                    cleanup: bool = True, translate_extras: bool = True,
                    simplify: bool = False) -> dict:
    """纯逻辑：算出要改哪些名字 / 删哪些骨 / 谁挂到谁下面（不动 Blender 数据）。

    ``simplify=True`` 时按 **Unity Humanoid** 的标准精简身体骨：只留
    Hips / Spine / Spine1 / Spine2 / Neck / Head / 眼 / 肩 / 臂 / 前臂 / 手 /
    UpperLeg / LowerLeg / Foot / Toes + 每只手 3 节手指，其余身体杂骨（腕捩 / 手捩 /
    腰 / 上半身1 / 手先 …）删掉，权重并进最近的保留祖先；头发 / 裙子 / 飘带 / 胸 /
    饰品等**全部保留**。
    """
    names = [str(name) for name in bone_names]
    parents = {str(key): (str(value) if value else None)
               for key, value in (parents or {}).items()}
    notes: list[str] = []
    rename: dict[str, str] = {}
    delete: list[str] = []
    kinds: dict[str, str] = {}
    hips = spine = None

    for name in names:
        kind, entry = classify(name)
        kinds[name] = kind
        if kind == "standard":
            target = prefix + entry["mixamo"]
            if target != name:              # 已经是 Mixamo 名字就不要再“改名”
                rename[name] = target
            if entry["base"] == "lower_body":
                hips = name
            elif entry["base"] == "upper_body":
                spine = name
        elif kind == "control":
            if cleanup:
                delete.append(name)
        elif translate_extras:
            target = translate_extra_name(name)
            if target and target != name:
                rename[name] = target

    if hips is None:
        center = next((name for name in names
                       if (lookup(name) or {}).get("base") == "center"), None)
        if center is not None:
            hips = center
            rename[center] = prefix + "Hips"
            if center in delete:
                delete.remove(center)
            notes.append("没有「下半身」：用「センター」当 Hips")
        else:
            notes.append("⚠ 找不到「下半身 / センター」，Mixamo 的 Hips 无法确定")

    # 按 Unity Humanoid 简化：身体骨只留 UNITY_HUMANOID 那套，其余“身体上的杂骨”
    # （腕捩 / 手捩 / 腰 / 上半身1 / 手先 / 目先 …）删掉并把权重并进最近的保留祖先；
    # 头发 / 裙子 / 飘带 / 胸 / 饰品一律不动。
    simplified: list[str] = []
    if simplify:
        for name in names:
            if name in delete or name in (hips, spine):
                continue
            if is_accessory(name):
                continue
            kind = kinds.get(name)
            if kind == "control" and not cleanup:
                continue
            if kind == "standard":
                # 标准骨里也有 Unity Humanoid 不要的：手指第 4 节 / 脚趾尖（Toe_End）
                if is_unity_humanoid_bone((lookup(name) or {}).get("mixamo")):
                    continue
            if name not in delete:
                delete.append(name)
                simplified.append(name)
            rename.pop(name, None)          # 要删的骨不用改名
        if simplified:
            notes.append("按 Unity Humanoid 简化：身体杂骨删 %d 根（权重并入父骨）"
                         % len(simplified))

    # 被删骨的权重 / 子级要找“最近一个保留的祖先”
    def kept_ancestor(name: str):
        seen = set()
        parent = parents.get(name)
        while parent is not None and parent not in seen:
            seen.add(parent)
            if parent not in delete:
                return parent
            parent = parents.get(parent)
        return None

    reparent: dict[str, str | None] = {}
    kept = [name for name in names if name not in delete]

    if spine and hips and parents.get(spine) != hips:
        reparent[spine] = hips
        notes.append("把「%s」挂到「%s」下（Mixamo 要求 Spine 是 Hips 的子级）" % (spine, hips))
    if hips and hips in parents and parents.get(hips) in delete:
        reparent[hips] = kept_ancestor(hips)
        notes.append("「%s」改成根骨（原父级是控制骨）" % hips)

    for name in kept:
        if name in reparent:
            continue
        parent = parents.get(name)
        if parent is not None and parent in delete:
            reparent[name] = kept_ancestor(name)

    deleted_targets = {name: kept_ancestor(name) for name in delete}

    def _has(text: str, words) -> bool:
        lowered = _norm(text)
        return any(word in text or word.lower() in lowered for word in words)

    hair = [name for name in kept
            if _has(name, ("髪", "髮", "hair", "アホ毛", "bangs", "劉海", "刘海",
                           "ツインテ", "ポニテ", "馬尾", "马尾", "ponytail", "braid",
                           "お団子", "twintail", "ahoge"))]
    skirt = [name for name in kept
             if _has(name, ("スカート", "skirt", "裾", "hem", "裙", "摆", "襬"))]

    return {
        "prefix": prefix,
        "rename": rename,
        "delete": delete,
        "reparent": reparent,
        "deleted_targets": deleted_targets,
        "kinds": kinds,
        "hips": hips,
        "spine": spine,
        "hair": hair,
        "skirt": skirt,
        "kept": kept,
        "simplified": simplified,
        "notes": notes,
    }





def summarize(plan: dict) -> str:
    """给面板 / 日志用的一行说明。"""
    rename = plan.get("rename") or {}
    prefixed = [target for target in rename.values()
                if any(target.startswith(prefix) for prefix in MIXAMO_PREFIXES if prefix)]
    parts = ["改名 %d 根（Mixamo %d 根）" % (len(rename), len(prefixed)),
             "删控制骨 %d 根" % len([name for name in (plan.get("delete") or [])
                                     if name not in (plan.get("simplified") or [])]),
             "保留头发 %d / 裙子 %d" % (len(plan.get("hair") or []),
                                       len(plan.get("skirt") or []))]
    if plan.get("simplified"):
        parts.append("Unity Humanoid 精简 %d 根" % len(plan["simplified"]))
    if plan.get("reparent"):
        parts.append("调整层级 %d 处" % len(plan["reparent"]))
    return "；".join(parts)



# --------------------------------------------------------------------------- 执行（需要 bpy）
def find_meshes(arm_obj) -> list:
    """找出受这根骨架影响的网格：有指向它的 Armature 修改器，或顶点组和骨骼同名。"""
    import bpy

    bones = {bone.name for bone in arm_obj.data.bones}
    found = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(modifier.type == "ARMATURE" and modifier.object == arm_obj
               for modifier in obj.modifiers):
            found.append(obj)
            continue
        if bones and any(group.name in bones for group in obj.vertex_groups):
            found.append(obj)
    return found


def strip_mmd_helpers(arm_obj) -> dict:
    """摘掉 MMD Tools 的追加变换（骨骼约束 + 驱动器）：改名后它们会把骨骼拉回原位。"""
    removed = {"constraints": 0, "drivers": 0}
    for pose_bone in arm_obj.pose.bones:
        for constraint in list(pose_bone.constraints):
            pose_bone.constraints.remove(constraint)
            removed["constraints"] += 1
    for owner in (arm_obj, arm_obj.data):
        animation = getattr(owner, "animation_data", None)
        if animation is None or not animation.drivers:
            continue
        for driver in list(animation.drivers):
            animation.drivers.remove(driver)
            removed["drivers"] += 1
    return removed


def _merge_group(obj, source, target_name: str, report: dict) -> None:
    target = obj.vertex_groups.get(target_name)
    if target is None:
        source.name = target_name
        return
    weights = []
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            if element.group == source.index:
                weights.append((vertex.index, element.weight))
                break
    for index, weight in weights:
        existing = 0.0
        for element in obj.data.vertices[index].groups:
            if element.group == target.index:
                existing = element.weight
                break
        target.add([index], min(1.0, existing + weight), "REPLACE")
    obj.vertex_groups.remove(source)
    report["groups_merged"] += 1


def apply_plan(arm_obj, plan: dict, rename_groups: bool = True,
               merge_weights: bool = True) -> dict:
    """按 ``plan_conversion`` 的结果改骨架（含顶点组），需要 Blender 环境。"""
    import bpy

    report = {"renamed": 0, "deleted": 0, "reparented": 0,
              "groups_renamed": 0, "groups_merged": 0, "warnings": []}
    meshes = find_meshes(arm_obj)

    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_obj.data.edit_bones

    # 1) 层级：Spine 挂到 Hips 下 / 被删父级的子骨找新父级
    #    必须在改名之前做 —— plan 里的名字是改名前的老名字
    for name, parent in (plan.get("reparent") or {}).items():
        bone = edit_bones.get(name)
        target = edit_bones.get(parent) if parent else None
        if bone is None:
            continue
        if parent and target is None:
            report["warnings"].append("想挂到 %s 下，但它已经不在骨架里" % parent)
            continue
        bone.parent = target
        bone.use_connect = False
        report["reparented"] += 1

    # 2) 改名：先挪到临时名，避免 A->B / B->A 撞车、或被 Blender 自动加 .001
    #    要被删掉的骨不用改名（否则顶点组会先被改名、后面合并找不到）
    deleted = set(plan.get("delete") or [])
    staged = {}
    for old, new in (plan.get("rename") or {}).items():
        if old in deleted:
            continue
        bone = edit_bones.get(old)
        if bone is None or old == new:
            continue

        temp = "__pc_tmp__%03d" % len(staged)
        bone.name = temp
        staged[temp] = (old, new)
    for temp, (old, new) in staged.items():
        bone = edit_bones.get(temp)
        if bone is None:
            report["warnings"].append("改名时找不到 %s" % old)
            continue
        bone.name = new
        report["renamed"] += 1

    # 3) 删控制骨（子级已经在第 1 步挪走）
    for name in (plan.get("delete") or []):
        bone = edit_bones.get(name)
        if bone is None:
            continue
        edit_bones.remove(bone)
        report["deleted"] += 1

    bpy.ops.object.mode_set(mode="OBJECT")

    # 4) 顶点组：改名 / 把被删骨的权重并进祖先 / 清掉没有去处的孤儿组
    if rename_groups:
        mapping = {old: new for old, new in (plan.get("rename") or {}).items()
                   if old not in set(plan.get("delete") or [])}
        deleted_targets = plan.get("deleted_targets") or {}

        if merge_weights:
            rename = plan.get("rename") or {}
            for old, target in deleted_targets.items():
                if target and old not in mapping:
                    mapping[old] = rename.get(target, target)
        for obj in meshes:
            for old, new in list(mapping.items()):
                if old == new:
                    continue
                source = obj.vertex_groups.get(old)
                if source is None:
                    continue
                if obj.vertex_groups.get(new) is None:
                    source.name = new
                    report["groups_renamed"] += 1
                else:
                    _merge_group(obj, source, new, report)
            for old, target in deleted_targets.items():
                if target or old in mapping:
                    continue
                orphan = obj.vertex_groups.get(old)
                if orphan is not None:
                    obj.vertex_groups.remove(orphan)
                    report["groups_removed"] = report.get("groups_removed", 0) + 1
                    report["warnings"].append(
                        "%s 没有保留的祖先，权重被丢弃（顶点组已删）" % old)
    elif merge_weights:
        report["warnings"].append("没有重命名顶点组：蒙皮会和新骨名对不上")


    arm_obj.data.pose_position = "POSE"
    for obj in bpy.context.view_layer.objects:
        obj.update_tag()
    bpy.context.view_layer.update()
    report["meshes"] = [obj.name for obj in meshes]
    return report

