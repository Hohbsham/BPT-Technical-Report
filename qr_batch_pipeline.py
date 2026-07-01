"""
QuadRemesher 全自动批量处理流水线
===================================
绕过 Blender GUI 的模态限制，直接调用 xremesh.exe 引擎。
流程：OBJ → [Blender后台] → FBX → [xremesh.exe] → retopo.FBX → [Blender后台] → OBJ

用法：
  python qr_batch_pipeline.py                  # 处理所有 excluded 样本
  python qr_batch_pipeline.py --test 3          # 测试模式：只处理 3 个
  python qr_batch_pipeline.py --test 3 --dryrun # 干跑：不实际执行
  python qr_batch_pipeline.py --resume           # 从断点续传
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import argparse
from pathlib import Path

# ============ 配置 ============

BLENDER = r"D:\UserData\Blender\blender-4.2.19-windows-x64\blender.exe"
ENGINE  = r"C:\ProgramData\Exoside\QuadRemesher\Datas_Blender\QuadRemesherEngine_1.4\xremesh.exe"
TEMP_DIR = Path(tempfile.gettempdir()) / "qr_batch"
EXCLUDED_DIRS = [
    Path(r"D:\ClothesNetData\supervised_data_excluded"),
    Path(r"D:\ClothesNetData\supervised_data_other_excluded"),
]
OUTPUT_DIR = Path(r"D:\ClothesNetData\supervised_data")  # 成功样本移到这里
PROGRESS_FILE = Path(r"D:\ClothesNetData\qr_batch_progress.json")

# 面数目标：根据服装类型分类
BIG_ITEMS_TARGET = 2500   # 大件：Dress, Tops, Pants, Skirts, One-piece, Trousers, Underwear
SMALL_ITEMS_TARGET = 1000 # 小件：Gloves, Hats, Masks, Socks, Scarves/Ties

# 分类关键词（大小写不敏感）
BIG_KEYWORDS = [
    "dress", "top", "shirt", "tshirt", "t-shirt", "pant", "trouser",
    "skirt", "one-piece", "onepiece", "underwear", "underpant", "jacket",
    "coat", "sweater", "vest", "bodysuit", "jumpsuit", "romper", "robe",
    "apron", "blazer", "cardigan", "hoodie", "pullover", "tunic", "blouse",
    "corset", "kimono", "poncho", "swimsuit", "swimwear", "bikini",
]
SMALL_KEYWORDS = [
    "glove", "hat", "mask", "sock", "scarf", "tie", "belt", "bag",
    "cap", "beanie", "beret", "bowler", "visor", "headband", "helmet",
    "mitten", "wristband", "cuff", "collar", "cravat", "bowtie",
]


# ============ 工具函数 ============

def classify_target_faces(sample_name: str) -> int:
    """根据样本名判断目标面数"""
    name_lower = sample_name.lower()
    for kw in SMALL_KEYWORDS:
        if kw in name_lower:
            return SMALL_ITEMS_TARGET
    for kw in BIG_KEYWORDS:
        if kw in name_lower:
            return BIG_ITEMS_TARGET
    # 默认按大件处理
    print(f"  [WARN] Cannot classify '{sample_name}', defaulting to {BIG_ITEMS_TARGET}")
    return BIG_ITEMS_TARGET


def collect_all_samples() -> list[tuple[Path, Path, str]]:
    """
    收集所有待处理样本。
    返回: [(source_dir, sample_dir, sample_name), ...]
    """
    samples = []
    for excluded_root in EXCLUDED_DIRS:
        if not excluded_root.exists():
            print(f"  [WARN] Directory not found: {excluded_root}")
            continue
        for sample_dir in sorted(excluded_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            gt_obj = sample_dir / "ground_truth.obj"
            if not gt_obj.exists():
                print(f"  [WARN] No ground_truth.obj in {sample_dir.name}, skipping")
                continue
            samples.append((excluded_root, sample_dir, sample_dir.name))
    return samples


# ============ Blender 操作 ============

def run_blender_script(script_path: str, *args) -> bool:
    """在 Blender 后台运行 Python 脚本"""
    cmd = [BLENDER, "--background", "--python", script_path, "--"] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [ERROR] Blender exit code {result.returncode}")
            print(f"  stdout: {result.stdout[-500:]}")
            print(f"  stderr: {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Blender timeout after 120s")
        return False
    except Exception as e:
        print(f"  [ERROR] Blender execution failed: {e}")
        return False


def obj_to_fbx(obj_path: Path, fbx_path: Path) -> bool:
    """用 Blender 后台将 OBJ 转为 FBX"""
    script = TEMP_DIR / "export_fbx.py"
    script.write_text(f"""
import bpy
import sys

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import OBJ
bpy.ops.wm.obj_import(filepath=r"{obj_path}")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print(f"Imported: {{len(obj.data.vertices)}}v, {{len(obj.data.polygons)}}f")

# Export FBX
bpy.ops.export_scene.fbx(
    filepath=r"{fbx_path}",
    use_selection=True,
    bake_anim=False,
    global_scale=1,
    apply_unit_scale=False,
    apply_scale_options='FBX_SCALE_NONE',
    use_space_transform=False,
    axis_forward='-X',
    axis_up='Z'
)
print(f"Exported FBX to: {fbx_path}")
""")
    return run_blender_script(str(script))


def fbx_to_obj(fbx_path: Path, obj_path: Path) -> bool:
    """用 Blender 后台将 FBX 转为 OBJ"""
    script = TEMP_DIR / "import_fbx.py"
    script.write_text(f"""
import bpy
import sys

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import FBX
bpy.ops.import_scene.fbx(filepath=r"{fbx_path}", global_scale=1,
                         use_manual_orientation=True, axis_forward='-X', axis_up='Z')
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print(f"Imported retopo: {{len(obj.data.vertices)}}v, {{len(obj.data.polygons)}}f")

# Count quads vs tris
quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
tris = sum(1 for p in obj.data.polygons if len(p.vertices) == 3)
others = len(obj.data.polygons) - quads - tris
print(f"FACES: total={{len(obj.data.polygons)}} quads={{quads}} tris={{tris}} other={{others}}")
print(f"QUAD_RATIO={{quads / max(len(obj.data.polygons), 1):.4f}}")

# Export OBJ
bpy.ops.wm.obj_export(
    filepath=r"{obj_path}",
    export_materials=False,
    export_uv=False,
    export_normals=True
)
print(f"Exported OBJ to: {obj_path}")
""")
    return run_blender_script(str(script))


# ============ QuadRemesher 引擎操作 ============

def run_quadremesher(input_fbx: Path, output_fbx: Path, target_faces: int) -> bool:
    """直接调用 xremesh.exe 引擎"""
    progress_file = TEMP_DIR / "progress.txt"
    settings_file = TEMP_DIR / "RetopoSettings.txt"

    # 删除旧文件
    for f in [output_fbx, progress_file]:
        if f.exists():
            f.unlink()

    # 写设置文件（参照 QuadRemesher 插件的格式）
    settings_content = f"""HostApp=Blender
HostAppVer=4.2.1
FileIn="{input_fbx}"
FileOut="{output_fbx}"
ProgressFile="{progress_file}"
TargetQuadCount={target_faces}
CurvatureAdaptivness=0
ExactQuadCount=1
UseVertexColorMap=False
UseMaterialIds=0
UseIndexedNormals=1
AutoDetectHardEdges=0
"""
    settings_file.write_text(settings_content)

    # 启动引擎
    try:
        process = subprocess.Popen(
            [ENGINE, "-s", str(settings_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print(f"  [ERROR] Failed to launch engine: {e}")
        return False

    # 等待完成（轮询 progress.txt）
    max_wait = 600  # 最多等 10 分钟
    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < max_wait:
        # 检查进程是否还活着
        if process.poll() is not None:
            if process.returncode != 0:
                stderr = process.stderr.read()[-500:] if process.stderr else ""
                print(f"  [ERROR] Engine crashed with code {process.returncode}")
                if stderr:
                    print(f"  stderr: {stderr}")
                return False
            break

        # 读进度
        if progress_file.exists():
            try:
                lines = progress_file.read_text().strip().splitlines()
                if lines:
                    val = float(lines[0])
                    if val == 2:  # SUCCESS
                        break
                    elif val < 0:  # ERROR
                        err_msg = lines[1] if len(lines) >= 2 else "Unknown error"
                        print(f"  [ERROR] Engine error {val}: {err_msg}")
                        process.terminate()
                        return False
                    elif val != last_progress:
                        last_progress = val
                        pct = int(val * 100)
                        if pct % 20 == 0:
                            print(f"  ... {pct}%")
            except (ValueError, IndexError):
                pass

        time.sleep(1.0)

    else:
        print(f"  [ERROR] Engine timeout after {max_wait}s")
        process.terminate()
        return False

    # 验证输出
    if not output_fbx.exists():
        print(f"  [ERROR] Engine did not produce output: {output_fbx}")
        return False

    file_size = output_fbx.stat().st_size
    if file_size < 1000:
        print(f"  [ERROR] Output FBX too small ({file_size} bytes)")
        return False

    print(f"  Engine complete: {file_size} bytes")
    return True


# ============ 主流程 ============

def process_one_sample(source_root: Path, sample_dir: Path, sample_name: str,
                       target_faces: int, dryrun: bool) -> dict | None:
    """
    处理单个样本。
    返回: 结果 dict，失败返回 None
    """
    print(f"\n{'='*60}")
    print(f"[{sample_name}] target={target_faces} faces")
    print(f"  Source: {sample_dir}")

    gt_obj = sample_dir / "ground_truth.obj"
    work_dir = TEMP_DIR / sample_name
    work_dir.mkdir(parents=True, exist_ok=True)

    input_fbx = work_dir / f"{sample_name}_input.fbx"
    retopo_fbx = work_dir / f"{sample_name}_retopo.fbx"
    retopo_obj = work_dir / f"{sample_name}_retopo.obj"

    if dryrun:
        print(f"  [DRYRUN] Would process: {gt_obj} -> {retopo_obj}")
        return {"name": sample_name, "target": target_faces, "dryrun": True}

    # Step 1: OBJ → FBX
    print(f"  Step 1/3: OBJ → FBX...")
    if not obj_to_fbx(gt_obj, input_fbx):
        return None

    # Step 2: QuadRemesher engine
    print(f"  Step 2/3: QuadRemeshing (target={target_faces})...")
    if not run_quadremesher(input_fbx, retopo_fbx, target_faces):
        return None

    # Step 3: FBX → OBJ + 验证
    print(f"  Step 3/3: FBX → OBJ + verify...")
    if not fbx_to_obj(retopo_fbx, retopo_obj):
        return None

    # 解析 Blender 输出的面信息
    # (fbx_to_obj 已经在stdout打印了面统计，这里用trimesh验证)
    quad_ratio = check_quad_ratio(retopo_obj)
    if quad_ratio is None:
        return None

    result = {
        "name": sample_name,
        "source_root": str(source_root),
        "target": target_faces,
        "quad_ratio": quad_ratio,
        "output_obj": str(retopo_obj),
    }

    if quad_ratio >= 0.95:
        print(f"  [OK] SUCCESS: quad_ratio={quad_ratio:.1%}")
        result["status"] = "success"
    elif quad_ratio >= 0.80:
        print(f"  [WARN] PARTIAL: quad_ratio={quad_ratio:.1%} (mostly quads)")
        result["status"] = "partial"
    else:
        print(f"  [FAIL] STILL TRIANGLES: quad_ratio={quad_ratio:.1%}")
        result["status"] = "failed"

    return result


def check_quad_ratio(obj_path: Path) -> float | None:
    """用简单的 OBJ 解析检查四边面比例（不依赖 trimesh）"""
    try:
        content = obj_path.read_text()
        quads = 0
        tris = 0
        for line in content.splitlines():
            if line.startswith("f "):
                parts = line.split()
                n_verts = len(parts) - 1
                if n_verts == 3:
                    tris += 1
                elif n_verts == 4:
                    quads += 1
        total = quads + tris
        if total == 0:
            print(f"  [ERROR] No faces found in OBJ")
            return None
        return quads / total
    except Exception as e:
        print(f"  [ERROR] Cannot read OBJ: {e}")
        return None


def move_to_supervised_data(sample_name: str, source_root: str, retopo_obj: Path):
    """将处理好的样本移入 supervised_data"""
    dest_dir = OUTPUT_DIR / sample_name
    # 如果目标已存在（旧 QuadriFlow 版本），先备份
    if dest_dir.exists():
        backup = OUTPUT_DIR / f"{sample_name}_old_quadriflow"
        shutil.move(str(dest_dir), str(backup))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 复制新的 ground_truth.obj
    shutil.copy2(str(retopo_obj), str(dest_dir / "ground_truth.obj"))

    # 从原 excluded 目录复制或重新生成 input 文件
    source_sample_dir = Path(source_root) / sample_name
    for fname in ["input_damaged.obj", "input_pointcloud.ply"]:
        src_file = source_sample_dir / fname
        if src_file.exists():
            shutil.copy2(str(src_file), str(dest_dir / fname))

    print(f"  Moved to: {dest_dir}")


def load_progress() -> dict:
    """加载进度文件"""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "current_index": 0}


def save_progress(progress: dict):
    """保存进度"""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="QuadRemesher batch pipeline")
    parser.add_argument("--test", type=int, default=0, help="Test mode: process N samples only")
    parser.add_argument("--dryrun", action="store_true", help="Dry run: don't actually process")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "cnm", "other"],
                        help="Process only: all / cnm (ClothesNetM) / other")
    args = parser.parse_args()

    # 创建临时目录
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 验证环境
    if not args.dryrun:
        for path, name in [(BLENDER, "Blender"), (ENGINE, "QuadRemesher engine")]:
            if not Path(path).exists():
                print(f"[FATAL] {name} not found: {path}")
                sys.exit(1)
        print(f"[OK] Blender: {BLENDER}")
        print(f"[OK] Engine: {ENGINE}")

    # 收集样本
    all_samples = collect_all_samples()
    print(f"\n[DATA] Total excluded samples: {len(all_samples)}")

    # 按来源过滤
    if args.source == "cnm":
        all_samples = [(r, d, n) for r, d, n in all_samples
                       if "other_excluded" not in str(r)]
    elif args.source == "other":
        all_samples = [(r, d, n) for r, d, n in all_samples
                       if "other_excluded" in str(r)]
    print(f"[DATA] After filter ({args.source}): {len(all_samples)}")

    # 加载进度
    progress = load_progress()
    completed_names = set(progress["completed"] + progress["failed"])

    # 测试模式
    if args.test > 0:
        all_samples = [(r, d, n) for r, d, n in all_samples
                       if n not in completed_names][:args.test]
        print(f"[TEST] Test mode: {len(all_samples)} samples")

    # 跳过已处理
    if not args.test and args.resume:
        pending = [(r, d, n) for r, d, n in all_samples if n not in completed_names]
        print(f"[RESUME] Resume: {len(pending)} remaining (skipping {len(completed_names)} done)")
    else:
        pending = [(r, d, n) for r, d, n in all_samples if n not in completed_names]

    if not pending:
        print("[OK] All samples already processed!")
        return

    # 统计面数分配
    target_counts = {}
    for _, _, name in pending:
        target_counts[classify_target_faces(name)] = target_counts.get(classify_target_faces(name), 0) + 1
    print(f"[STAT] Target distribution: {target_counts}")

    # 逐个处理
    results = {"success": [], "partial": [], "failed": [], "error": []}
    start_time = time.time()

    for i, (source_root, sample_dir, sample_name) in enumerate(pending):
        elapsed = time.time() - start_time
        remaining = (elapsed / max(i, 1)) * (len(pending) - i) if i > 0 else 0
        print(f"\n{'-'*60}")
        print(f"[{i+1}/{len(pending)}] {sample_name}  (elapsed: {elapsed/60:.1f}m, eta: {remaining/60:.1f}m)")

        target = classify_target_faces(sample_name)

        try:
            result = process_one_sample(source_root, sample_dir, sample_name,
                                        target, args.dryrun)
        except Exception as e:
            import traceback
            print(f"  [EXCEPTION] {e}")
            traceback.print_exc()
            result = None

        if result is None:
            results["error"].append(sample_name)
            progress["failed"].append(sample_name)
            save_progress(progress)
        elif result.get("dryrun"):
            pass
        elif result["status"] == "success":
            results["success"].append(sample_name)
            progress["completed"].append(sample_name)
            # 移入训练数据
            if not args.dryrun:
                move_to_supervised_data(sample_name, str(source_root),
                                        TEMP_DIR / sample_name / f"{sample_name}_retopo.obj")
            save_progress(progress)
        elif result["status"] == "partial":
            results["partial"].append(sample_name)
            progress["completed"].append(sample_name)
            if not args.dryrun:
                move_to_supervised_data(sample_name, str(source_root),
                                        TEMP_DIR / sample_name / f"{sample_name}_retopo.obj")
            save_progress(progress)
        else:
            results["failed"].append(sample_name)
            progress["failed"].append(sample_name)
            save_progress(progress)

        progress["current_index"] = i + 1
        save_progress(progress)

    # 最终报告
    total = len(pending)
    print(f"\n{'='*60}")
    print(f"[DONE] BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  [OK] Success (>=95% quads):  {len(results['success'])}/{total}")
    print(f"  [WARN]  Partial (80-95% quads): {len(results['partial'])}/{total}")
    print(f"  [FAIL] Failed (still tris):     {len(results['failed'])}/{total}")
    print(f"  [CRASH] Error:                   {len(results['error'])}/{total}")
    print(f"  Total time: {((time.time() - start_time) / 60):.1f} minutes")

    if results["success"]:
        recovered = len(results["success"]) + len(results["partial"])
        print(f"\n[WIN] Recovered {recovered} samples → moved to supervised_data/")


if __name__ == "__main__":
    main()
