"""
train_all.py
تدريب policy q-learning_2 على جميع الخرائط دفعة واحدة.
الاستخدام:
    python train_all.py
    python train_all.py --maps 0 1 2 13     ← خرائط محددة فقط
    python train_all.py --skip 20 21 22     ← تخطي خرائط كبيرة
"""

import argparse
import os
import subprocess
import sys
import time

ALL_MAPS = list(range(23))   # 0 → 22

MAP_NAMES = {
    0:  "vacuum-2rooms",
    1:  "vacuum-3rooms-v0",
    2:  "vacuum-3rooms-v1",
    3:  "vacuum-3rooms-v2",
    4:  "vacuum-4rooms-v0",
    5:  "vacuum-4rooms-v1",
    6:  "vacuum-4rooms-v2",
    7:  "vacuum-4rooms-v3",
    8:  "vacuum-5rooms-v0",
    9:  "vacuum-5rooms-v1",
    10: "vacuum-5rooms-v2",
    11: "vacuum-5rooms-v3",
    12: "vacuum-5rooms-v4",
    13: "vacuum-6rooms-v0",
    14: "vacuum-6rooms-v1",
    15: "vacuum-6rooms-v2",
    16: "vacuum-7rooms-v0",
    17: "vacuum-8rooms-v0",
    18: "vacuum-8rooms-v1",
    19: "vacuum-9rooms-v0",
    20: "vacuum-12rooms-v0",
    21: "vacuum-5x5-v0",
    22: "vacuum-6x6-v0",
}

POLICY        = 4       # q-learning_2
DATA_DIR      = "data"  # مجلد Q-tables


def qtable_exists(map_id: int) -> bool:
    """هل Q-table لهذه الخريطة موجودة مسبقاً؟"""
    world_name = MAP_NAMES.get(map_id, f"map-{map_id}")
    path = os.path.join(DATA_DIR, f"qlearning_table_map_{world_name}_v2.pkl")
    return os.path.exists(path)


def train_map(map_id: int, force_retrain: bool = False) -> bool:
    name = MAP_NAMES.get(map_id, f"map-{map_id}")
    print(f"\n{'='*60}")
    print(f"  Training map {map_id:>2}: {name}")
    print(f"{'='*60}")

    already_trained = qtable_exists(map_id)

    if already_trained and not force_retrain:
        print(f"  ⏭️  Q-table already exists — skipping training.")
        print(f"      (use --retrain to force retraining)")
        return True

    # ترتيب الأسئلة:
    # حالة 1 - لا توجد Q-table:
    #   [1] Change episodes/steps? → Enter (إبقاء الافتراضي)
    #   [2] Render training?       → Enter (لا)
    #   [3] Press Enter heatmap    → Enter
    #   [4] Press Enter simulation → Enter
    #
    # حالة 2 - Q-table موجودة:
    #   [1] Retrain? [y/n]         → y  (نعم نعيد)
    #   [2] Change episodes/steps? → Enter
    #   [3] Render training?       → Enter
    #   [4] Press Enter heatmap    → Enter
    #   [5] Press Enter simulation → Enter

    if already_trained and force_retrain:
        inputs = "y\n\n\n\n\n"
    else:
        inputs = "\n\n\n\n\n"   # Enter لكل سؤال

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "main.py", str(map_id), str(POLICY), "-cl"],
        input=inputs,
        text=True,
    )
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = "✅ Done" if ok else "❌ Failed"
    print(f"\n{status} — map {map_id} ({name}) in {elapsed:.1f}s")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning on all maps")
    parser.add_argument("--maps", nargs="+", type=int,
                        help="خرائط محددة للتدريب (default: all)")
    parser.add_argument("--skip", nargs="+", type=int, default=[],
                        help="خرائط تريدين تخطيها")
    parser.add_argument("--retrain", action="store_true",
                        help="أعيدي التدريب حتى لو Q-table موجودة")
    args = parser.parse_args()

    maps = args.maps if args.maps else ALL_MAPS
    maps = [m for m in maps if m not in args.skip]

    print(f"🚀 Training {len(maps)} maps: {maps}")
    if not args.retrain:
        print("   (خرائط تملك Q-table ستُتخطى — استخدمي --retrain لإعادة التدريب)")
    t_start = time.time()

    results = {}
    for map_id in maps:
        results[map_id] = train_map(map_id, force_retrain=args.retrain)

    # ── تقرير نهائي ──
    total = time.time() - t_start
    done  = [m for m, ok in results.items() if ok]
    fail  = [m for m, ok in results.items() if not ok]

    print(f"\n{'='*60}")
    print(f"  Training complete in {total/60:.1f} min")
    print(f"  ✅ Success ({len(done)}): {done}")
    if fail:
        print(f"  ❌ Failed  ({len(fail)}): {fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()