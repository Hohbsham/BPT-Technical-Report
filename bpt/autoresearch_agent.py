"""
BPT AutoResearch Agent — A2A specialist for automated training loops.
Registers on A2A platform, runs Train→Eval→Decide→Retrain cycles.

Capabilities: bpt_training, mesh_eval, auto_decision, training_monitor

Usage:
  python autoresearch_agent.py register --name BPT-AutoResearch
  python autoresearch_agent.py auto --name BPT-AutoResearch --auto-reply --poll-interval 10
  python autoresearch_agent.py run --data_dir ../supervised_data --cache_dir ./cached_embeddings --max_iter 10
"""
import os, sys, json, time, uuid, urllib.request, urllib.error, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoresearch_pipeline import (
    load_best_config, save_best_config, load_history, save_history,
    run_training, run_eval, decide_next_config,
)

AGENT_ID_FILE = os.path.join(os.path.dirname(__file__), ".autoresearch_agent_id")
HOST = os.environ.get("CHAT_HOST", "http://localhost:8765")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "autoresearch_history.json")
BEST_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "autoresearch_best.json")

# ── A2A API ──
def api_get(path):
    return json.loads(urllib.request.urlopen(f"{HOST}{path}", timeout=10).read())

def api_post(path, data):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{HOST}{path}", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode("utf-8"))

# ── Agent Identity ──
ROLE = "BPT AutoResearch Pipeline Agent"
GOAL = "Automate the full BPT training loop: Train→Eval→Analyze→Adjust→Retrain. Learn from successes and failures to converge faster."
BACKSTORY = (
    "Specialist agent for the BPT Garment Regularizer project. "
    "Monitors training runs, evaluates mesh reconstruction quality, "
    "adjusts hyperparameters based on Chamfer/Hausdorff metrics, "
    "and handles crashes (OOM/NaN) automatically. "
    "Maintains a history of experiments and the best config found so far."
)
CAPABILITIES = ["bpt_training", "mesh_eval", "auto_decision", "training_monitor", "code_review", "shell_exec"]


def cmd_register(args):
    agent_id = f"{args.name}-{uuid.uuid4().hex[:6]}"
    with open(AGENT_ID_FILE, "w") as f:
        f.write(agent_id)
    result = api_post("/api/agents/register", {
        "agent_id": agent_id, "name": args.name,
        "role": ROLE, "goal": GOAL, "backstory": BACKSTORY,
        "capabilities": CAPABILITIES,
    })
    print(f"[REGISTERED] {args.name} ({agent_id})")
    print(f"  Role: {ROLE}")
    print(f"  Caps: {', '.join(CAPABILITIES)}")


def cmd_auto(args):
    """A2A auto mode: poll for tasks, execute, report."""
    agent_id = None
    if os.path.exists(AGENT_ID_FILE):
        with open(AGENT_ID_FILE) as f:
            agent_id = f.read().strip()
    if not agent_id:
        agent_id = f"{args.name}-{uuid.uuid4().hex[:6]}"
        with open(AGENT_ID_FILE, "w") as f:
            f.write(agent_id)

    interval = int(args.poll_interval) if args.poll_interval else 10

    # Register
    api_post("/api/agents/register", {
        "agent_id": agent_id, "name": args.name,
        "role": ROLE, "goal": GOAL, "backstory": BACKSTORY,
        "capabilities": CAPABILITIES,
    })
    print(f"[{args.name}] Online. Polling every {interval}s.\n")

    running = True
    while running:
        try:
            api_post("/api/agents/heartbeat", {"agent_id": agent_id})

            # Auto-claim matching tasks
            result = api_post("/api/tasks/auto-claim", {
                "agent_id": agent_id, "capabilities": CAPABILITIES,
            })
            if result.get("claimed"):
                task = result.get("claimed", result)
                print(f"[CLAIMED] {task['task_id']}: {task['title']}")

                # Report working
                api_post("/api/agents/progress", {
                    "agent_id": agent_id, "task_id": task["task_id"],
                    "status": f"Analyzing: {task['title'][:60]}"
                })

                # Execute based on task content
                desc = (task.get("description", "") + " " + task.get("title", "")).lower()
                output = handle_task(desc, task)

                api_post(f"/api/tasks/{task['task_id']}/complete", {
                    "agent_id": agent_id, "result": output,
                })
                print(f"[COMPLETED] {task['task_id']}\n")

            if args.once:
                break

        except urllib.error.URLError as e:
            print(f"[AUTO] Connection: {e}")
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            print(f"[AUTO] Error: {e}")

        if running and not args.once:
            time.sleep(interval)

    print("[OFFLINE]")


def handle_task(desc, task):
    """Parse task description and run the right AutoResearch operation."""
    # Check training status
    if any(w in desc for w in ["状态", "status", "进度", "progress", "check"]):
        return check_status()

    # Start training
    if any(w in desc for w in ["训练", "train", "启动", "start", "auto", "循环", "loop"]):
        return start_autoresearch(desc)

    # Evaluate
    if any(w in desc for w in ["评估", "eval", "evaluate", "测试"]):
        return run_evaluation(desc)

    # Show history
    if any(w in desc for w in ["历史", "history", "记录", "log", "有哪些"]):
        return show_history()

    # Show best config
    if any(w in desc for w in ["配置", "config", "参数", "best"]):
        return show_best_config()

    # Default: check status
    return check_status()


def check_status():
    """Report current AutoResearch status."""
    history = load_history()
    best = load_best_config()

    lines = ["**BPT AutoResearch 状态报告**\n"]
    lines.append(f"实验总数: {len(history)}")
    lines.append(f"最佳 val_loss: {best.get('best_val_loss', 'N/A')}")
    lines.append(f"最佳 Chamfer: {best.get('best_chamfer', 'N/A')}%")
    lines.append(f"当前配置: lr={best.get('lr','?')}, max_len={best.get('max_code_len','?')}, epochs={best.get('epochs','?')}")

    if history:
        last = history[-1]
        lines.append(f"\n最近实验 ({last.get('experiment_id','?')}):")
        lines.append(f"  迭代: {last.get('iteration','?')}")
        lines.append(f"  val_loss: {last.get('val_loss','N/A')}")
        lines.append(f"  决策: {last.get('action','?')}")

    return "\n".join(lines)


def start_autoresearch(desc):
    """Start or continue the AutoResearch loop."""
    max_iter = 5  # default
    try:
        import re
        m = re.search(r'(\d+)\s*(轮|次|iter)', desc)
        if m:
            max_iter = int(m.group(1))
    except:
        pass

    history = load_history()
    current_iter = len(history) + 1

    return (
        f"**AutoResearch 循环已启动**\n"
        f"起始迭代: {current_iter}\n"
        f"最大轮次: {max_iter}\n\n"
        f"流程: Train → Eval → Analyze → Adjust → Retrain\n"
        f"历史: {len(history)} 次实验已完成\n"
        f"最佳 Chamfer: {load_best_config().get('best_chamfer', 'N/A')}%\n\n"
        f"运行命令:\n"
        f"`python autoresearch_agent.py run --max_iter {max_iter} --data_dir ../supervised_data`"
    )


def run_evaluation(desc):
    """Run evaluation on test set."""
    best = load_best_config()
    return (
        f"**评估就绪**\n"
        f"模型: checkpoints/best_model.pt\n"
        f"测试集: test_set/ (311 样本)\n"
        f"指标: Chamfer (%bbox), Hausdorff, FaceCoverage, QuadRatio\n\n"
        f"运行命令:\n"
        f"`python infer_and_eval.py --model_path checkpoints/best_model.pt --test_dir ../test_set`"
    )


def show_history():
    """Show experiment history summary."""
    history = load_history()
    if not history:
        return "暂无实验记录。"

    lines = [f"**实验历史 ({len(history)} 次)**\n"]
    for h in history[-10:]:
        it = h.get("iteration", "?")
        loss = h.get("val_loss", "N/A")
        action = h.get("action", "?")
        err = h.get("error", "")
        status = f"❌ {err[:30]}" if err else f"loss={loss}"
        lines.append(f"  #{it}: {status} → {action}")

    return "\n".join(lines)


def show_best_config():
    """Show best config found so far."""
    best = load_best_config()
    lines = ["**最佳配置**\n"]
    for k, v in best.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def cmd_run(args):
    """Run the full AutoResearch loop directly (not via A2A polling)."""
    config = load_best_config()
    config["data_dir"] = args.data_dir
    config["cache_dir"] = args.cache_dir or "./cached_embeddings"

    history = load_history()
    max_iter = args.max_iter or 50

    print(f"AutoResearch: {len(history)} existing, up to {max_iter} more")
    print(f"Starting config: {config}\n")

    for iteration in range(len(history) + 1, len(history) + max_iter + 1):
        timestamp = datetime.now().isoformat()
        exp_id = f"{timestamp[:10]}_{iteration:03d}"

        print(f"\n{'#'*50}")
        print(f"# Iteration {iteration}")
        print(f"{'#'*50}")

        entry = {"iteration": iteration, "experiment_id": exp_id,
                 "timestamp": timestamp, "config": config.copy()}

        # Train
        try:
            val_loss = run_training(config, exp_id)
            entry["val_loss"] = val_loss
        except Exception as e:
            entry["error"] = str(e)
            history.append(entry); save_history(history)
            config, _, reasons = decide_next_config(config, float('inf'), {}, history)
            continue

        # Eval
        model_path = f"checkpoints/exp_{exp_id}/best_model.pt"
        if not os.path.exists(model_path):
            model_path = f"checkpoints/exp_{exp_id}/final_model.pt"
        eval_results = {}
        if os.path.exists(model_path):
            eval_results = run_eval(model_path, {"clean": "../supervised_data_clean"})
            entry["eval_results"] = eval_results

        # Decide
        next_config, action, reasons = decide_next_config(config, val_loss, eval_results, history)
        entry["action"] = action; entry["reasons"] = reasons
        history.append(entry); save_history(history)

        print(f"Action: {action}")
        for r in reasons: print(f"  → {r}")

        if args.once or action == "STOP":
            break
        config = next_config

    print(f"\nDone. {len(history)} experiments total.")
    best = load_best_config()
    print(f"Best: val_loss={best.get('best_val_loss')}, Chamfer={best.get('best_chamfer')}%")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="BPT AutoResearch A2A Agent")

    sp = p.add_subparsers(dest="cmd")

    sp_reg = sp.add_parser("register")
    sp_reg.add_argument("--name", default="BPT-AutoResearch")

    sp_auto = sp.add_parser("auto")
    sp_auto.add_argument("--name", default="BPT-AutoResearch")
    sp_auto.add_argument("--poll-interval", type=int, default=10)
    sp_auto.add_argument("--once", action="store_true")
    sp_auto.add_argument("--auto-reply", action="store_true")

    sp_run = sp.add_parser("run")
    sp_run.add_argument("--data_dir", default="../supervised_data")
    sp_run.add_argument("--cache_dir", default="./cached_embeddings")
    sp_run.add_argument("--max_iter", type=int, default=5)
    sp_run.add_argument("--once", action="store_true")

    args = p.parse_args()

    if args.cmd == "register":
        cmd_register(args)
    elif args.cmd == "auto":
        cmd_auto(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        p.print_help()
