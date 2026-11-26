import os
import sys
import traceback
from filelock import FileLock, Timeout

from .engine import Halt, run
from .experiment import Experiment
from .statistics import report
from .storage import ROOT, folder, now, read, save


def update(root, **changes):
    with FileLock(str(root / "state.lock")):
        state = read(root / "state.json")
        state.update(**changes)
        save(root / "state.json", state)


def execute(key):
    root = folder("runs", key)
    try:
        with FileLock(str(ROOT / "training.lock"), timeout=0):
            with FileLock(str(root / "state.lock")):
                state = read(root / "state.json")
                if state["status"] != "queued":
                    return
                state.update(
                    status="running",
                    pid=os.getpid(),
                    started=state.get("started") or now(),
                )
                save(root / "state.json", state)

            def halt():
                st = read(root / "state.json")["status"]
                if st in ("pausing", "cancelling"):
                    raise Halt(st)

            try:
                c = Experiment(**read(root / "config.json"))
                run(
                    root,
                    c,
                    read(root / "plan.json"),
                    lambda p: save(root / "progress.json", p),
                    halt,
                )
                halt()
                save(root / "statistics.json", report(root, c))
                update(root, status="completed", ended=now())
            except Halt as e:
                update(
                    root,
                    status="paused" if str(e) == "pausing" else "cancelled",
                    ended=now(),
                )
            except Exception as e:
                traceback.print_exc()
                update(root, status="failed", error=str(e), ended=now())
    except Timeout:
        pass


if __name__ == "__main__":
    execute(sys.argv[1])
