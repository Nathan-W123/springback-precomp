"""Worker: run the QUEUE of variants.py, claiming jobs by atomic mkdir.
Usage: python3 run_queue.py <worker-name>"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proclens as P   # noqa: E402
import variants as V   # noqa: E402

me = sys.argv[1]
log = open(HERE / f"worker_{me}.log", "a", buffering=1)


def done(part, name):
    d = P.RUNS / part / name
    return (d / "COMPLETE").exists() or (d / "FAILED").exists()


while True:
    import importlib
    importlib.reload(P)
    importlib.reload(V)
    todo = [(p, n, s, dep) for p, n, s, dep in V.QUEUE if not done(p, n)
            and not (P.RUNS / p / f"{n}.claim2").exists()]
    ready = [j for j in todo if j[3] is None or done(j[0], j[3])]
    if not ready:
        if not todo:
            break
        time.sleep(30)
        continue
    part, name, spec, _ = ready[0]
    claim = P.RUNS / part / f"{name}.claim2"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
    except FileExistsError:
        continue
    t0 = time.time()
    log.write(f"{time.strftime('%H:%M:%S')} {me} start {part} {name} load {os.getloadavg()}\n")
    rec = P.run_variant(part, name, spec)
    log.write(f"{time.strftime('%H:%M:%S')} {me} end {part} {name} ok={rec.get('ok')} "
              f"wall={time.time() - t0:.0f}s {rec.get('error', '')[:300]}\n")
log.write(f"{time.strftime('%H:%M:%S')} {me} queue empty\n")
