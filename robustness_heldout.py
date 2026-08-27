"""Re-run the held-out causal test under the collapsed (shared-instrument) panel."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, contextlib, importlib
import activation_panel as ap

DROP = {("Argentina",2018,"T"),("Chile",2023,"L"),("Ghana",2023,"L"),
        ("Namibia",2023,"L"),("Zimbabwe",2022,"L")}
BASE = list(ap.EVENTS)
COLLAPSED = [e for e in BASE if (e[0],e[1],e[2]) not in DROP]

def run_heldout(events, tag):
    ap.EVENTS = events
    # exec the validation script fresh so its functions read the patched ap.EVENTS
    cvm = _os.path.join(_HERE, "causal_validation_multiyear.py")
    g = {"__file__": cvm, "__name__": "cvm_exec"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(cvm).read(), cvm, "exec"), g)
    out = buf.getvalue()
    # pull the summary lines
    keep = [ln for ln in out.splitlines()
            if any(k in ln for k in ["beats base", "beats placebo", "log-score",
                                     "CRITERION", "PASS", "FAIL", "MULTI", "per event",
                                     "Chile", "Argentina 2019", "nats"])]
    print(f"\n===== HELD-OUT under {tag} ({len(events)} events) =====")
    for ln in keep[:30]:
        print("  " + ln.strip())

run_heldout(BASE, "BASELINE panel")
run_heldout(COLLAPSED, "COLLAPSED panel")
ap.EVENTS = BASE
