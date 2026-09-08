"""Child-process entry: ICA component exclusion in its own dearpygui context.

Invoked as ``python -m asvimg.gui.subproc_ica <req> <res>``.  Reads a
pickled request (the ``IcaResult`` for ONE channel group, plus the window title
and the exclusion already recorded for it), opens the standalone dearpygui
``ica_select_gui`` (which creates and destroys its own context — it cannot share
the dashboard's), and writes the pickled excluded-index list — or ``None`` if
cancelled.

``None`` is load-bearing: it means "no answer", which is not the same as "exclude
nothing", and the caller must not confuse the two.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: subproc_ica <request.pkl> <result.pkl>", file=sys.stderr)
        return 2
    req_path, res_path = argv

    from asvimg.ica_gui import ica_select_gui
    from asvimg.gui import ipc

    req = ipc.load(req_path)
    if isinstance(req, dict):  # {"ica_result", "title", "preselected"}
        excluded = ica_select_gui(
            req["ica_result"],
            title=req.get("title") or "ICA Component Selection - Click to EXCLUDE",
            preselected=req.get("preselected"),
        )
    else:  # legacy: a bare IcaResult
        excluded = ica_select_gui(req)
    ipc.dump(excluded, res_path)  # list[int], or None if cancelled
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
