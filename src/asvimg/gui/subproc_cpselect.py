"""Child-process entry: control-point selection in its own dpg context.

Invoked as ``python -m asvimg.gui.subproc_cpselect <req> <res>``.
Reads a pickled request (source images, atlas RGB, boundaries, labels), opens
the standalone ``cpselect_gui`` (which creates and destroys its own dearpygui
context), and writes the pickled ``(src_pts, ref_pts)`` result — or ``None``
if the user cancelled.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: subproc_cpselect <request.pkl> <result.pkl>", file=sys.stderr)
        return 2
    req_path, res_path = argv

    from asvimg.cpselect import cpselect_gui
    from asvimg.gui import ipc

    req = ipc.load(req_path)
    result = cpselect_gui(
        req["source_imgs"],
        req["ref_rgb"],
        n_default_pairs=req.get("n_default_pairs", 2),
        boundaries=req.get("boundaries"),
        source_labels=req.get("labels"),
        title=req.get("title", "Control Point Selection"),
    )
    # result is (src_pts, ref_pts) or None
    ipc.dump(result, res_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
