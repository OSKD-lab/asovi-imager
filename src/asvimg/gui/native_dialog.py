"""System-native "open file" dialog, run in a subprocess.

Dear PyGui ships its own in-window file dialog, but the user asked for the
*OS-standard* picker.  Rather than pull in a second GUI toolkit (tkinter mixed
with dearpygui is fragile, especially on macOS), we shell out to the platform's
native chooser and read the selected path from stdout:

* macOS   -> ``osascript`` ``choose file`` (Cocoa dialog)
* Windows -> PowerShell ``System.Windows.Forms.OpenFileDialog``
* Linux   -> ``zenity`` if present (best-effort; not required by the task)

Returns the chosen path, or ``None`` if the user cancelled or no native
chooser is available.  The call blocks (the dialog is modal) but runs in its
own process, so it never fights the dearpygui render loop for the main thread.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def open_file(
    title: str = "Select a file",
    extensions: tuple[str, ...] = ("yaml", "yml"),
    default_dir: str | None = None,
) -> str | None:
    """Open the OS-native file picker; return the selected path or None."""
    try:
        if sys.platform == "darwin":
            return _macos(title, extensions, default_dir)
        if sys.platform.startswith("win"):
            return _windows(title, extensions, default_dir)
        return _linux(title, extensions, default_dir)
    except FileNotFoundError:
        return None  # the helper (osascript / powershell / zenity) is missing
    except Exception:  # noqa: BLE001 — a dialog failure must never crash the app
        return None


def _macos(title, extensions, default_dir) -> str | None:
    prompt = title.replace('"', "'")
    loc = ""
    if default_dir and Path(default_dir).is_dir():
        loc = f'default location (POSIX file "{default_dir}") '
    of_type = ""
    if extensions:
        joined = '","'.join(extensions)
        of_type = f'of type {{"{joined}"}} '
    script = (
        f'POSIX path of (choose file with prompt "{prompt}" {loc}{of_type})'
    )
    proc = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True
    )
    if proc.returncode != 0:  # user cancelled (-128) or error
        return None
    path = proc.stdout.strip()
    return path or None


def _windows(title, extensions, default_dir) -> str | None:
    if extensions:
        globs = ";".join(f"*.{e}" for e in extensions)
        filt = f"YAML ({globs})|{globs}|All files (*.*)|*.*"
    else:
        filt = "All files (*.*)|*.*"
    init = ""
    if default_dir and Path(default_dir).is_dir():
        init = f"$f.InitialDirectory = '{default_dir}'; "
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$f = New-Object System.Windows.Forms.OpenFileDialog; "
        f"$f.Title = '{title}'; $f.Filter = '{filt}'; {init}"
        "if ($f.ShowDialog() -eq 'OK') { $f.FileName }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    path = proc.stdout.strip()
    return path or None


def _linux(title, extensions, default_dir) -> str | None:
    cmd = ["zenity", "--file-selection", f"--title={title}"]
    if default_dir and Path(default_dir).is_dir():
        cmd.append(f"--filename={default_dir}/")
    if extensions:
        pattern = " ".join(f"*.{e}" for e in extensions)
        cmd.append(f"--file-filter=YAML | {pattern}")
        cmd.append("--file-filter=All files | *")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    path = proc.stdout.strip()
    return path or None


def open_in_file_manager(path) -> bool:
    """Reveal ``path`` in the OS file manager (Finder / Explorer / xdg). Returns
    True if the folder exists and a handler was launched."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(p)])
        elif sys.platform.startswith("win"):
            import os

            os.startfile(str(p))  # type: ignore[attr-defined]  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(p)])
        return True
    except Exception:  # noqa: BLE001
        return False


def open_folder(title: str = "Select a folder", default_dir: str | None = None) -> str | None:
    """Open the OS-native folder picker; return the selected directory or None."""
    try:
        if sys.platform == "darwin":
            return _macos_folder(title, default_dir)
        if sys.platform.startswith("win"):
            return _windows_folder(title)
        return _linux_folder(title, default_dir)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a dialog failure must never crash the app
        return None


def _macos_folder(title, default_dir) -> str | None:
    prompt = title.replace('"', "'")
    loc = ""
    if default_dir and Path(default_dir).is_dir():
        loc = f'default location (POSIX file "{default_dir}") '
    script = f'POSIX path of (choose folder with prompt "{prompt}" {loc})'
    proc = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True
    )
    if proc.returncode != 0:  # user cancelled
        return None
    return proc.stdout.strip() or None


def _windows_folder(title) -> str | None:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
        f"$f.Description = '{title}'; "
        "if ($f.ShowDialog() -eq 'OK') { $f.SelectedPath }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _linux_folder(title, default_dir) -> str | None:
    cmd = ["zenity", "--file-selection", "--directory", f"--title={title}"]
    if default_dir and Path(default_dir).is_dir():
        cmd.append(f"--filename={default_dir}/")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None
