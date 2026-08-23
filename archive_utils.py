"""
archive_utils.py
================
Extract LAS files from uploaded archives.

Directory structure inside the archive is discarded and every file is written
to one flat directory under its own base name. Archives produced on Windows
often carry an enclosing folder whose name is awkward to handle, and nothing
downstream needs the structure.

Run directly to inspect an archive without extracting it:

    python archive_utils.py /path/to/survey.rar
"""

import os
import re
import shutil
import zipfile

LAS_EXT = (".las", ".laz")


def _is_las(name):
    base = os.path.basename(name)
    return (
        base.lower().endswith(LAS_EXT)
        and not base.startswith(".")  # resource forks
        and "__MACOSX" not in name
    )


def _unique_path(directory, base):
    """Avoid collisions when two archive folders hold the same file name."""
    target = os.path.join(directory, base)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(base)
    n = 1
    while os.path.exists(os.path.join(directory, f"{stem}_{n}{ext}")):
        n += 1
    return os.path.join(directory, f"{stem}_{n}{ext}")


def rar_backend():
    """Report whether RAR extraction is possible, and via what.

    rarfile shells out to an external tool. Without one installed it raises
    only when an archive is opened, which is late and unclear, so this is
    checked up front.
    """
    try:
        import rarfile
    except ImportError:
        return None, "the rarfile package is not installed"
    for tool in ("unar", "unrar", "bsdtar", "7z"):
        if shutil.which(tool):
            return tool, None
    return None, (
        "no RAR tool found on the system (looked for unar, unrar, bsdtar, 7z)"
    )


def list_archive(path):
    """Names held in an archive, without extracting anything."""
    low = path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    if low.endswith(".rar"):
        tool, why = rar_backend()
        if tool is None:
            raise RuntimeError(f"Cannot read RAR archives: {why}")
        import rarfile

        with rarfile.RarFile(path) as rf:
            return rf.namelist()
    raise ValueError(f"Not an archive: {path}")


def extract_archives(paths, workdir):
    """Expand any archives among `paths` and return a flat list of LAS files.

    Files that are already LAS pass through untouched. Everything extracted
    lands directly in `workdir/unpacked` regardless of its location inside
    the archive.

    Returns
    -------
    las_paths : list of str
    log : list of str, one line per input, suitable for showing to the user
    """
    outdir = os.path.join(workdir, "unpacked")
    las_paths, log = [], []

    for p in paths:
        low = p.lower()
        name = os.path.basename(p)

        if low.endswith(LAS_EXT):
            las_paths.append(p)
            continue

        if low.endswith(".zip"):
            os.makedirs(outdir, exist_ok=True)
            found = 0
            with zipfile.ZipFile(p) as zf:
                for member in zf.namelist():
                    if not _is_las(member):
                        continue
                    target = _unique_path(outdir, os.path.basename(member))
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    las_paths.append(target)
                    found += 1
            log.append(f"{name}: extracted {found} LAS file(s)")
            if found == 0:
                log.append(
                    f"  archive contents: "
                    f"{', '.join(zipfile.ZipFile(p).namelist()[:10])}"
                )

        elif low.endswith(".rar"):
            tool, why = rar_backend()
            if tool is None:
                log.append(
                    f"{name}: cannot open RAR archives here ({why}). "
                    f"Please re-compress as ZIP: select the files in Windows "
                    f"Explorer, right-click, and choose "
                    f"Send to > Compressed (zipped) folder."
                )
                continue
            try:
                import rarfile

                os.makedirs(outdir, exist_ok=True)
                found = 0
                with rarfile.RarFile(p) as rf:
                    for member in rf.namelist():
                        if not _is_las(member):
                            continue
                        target = _unique_path(outdir, os.path.basename(member))
                        with rf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        las_paths.append(target)
                        found += 1
                log.append(f"{name}: extracted {found} LAS file(s) via {tool}")
                if found == 0:
                    with rarfile.RarFile(p) as rf:
                        inside = rf.namelist()[:10]
                    log.append(f"  archive contents: {', '.join(inside)}")
            except Exception as e:
                log.append(
                    f"{name}: RAR extraction failed ({type(e).__name__}: {e}). "
                    f"Please re-compress as ZIP."
                )

        else:
            log.append(f"{name}: unrecognised file type, ignored")

    if las_paths:
        log.append(f"{len(las_paths)} LAS file(s) ready")
    return las_paths, log


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(
            'usage: python archive_utils.py "<archive>"\n'
            "        quote the path if it contains spaces"
        )

    # An unquoted path with spaces arrives split across several arguments.
    # Try the whole thing joined back together before giving up, since
    # survey archives regularly have spaces in both the folder and the file
    # name.
    joined = " ".join(sys.argv[1:])
    if os.path.exists(joined):
        path = joined
    elif os.path.exists(sys.argv[1]):
        path = sys.argv[1]
    else:
        print("Could not find the archive.")
        print(f"  received {len(sys.argv) - 1} argument(s): {sys.argv[1:]!r}")
        print(f"  tried {joined!r}")
        if len(sys.argv) > 2:
            print("\nThe path was split on spaces, so it was probably not quoted. Try:")
            print(f'  python archive_utils.py "{joined}"')
        sys.exit(1)

    print(f"Archive: {path!r}\n")

    tool, why = rar_backend()
    print(f"RAR backend: {tool or 'unavailable'}" + (f"  ({why})" if why else ""))

    try:
        members = list_archive(path)
    except Exception as e:
        sys.exit(f"Could not read the archive: {type(e).__name__}: {e}")

    las = [m for m in members if _is_las(m)]
    print(f"\n{len(members)} entries, {len(las)} of them LAS")
    print("\nfirst 15 entries, with repr() so odd names are visible:")
    for m in members[:15]:
        print(f"  {'LAS' if _is_las(m) else '   '}  {m!r}")
    if not las:
        print(
            "\nNo LAS files matched. Check the names printed above for "
            "unexpected extensions or nesting."
        )
