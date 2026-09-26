#!/usr/bin/env python3
"""build_libadalang.py - Build libadalang.so for the dataset AST parser.

data/processing_scripts/parse_ada_ast.py prefers libadalang over its
structural scanner, but the bindings are only half a library: the `ast`
dependency group installs the pure-Python ctypes wrapper, and that wrapper
dlopens a shared `libadalang.so` which no package manager ships. This script
produces that shared object and installs it next to the wrapper.

Everything Ada-related goes through Alire, never a system tool:

  1. alire-ast.toml is copied into the gitignored .alire-ast/ workspace and
     resolved there with `alr update` (alr 1.2.1 has no --manifest option, the
     same reason scripts/ada_env.sh keeps a .alire-dev/ workspace for the
     SPARK toolchain). The manifest is kept apart from alire-dev.toml on
     purpose: only the dataset parser needs libadalang.
  2. ast.gpr pulls the resolved crates in and is built twice inside the Alire
     environment. The Alire crate ships a static library, but the ctypes
     wrapper needs a shared one, so the stack is compiled position
     independent first and libadalang alone is then relinked as a shared
     object. Both passes reuse the same objects, so only the final link is
     repeated.
  3. GMP is the one system library in the chain (gnatcoll_gmp links -lgmp).
     q3as never installs system packages, so the headers are unpacked from
     the distribution package into the user cache and the linker is pointed
     at the shared GMP. The static libgmp.a from the distribution is not
     position independent and cannot go into a shared object at all.

Run it with `make ast-deps`. It is idempotent: an installed, importable
libadalang.so is left alone unless --force is given.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AST_MANIFEST = PROJECT_ROOT / "alire-ast.toml"
AST_GPR = PROJECT_ROOT / "ast.gpr"
AST_WS = PROJECT_ROOT / ".alire-ast"
TOOLSHIMS = PROJECT_ROOT / "scripts" / "toolshims"
GMP_CACHE = Path.home() / ".cache" / "q3as-gmp"

logger = logging.getLogger("build_libadalang")


class BuildFailed(RuntimeError):
    """A step of the libadalang build failed."""


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    """Run *cmd* in *cwd*, returning stdout, raising BuildFailed on failure."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
        raise BuildFailed(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail)
        )
    return proc.stdout


def alr_env(cwd: Path) -> dict[str, str]:
    """Return a PATH that puts the Alire toolchain and the tool shims first.

    Only used for the `alr` calls themselves. Build commands are never run
    with a reconstructed environment: see alr_run.
    """
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(TOOLSHIMS), env.get("PATH", "")]  # unzip shim for the source archives
    )
    return env


def alr_run(cmd: list[str], workspace: Path, extra_env: dict[str, str] | None = None) -> str:
    """Run a toolchain command through `alr exec` inside the AST workspace.

    `alr exec` does more than prepend bins to PATH: it exports
    GPR_PROJECT_PATH pointing at every resolved crate and the gpr-set-externals
    those crates declare (GPR2_BUILD=release, GNATCOLL_BUILD_MODE=PROD, ...).
    The crate project files depend on both, so the command is handed to
    `alr exec` rather than run against a rebuilt environment.
    """
    env = {**alr_env(workspace), **(extra_env or {})}
    return run(["alr", "exec", "--", *cmd], cwd=workspace, env=env)


def prepare_workspace() -> Path:
    """Create .alire-ast/ with a copy of the AST manifest and project."""
    if not AST_MANIFEST.exists():
        raise BuildFailed(f"{AST_MANIFEST} not found")
    AST_WS.mkdir(exist_ok=True)
    # Only rewrite on change, so alr does not re-resolve on every run.
    for src, dest in ((AST_MANIFEST, AST_WS / "alire.toml"), (AST_GPR, AST_WS / "ast.gpr")):
        if not dest.exists() or src.read_bytes() != dest.read_bytes():
            shutil.copy2(src, dest)
    return AST_WS


def resolve_dependencies(workspace: Path) -> None:
    """Deploy and build the crate sources declared in alire-ast.toml."""
    run(["alr", "-n", "update"], cwd=workspace, env=alr_env(workspace))


def ensure_gmp() -> dict[str, str]:
    """Make GMP headers and a position independent libgmp available.

    Returns the environment additions (CPATH, LIBRARY_PATH) for the build.
    Distributions ship GMP headers in a separate package that needs root to
    install, so the .deb is unpacked into the user cache instead. Linking
    must use the shared GMP: the distribution's libgmp.a is not built with
    -fPIC and the linker rejects it inside a shared object.
    """
    if not shutil.which("apt-get"):
        raise BuildFailed(
            "GMP headers are missing and apt-get is unavailable. Install libgmp-dev, "
            "or provide gmp.h and a shared libgmp on the default search path."
        )
    include = GMP_CACHE / "root/usr/include/x86_64-linux-gnu"
    if not (include / "gmp.h").exists():
        logger.info("unpacking libgmp-dev into %s (no sudo)", GMP_CACHE)
        GMP_CACHE.mkdir(parents=True, exist_ok=True)
        run(
            ["apt-get", "download", "libgmp-dev"],
            cwd=GMP_CACHE,
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
        )
        for deb in GMP_CACHE.glob("*.deb"):
            run(["dpkg-deb", "-x", str(deb), str(GMP_CACHE / "root")], cwd=GMP_CACHE)
            deb.unlink()
    if not (include / "gmp.h").exists():
        raise BuildFailed(f"gmp.h not found under {include} after unpacking libgmp-dev")

    # A link directory holding only a shared GMP, so -lgmp cannot pick up the
    # non-PIC static archive that ships in the same place as the headers.
    pic_dir = GMP_CACHE / "pic"
    pic_dir.mkdir(parents=True, exist_ok=True)
    link = pic_dir / "libgmp.so"
    if not link.exists():
        shared = next(
            (p for p in Path("/usr/lib").rglob("libgmp.so.*") if p.is_file()),
            None,
        )
        if shared is None:
            raise BuildFailed("no shared libgmp.so.* found under /usr/lib")
        link.symlink_to(shared)
    logger.info("GMP headers: %s", include)
    logger.info("GMP link dir: %s -> %s", pic_dir, os.readlink(link))
    return {"CPATH": str(include), "LIBRARY_PATH": str(pic_dir)}


def build_library(workspace: Path, gmp_env: dict[str, str]) -> Path:
    """Build the stack position independent, then link libadalang.so."""
    common = ["gprbuild", "-P", "ast.gpr", "-j0", "-XLIBRARY_TYPE=static-pic"]

    # libgpr2 is a C project whose .build/<mode> tree is never created for us;
    # gprbuild refuses to start until the directories exist. -p makes it
    # create the project directories first.
    alr_run(
        ["gprbuild", "-P", "ast.gpr", "-p", "-XLIBRARY_TYPE=static-pic"],
        workspace,
        gmp_env,
    )
    # Pass 1: the whole stack as static-pic archives.
    alr_run(common, workspace, gmp_env)
    # Pass 2: libadalang alone as a shared object, reusing pass 1's objects
    # (the object directory does not depend on the library kind).
    alr_run([*common, "-XLIBADALANG_LIBRARY_TYPE=relocatable"], workspace, gmp_env)

    built = sorted(workspace.glob("alire/cache/dependencies/libadalang_*/lib/*/*/libadalang.so"))
    if not built:
        raise BuildFailed("the build produced no libadalang.so")
    return built[-1]


def package_dir() -> Path:
    """Directory of the installed pure-Python libadalang wrapper."""
    candidates = [
        Path(sysconfig.get_paths()["purelib"]) / "libadalang",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages" / "libadalang",
    ]
    for candidate in candidates:
        if (candidate / "__init__.py").exists():
            return candidate
    raise BuildFailed(
        "the libadalang Python wrapper is not installed. Run `uv sync` first; the "
        "wrapper comes from the `ast` dependency group."
    )


def install_library(built: Path) -> Path:
    """Copy libadalang.so next to the wrapper that dlopens it."""
    target = package_dir() / "libadalang.so"
    logger.info("installing %s (%.1f MB) -> %s", built.name, built.stat().st_size / 1e6, target)
    shutil.copy2(built, target)
    return target


def verify(target: Path) -> None:
    """Import the bindings in a fresh process and parse a trivial unit."""
    probe = (
        "import libadalang as lal\n"
        "ctx = lal.AnalysisContext()\n"
        "unit = ctx.get_from_buffer('p.adb', 'procedure P is begin null; end P;')\n"
        "assert not unit.diagnostics, unit.diagnostics\n"
        "print('libadalang', lal.__name__, 'ok')\n"
    )
    env = {**os.environ, "LD_LIBRARY_PATH": str(target.parent)}
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        raise BuildFailed(f"the installed library does not import:\n{proc.stderr.strip()}")
    logger.info("%s", proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and install libadalang.so for the AST parser.")
    parser.add_argument("--force", action="store_true", help="rebuild even if a usable library exists")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if not shutil.which("alr"):
        print(
            "error: alr not found. The libadalang build goes through Alire; install it first.",
            file=sys.stderr,
        )
        return 1

    try:
        target = package_dir() / "libadalang.so"
        if target.exists() and not args.force:
            logger.info("%s already installed; use --force to rebuild", target)
            verify(target)
            return 0

        workspace = prepare_workspace()
        resolve_dependencies(workspace)
        built = build_library(workspace, ensure_gmp())
        installed = install_library(built)
        verify(installed)
    except BuildFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    logger.info("done. The AST parser now uses libadalang for exact extraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
