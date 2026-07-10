"""Setuptools adapter that ships the repository-owned config tree in wheels."""

from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithConfigs(build_py):
    """Copy configs beside ``vrl`` because runtime loading needs real paths."""

    def run(self) -> None:
        super().run()
        source = Path("configs")
        destination = Path(self.build_lib) / "vrl" / "configs"
        copytree(source, destination, dirs_exist_ok=True)


setup(cmdclass={"build_py": BuildPyWithConfigs})
