"""Regression tests for convert input path types."""

from pathlib import Path

import pytest

from cv_benchkit.cli import main


@pytest.mark.parametrize("dst", ["yolo", "voc"])
def test_coco_rejects_directory(
    dst: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    args = ["convert", "--from", "coco", "--to", dst, "--input", str(tmp_path)]
    args.extend(["--output", str(output)])

    assert main(args) == 1
    assert capsys.readouterr().err == (
        f"Error: input path '{tmp_path}' is a directory, expected a COCO JSON file.\n"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("src", "dst"),
    [("yolo", "coco"), ("yolo", "voc"), ("voc", "coco"), ("voc", "yolo")],
)
def test_directory_formats_reject_file(
    src: str, dst: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inp = tmp_path / "labels.txt"
    inp.write_text("", encoding="utf-8")
    output = tmp_path / "output"
    args = ["convert", "--from", src, "--to", dst, "--input", str(inp)]
    args.extend(["--output", str(output), "--categories", "dog"])
    args.extend(["--image-width", "100", "--image-height", "100"])

    assert main(args) == 1
    assert capsys.readouterr().err == (
        f"Error: input path '{inp}' must be a directory for {src.upper()} source format.\n"
    )
    assert not output.exists()
