"""
conftest.py — Custom scored & annotated test reporter
======================================================

For every test this reporter prints one line:

    │  [  3/86]  ✅ PASS  Isotropic resolution (1×1×1 µm) — volume (µm³) is exact  Δ=0 µm³ (0.00%)

At the end it shows a progress bar and a final score.

Failure tracebacks are collected and printed in a dedicated section after
all tests have run, so the per-test score table is never interrupted.

Usage (picked up automatically by pytest via pytest.ini addopts):
    pytest                          # clean custom output only
    pytest -p terminal              # also re-enable pytest's own reporter
"""

from __future__ import annotations

import sys
import math

import pytest

# ─── Human-readable labels ────────────────────────────────────────────────────

_CLASS_LABELS: dict[str, str] = {
    "TestIsotropic":                    "Isotropic resolution (0.2×0.2×0.2 µm)",
    "TestAnisotropic":                  "Anisotropic resolution (0.3×0.2×0.1 µm)",
    "TestRotated45AroundZ":             "45° rotation around Z axis",
    "TestRotated30AroundY":             "30° rotation around Y axis",
    "TestRotated60AroundX":             "60° rotation around X axis",
    "TestRotatedAnisotropicResolution": "45° rotation + anisotropic resolution",
    "TestCube":                         "Perfect cube (all dimensions equal)",
    "TestElongatedRod":                 "Highly elongated rod (extreme aspect ratio)",
    "TestMultipleLabels":               "Multiple labels (two independent blocks)",
    "TestBorderTouching":               "Border-touching block (touches image edge)",
    "TestInteriorNotTouchingBorder":    "Interior block (well inside the volume)",
    "TestCompoundRotation":             "Compound rotation (45° around Z, then 30° around Y)",
    "TestResolutionScaling":            "Resolution scaling consistency",
}

_METRIC_LABELS: dict[str, str] = {
    "test_cell_volume_voxels":                   "volume (voxels) is exact",
    "test_cell_volume_um3":                      "volume (µm³) is exact",
    "test_cell_volume_pL":                       "volume (pL) is exact",
    "test_cm_bounding_box_size_0":               "OBB smallest dimension matches geometry",
    "test_cm_bounding_box_size_1":               "OBB middle dimension matches geometry",
    "test_cm_bounding_box_size_2":               "OBB longest dimension matches geometry",
    "test_average_second_largest_dim_um":        "cross-section 2nd-largest dim (µm) matches",
    "test_average_smallest_dim_um":              "cross-section smallest dim (µm) matches",
    "test_average_ratio_dim":                    "cross-section aspect ratio matches",
    "test_average_area_um2":                     "cross-section area (µm²) matches",
    "test_ratio_width_depth":                    "OBB width-to-depth ratio matches",
    "test_ratio_width_depth_near_one":           "OBB width-to-depth ratio ≈ 1 (cubic shape)",
    "test_ratio_skipped_border_slices":          "no border slices skipped",
    "test_average_second_largest_dim_border_um": "border cross-section 2nd-largest dim (µm) matches",
    "test_average_smallest_dim_border_um":       "border cross-section smallest dim (µm) matches",
    "test_average_ratio_dim_border":             "border cross-section aspect ratio matches",
    "test_average_area_border_um2":              "border cross-section area (µm²) matches",
    "test_border_equals_nonborder":              "border metrics equal non-border metrics",
    "test_cell_surface_um2":                     "surface area (µm²) matches analytical value",
    "test_cell_surface_to_volume_ratio_um":      "surface-to-volume ratio (µm⁻¹) matches analytical value",
    "test_boxiness":                             "boxiness ≈ 1 (perfectly rectangular shape)",
    "test_is_touching_border":                   "is_touching_border flag set correctly",
    "test_not_touching_border":                  "is_touching_border flag absent correctly",
    "test_volume":                               "volume matches analytical value",
    "test_two_rows_returned":                    "exactly two rows returned (one per label)",
    "test_label_1_volume":                       "label-1 volume matches analytical value",
    "test_label_2_volume":                       "label-2 volume matches analytical value",
    "test_label_1_boxiness":                     "label-1 boxiness ≈ 1",
    "test_label_2_boxiness":                     "label-2 boxiness ≈ 1",
    "test_label_2_obb_longest":                  "label-2 OBB longest dimension matches",
    "test_label_2_cross_section":                "label-2 cross-section aspect ratio ≈ 1",
    "test_obb_longest":                          "OBB longest dimension matches",
    "test_cross_section_ratio_near_one":         "cross-section aspect ratio ≈ 1 (square cross-section)",
    "test_cross_section_dims":                   "cross-section dimensions (µm) match",
    "test_sv_ratio":                             "surface-to-volume ratio matches",
    "test_surface":                              "surface area matches analytical value",
    "test_obb_recovers_smallest_dim":            "OBB recovers smallest dimension after rotation",
    "test_obb_recovers_middle_dim":              "OBB recovers middle dimension after rotation",
    "test_obb_recovers_longest_dim":             "OBB recovers longest dimension after rotation",
    "test_cross_section_second_largest":         "cross-section 2nd-largest dim after rotation",
    "test_cross_section_smallest":               "cross-section smallest dim after rotation",
    "test_cross_section_area":                   "cross-section area after rotation",
    "test_volume_approximately_preserved":       "volume approximately preserved after rotation",
    "test_obb_longest_dim_reasonable":           "OBB longest dimension within plausible range",
    "test_boxiness_positive":                    "boxiness is a valid positive fraction",
    "test_volume_scales_cubically":              "volume scales cubically with resolution",
    "test_surface_scales_quadratically":         "surface area scales quadratically with resolution",
    "test_sv_ratio_halves":                      "S/V ratio halves when resolution doubles",
    "test_voxel_count_unchanged":                "voxel count unchanged across resolutions",
    "test_obb_scales_linearly":                  "OBB dimensions scale linearly with resolution",
}

# ─── Unit for each test function (auto-detected by the `check` fixture) ───────

_METRIC_UNITS: dict[str, str] = {
    "test_cell_volume_voxels":                   "vox",
    "test_cell_volume_um3":                      "µm³",
    "test_cell_volume_pL":                       "pL",
    "test_cm_bounding_box_size_0":               "µm",
    "test_cm_bounding_box_size_1":               "µm",
    "test_cm_bounding_box_size_2":               "µm",
    "test_average_second_largest_dim_um":         "µm",
    "test_average_smallest_dim_um":              "µm",
    "test_average_ratio_dim":                    "",
    "test_average_area_um2":                     "µm²",
    "test_ratio_width_depth":                    "",
    "test_ratio_width_depth_near_one":           "",
    "test_ratio_skipped_border_slices":          "",
    "test_average_second_largest_dim_border_um":  "µm",
    "test_average_smallest_dim_border_um":       "µm",
    "test_average_ratio_dim_border":             "",
    "test_average_area_border_um2":              "µm²",
    "test_border_equals_nonborder":              "µm",
    "test_cell_surface_um2":                     "µm²",
    "test_cell_surface_to_volume_ratio_um":      "µm⁻¹",
    "test_boxiness":                             "",
    "test_volume":                               "µm³",
    "test_surface":                              "µm²",
    "test_sv_ratio":                             "µm⁻¹",
    "test_obb_longest":                          "µm",
    "test_cross_section_ratio_near_one":         "",
    "test_cross_section_dims":                   "µm",
    "test_label_1_volume":                       "µm³",
    "test_label_2_volume":                       "µm³",
    "test_label_1_boxiness":                     "",
    "test_label_2_boxiness":                     "",
    "test_label_2_obb_longest":                  "µm",
    "test_label_2_cross_section":                "",
    "test_volume_approximately_preserved":       "µm³",
    "test_obb_recovers_smallest_dim":            "µm",
    "test_obb_recovers_middle_dim":              "µm",
    "test_obb_recovers_longest_dim":             "µm",
    "test_cross_section_second_largest":         "µm",
    "test_cross_section_smallest":               "µm",
    "test_cross_section_area":                   "µm²",
    "test_volume_scales_cubically":              "µm³",
    "test_surface_scales_quadratically":         "µm²",
    "test_sv_ratio_halves":                      "µm⁻¹",
    "test_voxel_count_unchanged":                "vox",
    "test_obb_scales_linearly":                  "µm",
}

_WIDTH = 74


# ─── Description helper ───────────────────────────────────────────────────────

def _describe(item: pytest.Item) -> str:
    """Return a human-readable one-line description for *item*."""
    # Strip parametrize suffix ("[0]", "[param_name]", …)
    func_name = item.name.split("[")[0]

    # 1. Explicit method docstring (first line only)
    fn = getattr(item, "function", None)
    if fn and fn.__doc__:
        first = fn.__doc__.strip().split("\n")[0]
        if first:
            return first

    # 2. Class label + metric label
    class_label = ""
    cls = getattr(item, "cls", None)
    if cls:
        class_label = _CLASS_LABELS.get(cls.__name__, "")
        if not class_label and cls.__doc__:
            class_label = cls.__doc__.strip().split("\n")[0]

    metric_label = _METRIC_LABELS.get(func_name, "")
    if not metric_label:
        # Fall back: convert snake_case name to readable text
        raw = func_name[5:] if func_name.startswith("test_") else func_name
        metric_label = raw.replace("_", " ")

    if class_label:
        return f"{class_label} — {metric_label}"
    return metric_label.capitalize()


# ─── Reporter state ───────────────────────────────────────────────────────────

_results:      list[dict]            = []
_failures:     list[tuple[str, str]] = []   # (nodeid, longrepr)
_descriptions: dict[str, str]        = {}
_test_count:   int                   = 0
_current_num:  int                   = 0
_last_class:   str                   = ""
_metrics:      dict[str, list[dict]] = {}   # nodeid → [{abs_err, rel_err, unit}]


def _out(text: str = "", *, end: str = "\n") -> None:
    """Write directly to stdout, bypassing any pytest capture."""
    sys.stdout.write(text + end)
    sys.stdout.flush()


# ─── check fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def check(request):
    """Assert *actual* ≈ *expected* and record the error for the reporter.

    Usage inside a test method::

        def test_cell_volume_um3(self, row, check):
            check(row["cell_volume_um3"], ISO_VOLUME, **TOL_EXACT)

    Tolerance keyword arguments (``rel``, ``abs``) are forwarded to
    ``pytest.approx``.  If none are given, plain ``==`` is used.
    The unit is auto-detected from the test function name via
    ``_METRIC_UNITS``; pass ``unit=`` to override.
    """
    nodeid = request.node.nodeid
    func_name = request.node.name.split("[")[0]
    auto_unit = _METRIC_UNITS.get(func_name, "")
    _metrics[nodeid] = []

    def _check(actual, expected, unit=None, **tol_kwargs):
        u = unit if unit is not None else auto_unit
        try:
            a, e = float(actual), float(expected)
        except (TypeError, ValueError):
            a = e = 0.0
        abs_err = abs(a - e)
        if e != 0:
            rel_err = abs_err / abs(e) * 100.0
        elif a == 0:
            rel_err = 0.0
        else:
            rel_err = float("inf")

        _metrics[nodeid].append(
            {"actual": a, "expected": e, "abs_err": abs_err, "rel_err": rel_err, "unit": u}
        )

        if tol_kwargs:
            assert actual == pytest.approx(expected, **tol_kwargs)
        else:
            assert actual == expected

    return _check


# ─── Error formatting helper ─────────────────────────────────────────────────

def _format_error(nodeid: str) -> str:
    """Return a compact suffix with expected, computed, Δ, and %, or ``""``."""
    if nodeid not in _metrics or not _metrics[nodeid]:
        return ""

    entries = _metrics[nodeid]
    # Pick the single entry, or the one with the largest relative error
    e = max(entries, key=lambda x: x["rel_err"]) if len(entries) > 1 else entries[0]

    unit_str = f" {e['unit']}" if e["unit"] else ""
    expected_str = f"{e['expected']:.4g}"
    actual_str   = f"{e['actual']:.4g}"
    abs_str      = f"{e['abs_err']:.4g}"

    base = f"  expected={expected_str}{unit_str}  computed={actual_str}{unit_str}  Δ={abs_str}{unit_str}"

    if math.isinf(e["rel_err"]):
        return base
    return f"{base} ({e['rel_err']:.2f}%)"


# ─── Pytest hooks ─────────────────────────────────────────────────────────────

def pytest_collection_finish(session: pytest.Session) -> None:
    global _test_count
    _test_count = len(session.items)
    for item in session.items:
        _descriptions[item.nodeid] = _describe(item)

    _out()
    _out("═" * _WIDTH)
    _out("  CMSegmentationToolkit — Morphology Analysis Test Suite")
    _out(f"  {_test_count} tests collected")
    _out("═" * _WIDTH)
    _out()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:  # noqa: C901
    global _current_num, _last_class

    is_call_result  = report.when == "call"
    is_setup_error  = report.when == "setup" and report.failed

    if not (is_call_result or is_setup_error):
        return

    _current_num += 1
    nodeid = report.nodeid
    desc   = _descriptions.get(nodeid, nodeid.split("::")[-1])

    # ── section header when the test class changes ──
    parts      = nodeid.split("::")
    class_name = parts[-2] if len(parts) >= 3 else ""
    if class_name and class_name != _last_class:
        if _last_class:
            _out("  │")
        _last_class = class_name
        label = _CLASS_LABELS.get(class_name, class_name)
        _out(f"  ┌─ {label}")

    # ── status tag ──
    if report.passed:
        tag = "✅ PASS "
    elif is_setup_error:
        tag = "💥 ERROR"
    else:
        tag = "❌ FAIL "

    score = f"[{_current_num:3d}/{_test_count}]"
    error_str = _format_error(nodeid)
    _out(f"  │  {score}  {tag}  {desc}{error_str}")

    # ── collect failure details for end-of-run display ──
    if not report.passed and report.longrepr:
        _failures.append((nodeid, str(report.longrepr)))

    _results.append({"passed": report.passed, "nodeid": nodeid})


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print failure details and the final score bar.

    Using ``pytest_sessionfinish`` (called by pytest core) rather than
    ``pytest_terminal_summary`` (called only by the terminal reporter, which
    is disabled via ``-p no:terminal``).
    """
    passed = sum(1 for r in _results if r["passed"])
    failed = len(_results) - passed
    total  = len(_results)
    pct    = (passed / total * 100) if total else 0.0

    # ── failure details ──
    if _failures:
        _out()
        _out("─" * _WIDTH)
        _out("  FAILURES")
        _out("─" * _WIDTH)
        for nodeid, longrepr in _failures:
            _out()
            _out(f"  ✗  {nodeid}")
            for line in longrepr.splitlines():
                _out(f"     {line}")

    # ── score bar ──
    _out()
    _out("═" * _WIDTH)
    bar_len = 44
    filled  = round(bar_len * passed / total) if total else 0
    bar     = "█" * filled + "░" * (bar_len - filled)
    _out(f"  [{bar}]")
    score_line = f"  FINAL SCORE:  {passed} / {total}  ({pct:.1f} % passed)"
    if failed == 0:
        _out(score_line + "  🎉")
    else:
        _out(score_line + f"  —  {failed} FAILED")
    _out("═" * _WIDTH)
    _out()

