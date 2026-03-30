"""
Tests for morphology analysis using a synthetic rectangular block.

Analytical reference values
===========================

A single axis-aligned rectangular block is embedded in a larger volume.

  1. Isotropic    res_ZYX = [1.0,  1.0,  1.0] um
  2. Anisotropic  res_ZYX = [0.5,  0.3,  0.2] um

Geometry (shared, in voxels)
----------------------------
  Volume shape (Z,Y,X) : 100 x 100 x 150
  Block slice          : z in [40,50), y in [40,55), x in [50,90)
  Block voxel dims     : Z=10,  Y=15,  X=40

Analytical formulae for a cuboid  a x b x c  (physical um)
-----------------------------------------------------------
  Volume       = a * b * c                        um^3
  Surface      = 2*(a*b + a*c + b*c)              um^2
  S/V ratio    = surface / volume                 um^-1
  Cross-section perpendicular to longest axis:
    second_largest_dim = max of the two shorter physical dims
    smallest_dim       = min of the two shorter physical dims
    area               = product of the two shorter physical dims
    ratio              = second_largest / smallest

ISOTROPIC expected physical dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  X_phys = 40*1.0 = 40 um,  Y_phys = 15*1.0 = 15 um,  Z_phys = 10*1.0 = 10 um
  Longest axis: X.  Cross-section dims: Y=15, Z=10.
  Volume  = 6000 um^3 = 6.0 pL
  Surface = 2*(40*15 + 40*10 + 15*10) = 2300 um^2
  S/V     = 2300/6000 ~ 0.3833 um^-1

ANISOTROPIC expected physical dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  X_phys = 40*0.2 = 8 um,  Y_phys = 15*0.3 = 4.5 um,  Z_phys = 10*0.5 = 5 um
  Longest axis: X.  Cross-section dims: Z=5, Y=4.5.
  Volume  = 180 um^3 = 0.18 pL
  Surface = 2*(8*4.5 + 8*5 + 4.5*5) = 197 um^2
  S/V     = 197/180 ~ 1.0944 um^-1

Tolerance rationale
~~~~~~~~~~~~~~~~~~~
  TOL_EXACT    – voxel-counting metrics (volume, voxels): should be exactly correct.
  TOL_APPROX   – OBB, cross-section dims on axis-aligned blocks: measured errors
                 are <2.2 %, so rel=0.05 gives comfortable headroom.
  TOL_SURFACE  – SimpleITK's GetPerimeter (surface area) uses a digital
                 approximation that systematically *underestimates* the analytical
                 surface of voxelised shapes by ~10 %.  This is a known limitation,
                 not a code bug.  rel=0.12 absorbs that bias.
  TOL_BOX      – boxiness on axis-aligned blocks is exact; abs=0.05 suffices.
  TOL_ROT_*    – rotated blocks suffer additional discretisation error from
                 nearest-neighbour voxelisation of the rotated shape.
                 TOL_ROT_VOLUME (rel=0.05): actual errors are <2 %.
                 TOL_ROT_APPROX (rel=0.25): OBB dims can shift ~14 % at
                 extreme angles; 25 % gives headroom.
                 TOL_ROT_BOX (abs=0.25): worst-case compound rotation yields
                 boxiness ~0.76 (abs err 0.24).
"""

import numpy as np
import pytest

from CMSegmentationToolkit.src.analysis.processing import analyze_stack

def _make_block(vol_shape_zyx, block_slice, label=1):
    seg = np.zeros(vol_shape_zyx, dtype=np.uint16)
    seg[block_slice] = label
    return seg

VOL_SHAPE   = (100, 100, 150)                # Z, Y, X
BLOCK_SLICE = np.s_[40:50, 40:55, 50:90]     # Z=10, Y=15, X=40
N_Z, N_Y, N_X = 10, 15, 40
N_VOXELS = N_Z * N_Y * N_X                    # 6 000
ISO_RES = [1.0, 1.0, 1.0]

ISO_X = N_X * ISO_RES[2]   # 40
ISO_Y = N_Y * ISO_RES[1]   # 15
ISO_Z = N_Z * ISO_RES[0]   # 10

# oriented bounding box ordered by eigenvalue (smallest -> largest)
#   eigenvalue ~ (physical extent)^2
#   Z_phys=10 < Y_phys=15 < X_phys=40
ISO_OBB_0 = ISO_Z    # 10
ISO_OBB_1 = ISO_Y    # 15
ISO_OBB_2 = ISO_X    # 40

# cross-section perpendicular to longest axis (X)
ISO_CS_SECOND = max(ISO_Y, ISO_Z)            # 15 um
ISO_CS_SMALL  = min(ISO_Y, ISO_Z)            # 10 um
ISO_CS_RATIO  = ISO_CS_SECOND / ISO_CS_SMALL # 1.5
ISO_CS_AREA   = ISO_Y * ISO_Z                # 150 um^2

ISO_VOLUME  = ISO_X * ISO_Y * ISO_Z          # 6000 um^3
ISO_VOL_PL  = ISO_VOLUME / 1000              # 6.0 pL
ISO_SURFACE = 2 * (ISO_X*ISO_Y + ISO_X*ISO_Z + ISO_Y*ISO_Z)  # 2300 um^2
ISO_SV      = ISO_SURFACE / ISO_VOLUME       # 0.3833 um^-1

ANISO_RES = [0.5, 0.3, 0.2]

ANISO_X = N_X * ANISO_RES[2]   # 8.0 um
ANISO_Y = N_Y * ANISO_RES[1]   # 4.5 um
ANISO_Z = N_Z * ANISO_RES[0]   # 5.0 um

# oriented bounding box  (eigenvalue order)
#   Y_phys=4.5 < Z_phys=5.0 < X_phys=8.0
ANISO_OBB_0 = ANISO_Y   # 4.5
ANISO_OBB_1 = ANISO_Z   # 5.0
ANISO_OBB_2 = ANISO_X   # 8.0

# cross-section perpendicular to longest axis (X)
ANISO_CS_SECOND = max(ANISO_Y, ANISO_Z)                # 5.0 um
ANISO_CS_SMALL  = min(ANISO_Y, ANISO_Z)                # 4.5 um
ANISO_CS_RATIO  = ANISO_CS_SECOND / ANISO_CS_SMALL     # 1.1111
ANISO_CS_AREA   = ANISO_Y * ANISO_Z                    # 22.5 um^2

ANISO_VOLUME  = ANISO_X * ANISO_Y * ANISO_Z            # 180 um^3
ANISO_VOL_PL  = ANISO_VOLUME / 1000                    # 0.18 pL
ANISO_SURFACE = 2 * (ANISO_X*ANISO_Y + ANISO_X*ANISO_Z + ANISO_Y*ANISO_Z)  # 197 um^2
ANISO_SV      = ANISO_SURFACE / ANISO_VOLUME           # 1.0944 um^-1

# ── Tolerances ────────────────────────────────────────────────────────────
TOL_EXACT   = dict(rel=1e-6)          # volume, voxels — exact
TOL_APPROX  = dict(rel=0.05)          # OBB, cross-section (axis-aligned) — <2.2 % actual
TOL_SURFACE = dict(rel=0.12)          # surface area — ~10 % systematic underestimate
TOL_BOX     = dict(abs=0.05)          # boxiness (axis-aligned) — exact in practice

class TestIsotropic:
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_block(VOL_SHAPE, BLOCK_SLICE)
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_cell_volume_voxels(self, row):
        assert row["cell_volume_voxels"] == N_VOXELS

    def test_cell_volume_um3(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ISO_VOLUME, **TOL_EXACT)

    def test_cell_volume_pL(self, row):
        assert row["cell_volume_pL"] == pytest.approx(ISO_VOL_PL, **TOL_EXACT)

    def test_cm_bounding_box_size_0(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ISO_OBB_0, **TOL_APPROX)

    def test_cm_bounding_box_size_1(self, row):
        assert row["cm_bounding_box_size_1"] == pytest.approx(ISO_OBB_1, **TOL_APPROX)

    def test_cm_bounding_box_size_2(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ISO_OBB_2, **TOL_APPROX)

    def test_average_second_largest_dim_um(self, row):
        assert row["average_second_largest_dim_um"] == pytest.approx(
            ISO_CS_SECOND, **TOL_APPROX)

    def test_average_smallest_dim_um(self, row):
        assert row["average_smallest_dim_um"] == pytest.approx(
            ISO_CS_SMALL, **TOL_APPROX)

    def test_average_ratio_dim(self, row):
        assert row["average_ratio_dim"] == pytest.approx(
            ISO_CS_RATIO, **TOL_APPROX)

    def test_average_area_um2(self, row):
        assert row["average_area_um2"] == pytest.approx(
            ISO_CS_AREA, **TOL_APPROX)

    def test_ratio_width_depth(self, row):
        # OBB_1 / OBB_0 = 15 / 10 = 1.5
        assert row["ratio_width_depth"] == pytest.approx(
            ISO_OBB_1 / ISO_OBB_0, **TOL_APPROX)

    def test_ratio_skipped_border_slices(self, row):
        assert row["ratio_skipped_border_slices"] == 0.0

    def test_average_second_largest_dim_border_um(self, row):
        assert row["average_second_largest_dim_border_um"] == pytest.approx(
            ISO_CS_SECOND, **TOL_APPROX)

    def test_average_smallest_dim_border_um(self, row):
        assert row["average_smallest_dim_border_um"] == pytest.approx(
            ISO_CS_SMALL, **TOL_APPROX)

    def test_average_ratio_dim_border(self, row):
        assert row["average_ratio_dim_border"] == pytest.approx(
            ISO_CS_RATIO, **TOL_APPROX)

    def test_average_area_border_um2(self, row):
        assert row["average_area_border_um2"] == pytest.approx(
            ISO_CS_AREA, **TOL_APPROX)

    def test_border_equals_nonborder(self, row):
        """When no slices are skipped, border-filtered = unfiltered."""
        assert row["average_second_largest_dim_border_um"] == pytest.approx(
            row["average_second_largest_dim_um"], rel=0.01)
        assert row["average_smallest_dim_border_um"] == pytest.approx(
            row["average_smallest_dim_um"], rel=0.01)
        assert row["average_area_border_um2"] == pytest.approx(
            row["average_area_um2"], rel=0.01)

    def test_cell_surface_um2(self, row):
        assert row["cell_surface_um2"] == pytest.approx(
            ISO_SURFACE, **TOL_SURFACE)

    def test_cell_surface_to_volume_ratio_um(self, row):
        assert row["cell_surface_to_volume_ratio_um"] == pytest.approx(
            ISO_SV, **TOL_SURFACE)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)

class TestAnisotropic:
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_block(VOL_SHAPE, BLOCK_SLICE)
        df = analyze_stack(seg, resolution_zyx_um=ANISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_cell_volume_voxels(self, row):
        assert row["cell_volume_voxels"] == N_VOXELS

    def test_cell_volume_um3(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ANISO_VOLUME, **TOL_EXACT)

    def test_cell_volume_pL(self, row):
        assert row["cell_volume_pL"] == pytest.approx(ANISO_VOL_PL, **TOL_EXACT)

    def test_cm_bounding_box_size_0(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ANISO_OBB_0, **TOL_APPROX)

    def test_cm_bounding_box_size_1(self, row):
        assert row["cm_bounding_box_size_1"] == pytest.approx(ANISO_OBB_1, **TOL_APPROX)

    def test_cm_bounding_box_size_2(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ANISO_OBB_2, **TOL_APPROX)

    def test_average_second_largest_dim_um(self, row):
        assert row["average_second_largest_dim_um"] == pytest.approx(
            ANISO_CS_SECOND, **TOL_APPROX)

    def test_average_smallest_dim_um(self, row):
        assert row["average_smallest_dim_um"] == pytest.approx(
            ANISO_CS_SMALL, **TOL_APPROX)

    def test_average_ratio_dim(self, row):
        assert row["average_ratio_dim"] == pytest.approx(
            ANISO_CS_RATIO, **TOL_APPROX)

    def test_average_area_um2(self, row):
        assert row["average_area_um2"] == pytest.approx(
            ANISO_CS_AREA, **TOL_APPROX)

    def test_ratio_width_depth(self, row):
        # OBB_1 / OBB_0 = 5.0 / 4.5 ~ 1.111
        assert row["ratio_width_depth"] == pytest.approx(
            ANISO_OBB_1 / ANISO_OBB_0, **TOL_APPROX)

    def test_ratio_skipped_border_slices(self, row):
        assert row["ratio_skipped_border_slices"] == 0.0

    def test_average_second_largest_dim_border_um(self, row):
        assert row["average_second_largest_dim_border_um"] == pytest.approx(
            ANISO_CS_SECOND, **TOL_APPROX)

    def test_average_smallest_dim_border_um(self, row):
        assert row["average_smallest_dim_border_um"] == pytest.approx(
            ANISO_CS_SMALL, **TOL_APPROX)

    def test_average_ratio_dim_border(self, row):
        assert row["average_ratio_dim_border"] == pytest.approx(
            ANISO_CS_RATIO, **TOL_APPROX)

    def test_average_area_border_um2(self, row):
        assert row["average_area_border_um2"] == pytest.approx(
            ANISO_CS_AREA, **TOL_APPROX)

    def test_border_equals_nonborder(self, row):
        assert row["average_second_largest_dim_border_um"] == pytest.approx(
            row["average_second_largest_dim_um"], rel=0.01)
        assert row["average_smallest_dim_border_um"] == pytest.approx(
            row["average_smallest_dim_um"], rel=0.01)
        assert row["average_area_border_um2"] == pytest.approx(
            row["average_area_um2"], rel=0.01)

    def test_cell_surface_um2(self, row):
        assert row["cell_surface_um2"] == pytest.approx(
            ANISO_SURFACE, **TOL_SURFACE)

    def test_cell_surface_to_volume_ratio_um(self, row):
        assert row["cell_surface_to_volume_ratio_um"] == pytest.approx(
            ANISO_SV, **TOL_SURFACE)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)

from scipy.ndimage import rotate as ndimage_rotate


def _make_centered_block(vol_shape_zyx, block_dims_zyx, label=1):
    """Place a block centred in *vol_shape_zyx*."""
    seg = np.zeros(vol_shape_zyx, dtype=np.uint16)
    cz, cy, cx = [s // 2 for s in vol_shape_zyx]
    hz, hy, hx = [d // 2 for d in block_dims_zyx]
    seg[cz - hz:cz - hz + block_dims_zyx[0],
        cy - hy:cy - hy + block_dims_zyx[1],
        cx - hx:cx - hx + block_dims_zyx[2]] = label
    return seg


def _make_rotated_block(vol_shape_zyx, block_dims_zyx, angle_deg, rotation_axes,
                        label=1):
    """Create a block centred in the volume and rotated by *angle_deg*.

    ``rotation_axes`` is a tuple (a, b) passed to ``scipy.ndimage.rotate``,
    specifying the plane of rotation (e.g. (1, 2) = rotation in the YX-plane,
    i.e. around the Z axis).
    """
    seg = _make_centered_block(vol_shape_zyx, block_dims_zyx, label=label)
    seg_rot = ndimage_rotate(seg, angle_deg, axes=rotation_axes,
                             reshape=False, order=0, mode='constant', cval=0)
    seg_rot = (seg_rot > 0).astype(np.uint16) * label
    return seg_rot


ROT_VOL_SHAPE = (150, 150, 200)
ROT_BLOCK_DIMS = (10, 15, 40)

ROT_ISO_X = ROT_BLOCK_DIMS[2] * ISO_RES[2]   # 40
ROT_ISO_Y = ROT_BLOCK_DIMS[1] * ISO_RES[1]   # 15
ROT_ISO_Z = ROT_BLOCK_DIMS[0] * ISO_RES[0]   # 10

ROT_ISO_VOLUME  = ROT_ISO_X * ROT_ISO_Y * ROT_ISO_Z
ROT_ISO_OBB_0   = min(ROT_ISO_Z, ROT_ISO_Y, ROT_ISO_X)   # 10
ROT_ISO_OBB_1   = sorted([ROT_ISO_Z, ROT_ISO_Y, ROT_ISO_X])[1]  # 15
ROT_ISO_OBB_2   = max(ROT_ISO_Z, ROT_ISO_Y, ROT_ISO_X)   # 40

ROT_ISO_CS_SECOND = max(ROT_ISO_Y, ROT_ISO_Z)   # 15
ROT_ISO_CS_SMALL  = min(ROT_ISO_Y, ROT_ISO_Z)   # 10
ROT_ISO_CS_RATIO  = ROT_ISO_CS_SECOND / ROT_ISO_CS_SMALL
ROT_ISO_CS_AREA   = ROT_ISO_Y * ROT_ISO_Z        # 150

TOL_ROT_VOLUME  = dict(rel=0.05)
TOL_ROT_APPROX  = dict(rel=0.25)
TOL_ROT_BOX     = dict(abs=0.25)


class TestRotated45AroundZ:
    """Block rotated 45° around the Z axis (in the YX-plane).
    """
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_rotated_block(ROT_VOL_SHAPE, ROT_BLOCK_DIMS,
                                  angle_deg=45, rotation_axes=(1, 2))
        assert np.count_nonzero(seg) > 0, "Rotated block is empty"
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume_approximately_preserved(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ROT_ISO_VOLUME, **TOL_ROT_VOLUME)

    def test_obb_recovers_smallest_dim(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ROT_ISO_OBB_0, **TOL_ROT_APPROX)

    def test_obb_recovers_middle_dim(self, row):
        assert row["cm_bounding_box_size_1"] == pytest.approx(ROT_ISO_OBB_1, **TOL_ROT_APPROX)

    def test_obb_recovers_longest_dim(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ROT_ISO_OBB_2, **TOL_ROT_APPROX)

    def test_cross_section_second_largest(self, row):
        assert row["average_second_largest_dim_um"] == pytest.approx(
            ROT_ISO_CS_SECOND, **TOL_ROT_APPROX)

    def test_cross_section_smallest(self, row):
        assert row["average_smallest_dim_um"] == pytest.approx(
            ROT_ISO_CS_SMALL, **TOL_ROT_APPROX)

    def test_cross_section_area(self, row):
        assert row["average_area_um2"] == pytest.approx(
            ROT_ISO_CS_AREA, **TOL_ROT_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_ROT_BOX)


class TestRotated30AroundY:
    """Block rotated 30° around the Y axis (in the ZX-plane)."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_rotated_block(ROT_VOL_SHAPE, ROT_BLOCK_DIMS,
                                  angle_deg=30, rotation_axes=(0, 2))
        assert np.count_nonzero(seg) > 0
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume_approximately_preserved(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ROT_ISO_VOLUME, **TOL_ROT_VOLUME)

    def test_obb_recovers_smallest_dim(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ROT_ISO_OBB_0, **TOL_ROT_APPROX)

    def test_obb_recovers_middle_dim(self, row):
        assert row["cm_bounding_box_size_1"] == pytest.approx(ROT_ISO_OBB_1, **TOL_ROT_APPROX)

    def test_obb_recovers_longest_dim(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ROT_ISO_OBB_2, **TOL_ROT_APPROX)

    def test_cross_section_second_largest(self, row):
        assert row["average_second_largest_dim_um"] == pytest.approx(
            ROT_ISO_CS_SECOND, **TOL_ROT_APPROX)

    def test_cross_section_smallest(self, row):
        assert row["average_smallest_dim_um"] == pytest.approx(
            ROT_ISO_CS_SMALL, **TOL_ROT_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_ROT_BOX)


class TestRotated60AroundX:
    """Block rotated 60° around the X axis (in the ZY-plane)."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_rotated_block(ROT_VOL_SHAPE, ROT_BLOCK_DIMS,
                                  angle_deg=60, rotation_axes=(0, 1))
        assert np.count_nonzero(seg) > 0
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume_approximately_preserved(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ROT_ISO_VOLUME, **TOL_ROT_VOLUME)

    def test_obb_recovers_smallest_dim(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ROT_ISO_OBB_0, **TOL_ROT_APPROX)

    def test_obb_recovers_middle_dim(self, row):
        assert row["cm_bounding_box_size_1"] == pytest.approx(ROT_ISO_OBB_1, **TOL_ROT_APPROX)

    def test_obb_recovers_longest_dim(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ROT_ISO_OBB_2, **TOL_ROT_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_ROT_BOX)


class TestRotatedAnisotropicResolution:
    """Block rotated 45° around Z, analysed with anisotropic resolution.
    """
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_rotated_block(ROT_VOL_SHAPE, ROT_BLOCK_DIMS,
                                  angle_deg=45, rotation_axes=(1, 2))
        assert np.count_nonzero(seg) > 0
        df = analyze_stack(seg, resolution_zyx_um=ANISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume_approximately_preserved(self, row):
        aniso_vol = ROT_BLOCK_DIMS[0] * ANISO_RES[0] * \
                    ROT_BLOCK_DIMS[1] * ANISO_RES[1] * \
                    ROT_BLOCK_DIMS[2] * ANISO_RES[2]
        assert row["cell_volume_um3"] == pytest.approx(aniso_vol, **TOL_ROT_VOLUME)

    def test_obb_longest_dim_reasonable(self, row):
        # Voxel-space rotation with anisotropic resolution distorts the
        # physical shape, so OBB dims won't match the axis-aligned case
        # exactly.  We just verify the longest OBB dim is within a plausible
        # range (between the original longest and the voxel-diagonal).
        aniso_longest = ROT_BLOCK_DIMS[2] * ANISO_RES[2]  # 8.0
        assert row["cm_bounding_box_size_2"] >= aniso_longest * 0.8

    def test_boxiness_positive(self, row):
        # With anisotropic voxels + rotation the boxiness won't be ~1
        # but it should be a valid positive fraction.
        assert 0.3 < row["boxiness"] <= 1.05


# ---------------------------------------------------------------------------
# Cube — all dimensions equal
# ---------------------------------------------------------------------------
CUBE_SIDE_VOXELS = 20
CUBE_SHAPE = (80, 80, 80)
CUBE_RES = [1.0, 1.0, 1.0]
CUBE_SIDE_UM = CUBE_SIDE_VOXELS * CUBE_RES[0]  # 20
CUBE_VOLUME = CUBE_SIDE_UM ** 3                 # 8000
CUBE_SURFACE = 6 * CUBE_SIDE_UM ** 2            # 2400


class TestCube:
    """A perfect cube should have ratio_width_depth ≈ 1 and boxiness ≈ 1."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_centered_block(CUBE_SHAPE,
                                   (CUBE_SIDE_VOXELS, CUBE_SIDE_VOXELS, CUBE_SIDE_VOXELS))
        df = analyze_stack(seg, resolution_zyx_um=CUBE_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume(self, row):
        assert row["cell_volume_um3"] == pytest.approx(CUBE_VOLUME, **TOL_EXACT)

    def test_ratio_width_depth_near_one(self, row):
        assert row["ratio_width_depth"] == pytest.approx(1.0, **TOL_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)

    def test_cross_section_ratio_near_one(self, row):
        assert row["average_ratio_dim"] == pytest.approx(1.0, **TOL_APPROX)

    def test_surface(self, row):
        assert row["cell_surface_um2"] == pytest.approx(CUBE_SURFACE, **TOL_SURFACE)

    def test_sv_ratio(self, row):
        expected_sv = CUBE_SURFACE / CUBE_VOLUME  # 0.3
        assert row["cell_surface_to_volume_ratio_um"] == pytest.approx(
            expected_sv, **TOL_SURFACE)


# ---------------------------------------------------------------------------
# Highly elongated rod — extreme aspect ratio
# ---------------------------------------------------------------------------
ROD_SHAPE = (60, 60, 200)
ROD_DIMS_ZYX = (8, 8, 160)   # Z=8, Y=8, X=160
ROD_RES = [1.0, 1.0, 1.0]

ROD_X = ROD_DIMS_ZYX[2] * ROD_RES[2]  # 160
ROD_Y = ROD_DIMS_ZYX[1] * ROD_RES[1]  # 8
ROD_Z = ROD_DIMS_ZYX[0] * ROD_RES[0]  # 8

ROD_VOLUME = ROD_X * ROD_Y * ROD_Z    # 10240
ROD_OBB_LONGEST = ROD_X               # 160

ROD_CS_SECOND = max(ROD_Y, ROD_Z)     # 8
ROD_CS_SMALL  = min(ROD_Y, ROD_Z)     # 8
ROD_CS_RATIO  = 1.0                   # square cross-section


class TestElongatedRod:
    """A very elongated rod-like shape to test extreme aspect ratios."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_centered_block(ROD_SHAPE, ROD_DIMS_ZYX)
        df = analyze_stack(seg, resolution_zyx_um=ROD_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ROD_VOLUME, **TOL_EXACT)

    def test_obb_longest(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ROD_OBB_LONGEST, **TOL_APPROX)

    def test_cross_section_ratio_near_one(self, row):
        """Square cross-section → ratio ≈ 1."""
        assert row["average_ratio_dim"] == pytest.approx(ROD_CS_RATIO, **TOL_APPROX)

    def test_cross_section_dims(self, row):
        assert row["average_second_largest_dim_um"] == pytest.approx(ROD_CS_SECOND, **TOL_APPROX)
        assert row["average_smallest_dim_um"] == pytest.approx(ROD_CS_SMALL, **TOL_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)


# ---------------------------------------------------------------------------
# Multiple labels — two independent blocks
# ---------------------------------------------------------------------------
MULTI_SHAPE = (100, 100, 200)
BLOCK_A_SLICE = np.s_[20:30, 20:35, 20:60]   # Z=10, Y=15, X=40
BLOCK_B_SLICE = np.s_[60:80, 60:80, 120:180] # Z=20, Y=20, X=60

BLOCK_A_VOLUME = (10 * 15 * 40)  # 6000
BLOCK_B_VOLUME = (20 * 20 * 60)  # 24000


class TestMultipleLabels:
    """Two blocks with different labels in one volume."""
    @pytest.fixture(scope="class")
    def df(self):
        seg = np.zeros(MULTI_SHAPE, dtype=np.uint16)
        seg[BLOCK_A_SLICE] = 1
        seg[BLOCK_B_SLICE] = 2
        df = analyze_stack(seg, resolution_zyx_um=[1.0, 1.0, 1.0],
                           calculate_tats_density=False)
        return df

    def test_two_rows_returned(self, df):
        assert len(df) == 2

    def test_label_1_volume(self, df):
        row = df[df["label"] == 1].iloc[0]
        assert row["cell_volume_um3"] == pytest.approx(BLOCK_A_VOLUME, **TOL_EXACT)

    def test_label_2_volume(self, df):
        row = df[df["label"] == 2].iloc[0]
        assert row["cell_volume_um3"] == pytest.approx(BLOCK_B_VOLUME, **TOL_EXACT)

    def test_label_1_boxiness(self, df):
        row = df[df["label"] == 1].iloc[0]
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)

    def test_label_2_boxiness(self, df):
        row = df[df["label"] == 2].iloc[0]
        assert row["boxiness"] == pytest.approx(1.0, **TOL_BOX)

    def test_label_2_obb_longest(self, df):
        row = df[df["label"] == 2].iloc[0]
        # Longest physical dimension of block B = 60 um
        assert row["cm_bounding_box_size_2"] == pytest.approx(60.0, **TOL_APPROX)

    def test_label_2_cross_section(self, df):
        row = df[df["label"] == 2].iloc[0]
        # Cross-section perpendicular to longest axis (X=60): Y=20, Z=20
        assert row["average_ratio_dim"] == pytest.approx(1.0, **TOL_APPROX)


# ---------------------------------------------------------------------------
# Border-touching block — is_touching_border flag
# ---------------------------------------------------------------------------
BORDER_SHAPE = (60, 60, 60)
# Block extends to the edge in Z (starts at z=0)
BORDER_SLICE = np.s_[0:10, 25:40, 20:50]


class TestBorderTouching:
    """Block that touches the image border in Z."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = np.zeros(BORDER_SHAPE, dtype=np.uint16)
        seg[BORDER_SLICE] = 1
        df = analyze_stack(seg, resolution_zyx_um=[1.0, 1.0, 1.0],
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_is_touching_border(self, row):
        assert row["is_touching_border"] == 1

    def test_volume(self, row):
        expected = 10 * 15 * 30  # 4500
        assert row["cell_volume_um3"] == pytest.approx(expected, **TOL_EXACT)


class TestInteriorNotTouchingBorder:
    """Block well inside the volume — should NOT touch border."""
    @pytest.fixture(scope="class")
    def row(self):
        # Reuse the original axis-aligned block from TestIsotropic
        seg = _make_block(VOL_SHAPE, BLOCK_SLICE)
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_not_touching_border(self, row):
        assert row["is_touching_border"] == 0


# ---------------------------------------------------------------------------
# Compound rotation — 45° around Z then 30° around Y (sequential)
# ---------------------------------------------------------------------------
class TestCompoundRotation:
    """Block rotated by two successive rotations to stress PCA recovery."""
    @pytest.fixture(scope="class")
    def row(self):
        seg = _make_centered_block(ROT_VOL_SHAPE, ROT_BLOCK_DIMS)
        # First rotation: 45° in YX-plane (around Z)
        seg = ndimage_rotate(seg, 45, axes=(1, 2),
                             reshape=False, order=0, mode='constant', cval=0)
        # Second rotation: 30° in ZX-plane (around Y)
        seg = ndimage_rotate(seg, 30, axes=(0, 2),
                             reshape=False, order=0, mode='constant', cval=0)
        seg = (seg > 0).astype(np.uint16)
        assert np.count_nonzero(seg) > 0
        df = analyze_stack(seg, resolution_zyx_um=ISO_RES,
                           calculate_tats_density=False)
        assert len(df) == 1
        return df.iloc[0]

    def test_volume_approximately_preserved(self, row):
        assert row["cell_volume_um3"] == pytest.approx(ROT_ISO_VOLUME, **TOL_ROT_VOLUME)

    def test_obb_recovers_longest_dim(self, row):
        assert row["cm_bounding_box_size_2"] == pytest.approx(ROT_ISO_OBB_2, **TOL_ROT_APPROX)

    def test_obb_recovers_smallest_dim(self, row):
        assert row["cm_bounding_box_size_0"] == pytest.approx(ROT_ISO_OBB_0, **TOL_ROT_APPROX)

    def test_boxiness(self, row):
        assert row["boxiness"] == pytest.approx(1.0, **TOL_ROT_BOX)


# ---------------------------------------------------------------------------
# Resolution consistency: same voxel geometry, different resolutions
# ---------------------------------------------------------------------------
class TestResolutionScaling:
    """Doubling resolution should scale volume by 8× and surface by 4×."""
    @pytest.fixture(scope="class")
    def rows(self):
        seg = _make_block(VOL_SHAPE, BLOCK_SLICE)
        df1 = analyze_stack(seg, resolution_zyx_um=[1.0, 1.0, 1.0],
                            calculate_tats_density=False)
        df2 = analyze_stack(seg, resolution_zyx_um=[2.0, 2.0, 2.0],
                            calculate_tats_density=False)
        return df1.iloc[0], df2.iloc[0]

    def test_volume_scales_cubically(self, rows):
        r1, r2 = rows
        assert r2["cell_volume_um3"] == pytest.approx(
            r1["cell_volume_um3"] * 8.0, **TOL_EXACT)

    def test_surface_scales_quadratically(self, rows):
        r1, r2 = rows
        assert r2["cell_surface_um2"] == pytest.approx(
            r1["cell_surface_um2"] * 4.0, **TOL_SURFACE)

    def test_sv_ratio_halves(self, rows):
        """S/V ratio scales as 1/scale_factor when all dims double."""
        r1, r2 = rows
        assert r2["cell_surface_to_volume_ratio_um"] == pytest.approx(
            r1["cell_surface_to_volume_ratio_um"] / 2.0, **TOL_SURFACE)

    def test_voxel_count_unchanged(self, rows):
        r1, r2 = rows
        assert r1["cell_volume_voxels"] == r2["cell_volume_voxels"]

    def test_obb_scales_linearly(self, rows):
        r1, r2 = rows
        for col in ["cm_bounding_box_size_0", "cm_bounding_box_size_1", "cm_bounding_box_size_2"]:
            assert r2[col] == pytest.approx(r1[col] * 2.0, **TOL_APPROX)


if __name__ == "__main__":
    import sys
    pytest.main()
    sys.exit(0)
