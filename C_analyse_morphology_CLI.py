import os
os.environ["LC_NUMERIC"] = "C"  # Fix locale-dependent decimal parsing in ITK/NRRD

import sys
import logging
import argparse

import SimpleITK as sitk

from CMSegmentationToolkit.src.analysis.processing import analyze_stack
from CMSegmentationToolkit.src.fileIO.download_files import download_testfile_morph_analysis
from CMSegmentationToolkit.src.fileIO.export_to_excel import export_dataframe_to_excel_autofit


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="CM Segmentation Toolkit - Morphology Analysis (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Analyse a segmentation file with specified resolution\n"
            "  python C_analyse_morphology_CLI.py -i seg.tif -o results/ -rz 0.2 -ry 0.2 -rx 0.2\n"
            "\n"
            "  # Filter for the 30 largest cells\n"
            "  python C_analyse_morphology_CLI.py -i seg.tif -o results/ -rz 0.2 -ry 0.2 -rx 0.2 --filter-n 30\n"
            "\n"
            "  # Filter by minimum cell volume (pL)\n"
            "  python C_analyse_morphology_CLI.py -i seg.tif -o results/ -rz 0.2 -ry 0.2 -rx 0.2 --filter-volume 1.0\n"
            "\n"
            "  # Download test data and analyse it\n"
            "  python C_analyse_morphology_CLI.py --download-test-data -o results/\n"
        ),
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to the 3D segmentation file (ITK-readable: .tif, .nrrd, …). Expected axes: ZYX.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output folder for the Excel result files.",
    )
    parser.add_argument(
        "-rz", "--resolution-z",
        type=float,
        default=None,
        help="Voxel size in Z (micrometers). Default: 1.0",
    )
    parser.add_argument(
        "-ry", "--resolution-y",
        type=float,
        default=None,
        help="Voxel size in Y (micrometers). Default: 1.0",
    )
    parser.add_argument(
        "-rx", "--resolution-x",
        type=float,
        default=None,
        help="Voxel size in X (micrometers). Default: 1.0",
    )
    parser.add_argument(
        "--filter-volume",
        type=float,
        default=None,
        metavar="THRESHOLD_PL",
        help="If set, additionally filter cells whose volume is greater than this threshold (in pL).",
    )
    parser.add_argument(
        "--filter-n",
        type=int,
        default=None,
        metavar="N",
        help="If set, additionally keep only the N largest cells per stack.",
    )
    parser.add_argument(
        "--download-test-data",
        action="store_true",
        help="Download the test data and use it as input (ignores -i).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    return parser.parse_args(args)


def main(args=None):
    opts = parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if opts.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if opts.download_test_data:
        tmp_directory = os.path.join(
            os.path.expanduser("~"), "CMSegmentationToolkit", "test_morph_analysis"
        )
        path_img = download_testfile_morph_analysis(
            path_to_download_directory=tmp_directory, overwrite=False
        )
        logging.info(f"Test data downloaded to {path_img}")
        if opts.resolution_z is None:
            opts.resolution_z = 0.2
        if opts.resolution_y is None:
            opts.resolution_y = 0.2
        if opts.resolution_x is None:
            opts.resolution_x = 0.2
    elif opts.input:
        path_img = opts.input
    else:
        logging.error("Either --input or --download-test-data must be specified.")
        sys.exit(1)

    if not os.path.exists(path_img):
        logging.error(f"File {path_img} does not exist. Please check the path.")
        sys.exit(1)

    if opts.resolution_z is None or opts.resolution_y is None or opts.resolution_x is None:
        logging.warning(
            "Resolution not fully specified, using default resolution of "
            "[1.0, 1.0, 1.0] (Z, Y, X in micrometers)"
        )
        resolution = [
            opts.resolution_z if opts.resolution_z is not None else 1.0,
            opts.resolution_y if opts.resolution_y is not None else 1.0,
            opts.resolution_x if opts.resolution_x is not None else 1.0,
        ]
    else:
        resolution = [opts.resolution_z, opts.resolution_y, opts.resolution_x]
    logging.info(f"Using resolution: {resolution} (Z, Y, X in micrometers)")

    os.makedirs(opts.output, exist_ok=True)

    stackname = os.path.splitext(os.path.basename(path_img))[0]
    output_path_all = os.path.join(opts.output, f"{stackname}_morphology.xlsx")
    output_path_mean = os.path.join(opts.output, f"{stackname}_morphology_mean.xlsx")

    logging.info(f"Start analysis: {path_img}")
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(path_img))

    df = analyze_stack(img_array, resolution_zyx_um=resolution, calculate_tats_density=False)
    df["stackname"] = stackname
    df.index = stackname + "_cell_" + df.index.astype(str)

    export_dataframe_to_excel_autofit(df, output_path_all)
    logging.info(f"Saved per-cell results to {output_path_all}")

    df_mean = df.groupby("stackname").mean(numeric_only=True).reset_index()
    export_dataframe_to_excel_autofit(df_mean, output_path_mean)
    logging.info(f"Saved mean results to {output_path_mean}")

    if opts.filter_volume is not None:
        threshold = opts.filter_volume
        logging.info(f"Filtering by cell volume > {threshold} pL")
        df_filtered = df[df["cell_volume_pL"] > threshold]

        output_path_filtered = os.path.join(opts.output, f"{stackname}_morphology_filtered.xlsx")
        export_dataframe_to_excel_autofit(df_filtered, output_path_filtered)
        logging.info(f"Saved volume-filtered results to {output_path_filtered}")

        df_filtered_mean = df_filtered.groupby("stackname").mean(numeric_only=True).reset_index()
        output_path_filtered_mean = os.path.join(
            opts.output, f"{stackname}_morphology_filtered_mean.xlsx"
        )
        export_dataframe_to_excel_autofit(df_filtered_mean, output_path_filtered_mean)
        logging.info(f"Saved volume-filtered mean results to {output_path_filtered_mean}")

    # ── Optional N-largest filter ───────────────────────────────────────
    if opts.filter_n is not None:
        N = opts.filter_n
        if N <= 0:
            logging.error("N must be a positive integer.")
            sys.exit(1)
        logging.info(f"Filtering for {N} largest cells")
        df_filtered_N = df.nlargest(N, "cell_volume_pL")

        output_path_filtered_N = os.path.join(
            opts.output, f"{stackname}_morphology_filtered_N_{N}.xlsx"
        )
        export_dataframe_to_excel_autofit(df_filtered_N, output_path_filtered_N)
        logging.info(f"Saved N-largest filtered results to {output_path_filtered_N}")

        df_filtered_mean_N = df_filtered_N.groupby("stackname").mean(numeric_only=True).reset_index()
        output_path_filtered_mean_N = os.path.join(
            opts.output, f"{stackname}_morphology_filtered_mean_N_{N}.xlsx"
        )
        export_dataframe_to_excel_autofit(df_filtered_mean_N, output_path_filtered_mean_N)
        logging.info(f"Saved N-largest filtered mean results to {output_path_filtered_mean_N}")

    logging.info("Done")


if __name__ == "__main__":
    main()

