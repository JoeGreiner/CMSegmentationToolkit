import os
os.environ["LC_NUMERIC"] = "C"  # Fix locale-dependent decimal parsing in ITK/NRRD

import logging
import math
import subprocess
import sys
import SimpleITK as sitk

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QPushButton, QApplication, QLabel,
                             QHBoxLayout,
                             QMessageBox, QFileDialog, QCheckBox)

from CMSegmentationToolkit.src.analysis.processing import analyze_stack
from CMSegmentationToolkit.src.fileIO.download_files import download_testfile_morph_analysis
from CMSegmentationToolkit.src.fileIO.export_to_excel import export_dataframe_to_excel_autofit


class GUI_Restoration(QWidget):
    def __init__(self):
        super(GUI_Restoration, self).__init__()
        self.image_metadata = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CM Segmentation Toolkit - Morphology Analysis')
        figure_width = 800
        figure_height = 400
        self.setGeometry(300, 300, figure_width, figure_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        ## Output Folder Selection
        self.output_folder_layout = QHBoxLayout()
        self.output_folder_label = QLabel("Output Folder: ")
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText("Output Folder")
        self.browseButton_output = QPushButton("Select Output Folder", self)
        self.browseButton_output.clicked.connect(self.showFileDialogOutputPath)
        self.openButton_output = QPushButton("Open Output Folder", self)
        self.openButton_output.clicked.connect(self.openOutputFolder)

        self.output_folder_layout.addWidget(self.output_folder_label)
        self.output_folder_layout.addWidget(self.output_folder)
        self.output_folder_layout.addWidget(self.browseButton_output)
        self.output_folder_layout.addWidget(self.openButton_output)
        layout.addLayout(self.output_folder_layout)

        self.test_data_layout = QHBoxLayout()
        self.download_test_data_button = QPushButton("Download Test Data", self)
        self.download_test_data_button.clicked.connect(self.download_test_data_from_web)
        self.restore_defaults_button = QPushButton("Restore Default Settings", self)
        self.restore_defaults_button.clicked.connect(self.restoreDefaultSettings)

        self.test_data_layout.addWidget(self.download_test_data_button)
        self.test_data_layout.addWidget(self.restore_defaults_button)
        layout.addLayout(self.test_data_layout)

        ## Drop Folder for TIFs Selection
        self.dropLabelField = QLabel("drop 3D Segmentation (ITK-readable: .tif, .nrrd, ...) here\n expected axes: ZYX", self)
        self.dropLabelField.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 20, QFont.Bold)
        self.dropLabelField.setFont(font)
        self.dropLabelField.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
        self.dropLabelField.setGeometry(0, 0, figure_width, 300)
        layout.addWidget(self.dropLabelField)

        self.imageInputLayout = QHBoxLayout()
        layout.addLayout(self.imageInputLayout)

        self.imageInputLabel = QLabel("Image Path: ", self)
        self.imageInputLayout.addWidget(self.imageInputLabel)
        self.lineWidgetImage = QLineEdit()
        self.imageInputLayout.addWidget(self.lineWidgetImage)

        self.imageMetadataLayout = QVBoxLayout()
        self.imageMetadataTitle = QLabel(
            "Loaded image metadata (from file header; reordered to ZYX for comparison with the fields below):",
            self
        )
        self.imageMetadataTitle.setWordWrap(True)
        self.imageMetadataLayout.addWidget(self.imageMetadataTitle)

        self.imageSizeXYZLabel = QLabel("Image size (XYZ in file): not loaded", self)
        self.imageSizeZYXLabel = QLabel("Image size (ZYX for GUI/array): not loaded", self)
        self.imageSpacingXYZLabel = QLabel("Image spacing (XYZ in file): not loaded", self)
        self.imageSpacingZYXLabel = QLabel("Image spacing (ZYX for manual comparison): not loaded", self)
        self.imageResolutionCheckLabel = QLabel(
            "Resolution check: drag a file or download the test image to compare it with the manual Z,Y,X input.",
            self
        )
        self.imageResolutionCheckLabel.setWordWrap(True)

        self.imageMetadataLayout.addWidget(self.imageSizeXYZLabel)
        self.imageMetadataLayout.addWidget(self.imageSizeZYXLabel)
        self.imageMetadataLayout.addWidget(self.imageSpacingXYZLabel)
        self.imageMetadataLayout.addWidget(self.imageSpacingZYXLabel)
        self.imageMetadataLayout.addWidget(self.imageResolutionCheckLabel)
        layout.addLayout(self.imageMetadataLayout)

        self.img_added = False

        # 3 line edits for resolution z y x in micrometers
        self.resolutionLayout = QHBoxLayout()
        self.resolutionLabel = QLabel("Resolution (Z, Y, X) in micrometers: ", self)
        self.resolutionLayout.addWidget(self.resolutionLabel)
        self.resolutionZ = QLineEdit()
        self.resolutionZ.setPlaceholderText("Z (micrometers)")
        self.resolutionY = QLineEdit()
        self.resolutionY.setPlaceholderText("Y (micrometers)")
        self.resolutionX = QLineEdit()
        self.resolutionX.setPlaceholderText("X (micrometers)")
        self.resolutionLayout.addWidget(self.resolutionZ)
        self.resolutionLayout.addWidget(self.resolutionY)
        self.resolutionLayout.addWidget(self.resolutionX)
        layout.addLayout(self.resolutionLayout)

        # checkbox for filter by size and filter size in pL
        self.filterLayout = QHBoxLayout()
        self.filterCheckBox = QCheckBox("Additional filter by volume (pL): ", self)
        self.filterCheckBox.setChecked(False)
        self.filterLayout.addWidget(self.filterCheckBox)
        self.filterSize = QLineEdit()
        self.filterSize.setPlaceholderText("Threshold (pL)")
        self.filterLayout.addWidget(self.filterSize)
        layout.addLayout(self.filterLayout)

        # checkbox for filter for N largest cells
        self.filterNLayout = QHBoxLayout()
        self.filterNCheckBox = QCheckBox("Additional filter for N largest cells per stack: ", self)
        self.filterNCheckBox.setChecked(True)
        self.filterNLayout.addWidget(self.filterNCheckBox)
        self.filterNSize = QLineEdit()
        self.filterNSize.setPlaceholderText("N (number of cells)")
        self.filterNSize.setText("30")
        self.filterNLayout.addWidget(self.filterNSize)
        layout.addLayout(self.filterNLayout)




        ## Start Analysis Button
        self.startButton = QPushButton('Start Analysis', self)
        layout.addWidget(self.startButton)
        self.startButton.hide()
        self.startButton.clicked.connect(self.startProcessing)
        # make it green
        self.startButton.setStyleSheet("background-color: #10c46a; color: white; font-size: 20px;")

        self.setAcceptDrops(True)

        self.settings = QSettings("JoeGreiner", "CMSegmentationToolkit")
        self.loadSettings()

        # connect everything to save settings when  changed
        self.output_folder.textChanged.connect(self.saveSettings)
        self.resolutionZ.textChanged.connect(self.saveSettings)
        self.resolutionY.textChanged.connect(self.saveSettings)
        self.resolutionX.textChanged.connect(self.saveSettings)
        self.resolutionZ.textChanged.connect(self.update_resolution_comparison)
        self.resolutionY.textChanged.connect(self.update_resolution_comparison)
        self.resolutionX.textChanged.connect(self.update_resolution_comparison)
        self.filterCheckBox.stateChanged.connect(self.saveSettings)
        self.filterSize.textChanged.connect(self.saveSettings)
        self.filterNCheckBox.stateChanged.connect(self.saveSettings)
        self.filterNSize.textChanged.connect(self.saveSettings)
        self.lineWidgetImage.textChanged.connect(self.saveSettings)

    def widgetIsAdded(self, object):
        # Check if the button is in the layout
        index = self.layout().indexOf(object)
        return index != -1

    def startProcessing(self):
        self.process(path_img=self.lineWidgetImage.text())
        msg = QMessageBox()
        msg.setWindowTitle("Processing Done")
        msg.setText("The analysis is is done! Please manually validate the results.")
        msg.exec()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            print("Drag Enter Image")
            self.dropLabelField.setStyleSheet("background-color: #bce0ce; border: 2px dashed #888888; color: #10c46a;")
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        print("Drag Leave")
        self.dropLabelField.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

    def dropEvent(self, e):
        local_paths = [str(url.toLocalFile()) for url in e.mimeData().urls() if url.isLocalFile()]
        if not local_paths:
            self.dropLabelField.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
            return

        path = local_paths[0]
        if len(local_paths) > 1:
            logging.info("Multiple files dropped. Using the first file for preview: %s", path)

        self.set_selected_image(path)
        self.img_added = True

        self.dropLabelField.setStyleSheet(
            "background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

        if self.img_added:
            self.startButton.show()

    @staticmethod
    def _format_sequence(values, decimals=4):
        formatted = []
        for value in values:
            if isinstance(value, int):
                formatted.append(str(value))
            elif isinstance(value, float):
                formatted.append(f"{value:.{decimals}f}")
            else:
                formatted.append(str(value))
        return ", ".join(formatted)

    def _parse_manual_resolution_zyx(self):
        if not self.resolutionZ.text() or not self.resolutionY.text() or not self.resolutionX.text():
            return None

        try:
            return (
                float(self.resolutionZ.text()),
                float(self.resolutionY.text()),
                float(self.resolutionX.text())
            )
        except ValueError:
            return None

    @staticmethod
    def _values_match(values_a, values_b, rel_tol=1e-6, abs_tol=1e-6):
        return all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(values_a, values_b))

    def update_resolution_comparison(self):
        if self.image_metadata is None:
            self.imageResolutionCheckLabel.setText(
                "Resolution check: drag a file or download the test image to compare it with the manual Z,Y,X input."
            )
            self.imageResolutionCheckLabel.setStyleSheet("")
            return

        manual_resolution_zyx = self._parse_manual_resolution_zyx()
        if manual_resolution_zyx is None:
            self.imageResolutionCheckLabel.setText(
                "Resolution check: enter numeric Z, Y, X values to compare them with the image header."
            )
            self.imageResolutionCheckLabel.setStyleSheet("color: #666666;")
            return

        image_spacing_zyx = self.image_metadata['spacing_zyx']
        image_spacing_xyz = self.image_metadata['spacing_xyz']

        if len(manual_resolution_zyx) != len(image_spacing_zyx):
            self.imageResolutionCheckLabel.setText(
                "Resolution check: the loaded image spacing is not 3D, so it cannot be compared directly with Z,Y,X fields."
            )
            self.imageResolutionCheckLabel.setStyleSheet("color: #b26a00;")
            return

        if self._values_match(manual_resolution_zyx, image_spacing_zyx):
            self.imageResolutionCheckLabel.setText(
                "Resolution check: manual Z,Y,X matches the image header after reordering the file metadata to Z,Y,X."
            )
            self.imageResolutionCheckLabel.setStyleSheet("color: #10c46a;")
        elif self._values_match(manual_resolution_zyx, image_spacing_xyz):
            self.imageResolutionCheckLabel.setText(
                "Resolution check: manual values match the file's native X,Y,Z spacing. The fields below expect Z,Y,X, so please verify the axis order."
            )
            self.imageResolutionCheckLabel.setStyleSheet("color: #b26a00;")
        else:
            self.imageResolutionCheckLabel.setText(
                "Resolution check: manual Z,Y,X does not match the image header spacing in either Z,Y,X or X,Y,Z order. Please verify the spacing and axes."
            )
            self.imageResolutionCheckLabel.setStyleSheet("color: #c0392b;")

    def clear_image_metadata(self, message="Image metadata not loaded"):
        self.image_metadata = None
        self.imageSizeXYZLabel.setText(f"Image size (XYZ in file): {message}")
        self.imageSizeZYXLabel.setText(f"Image size (ZYX for GUI/array): {message}")
        self.imageSpacingXYZLabel.setText(f"Image spacing (XYZ in file): {message}")
        self.imageSpacingZYXLabel.setText(f"Image spacing (ZYX for manual comparison): {message}")
        self.update_resolution_comparison()

    def load_image_metadata(self, path):
        try:
            img = sitk.ReadImage(path)
            size_xyz = tuple(int(v) for v in img.GetSize())
            spacing_xyz = tuple(float(v) for v in img.GetSpacing())
        except Exception as exc:
            logging.exception("Failed to read image metadata from %s", path)
            self.clear_image_metadata("unavailable")
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText(f"Could not read image metadata from {path}.\n\n{exc}")
            msg.exec()
            return False

        size_zyx = tuple(reversed(size_xyz))
        spacing_zyx = tuple(reversed(spacing_xyz))
        self.image_metadata = {
            'size_xyz': size_xyz,
            'size_zyx': size_zyx,
            'spacing_xyz': spacing_xyz,
            'spacing_zyx': spacing_zyx,
        }

        self.imageSizeXYZLabel.setText(f"Image size (XYZ in file): {self._format_sequence(size_xyz, decimals=0)}")
        self.imageSizeZYXLabel.setText(f"Image size (ZYX for GUI/array): {self._format_sequence(size_zyx, decimals=0)}")
        self.imageSpacingXYZLabel.setText(
            f"Image spacing (XYZ in file): {self._format_sequence(spacing_xyz)} µm"
        )
        self.imageSpacingZYXLabel.setText(
            f"Image spacing (ZYX for manual comparison): {self._format_sequence(spacing_zyx)} µm"
        )
        self.update_resolution_comparison()
        logging.info("Loaded image metadata for %s: size_xyz=%s spacing_xyz=%s", path, size_xyz, spacing_xyz)
        return True

    def set_selected_image(self, path):
        self.lineWidgetImage.setText(path)
        return self.load_image_metadata(path)

    def showFileDialogOutputPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.output_folder.setText(dir_path)
            self.settings.setValue("outputPath", dir_path)

    def openOutputFolder(self):
        folder = self.output_folder.text()
        if not folder or not os.path.isdir(folder):
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Output folder does not exist. Please select a valid output folder first.")
            msg.exec()
            return

        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            logging.exception("Failed to open output folder: %s", folder)
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText(f"Could not open the output folder:\n{exc}")
            msg.exec()

    def showFileDialogModelPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            logging.info("Selected directory (unused model path handler): %s", dir_path)
            self.settings.setValue("modelPath", dir_path)

    def loadSettings(self):
        logging.info("Loading settings")
        # list all
        for key in self.settings.allKeys():
            logging.info(f"\tLoaded {key}: {self.settings.value(key)}")

        output_path = self.settings.value("outputPath")
        if output_path:
            self.output_folder.setText(output_path)
        resolutionZ = self.settings.value("resolutionZ")
        resolutionY = self.settings.value("resolutionY")
        resolutionX = self.settings.value("resolutionX")
        if resolutionZ and resolutionY and resolutionX:
            self.resolutionZ.setText(resolutionZ)
            self.resolutionY.setText(resolutionY)
            self.resolutionX.setText(resolutionX)
        filter_by_volume = self.settings.value("filterByVolume")
        if filter_by_volume:
            self.filterCheckBox.setChecked(filter_by_volume == 'true')
        filter_size = self.settings.value("filterSize")
        if filter_size:
            self.filterSize.setText(filter_size)
        filter_n = self.settings.value("filterNSize")
        if filter_n:
            self.filterNSize.setText(filter_n)
        filter_by_n = self.settings.value("filterByN")
        if filter_by_n:
            self.filterNCheckBox.setChecked(filter_by_n == 'true')
        image_path = self.settings.value("imagePath")
        if image_path and os.path.exists(image_path):
            self.set_selected_image(image_path)
            self.img_added = True
            self.startButton.show()


    def saveSettings(self):
        logging.info("Saving settings")
        self.settings.setValue("outputPath", self.output_folder.text())
        self.settings.setValue("resolutionZ", self.resolutionZ.text())
        self.settings.setValue("resolutionY", self.resolutionY.text())
        self.settings.setValue("resolutionX", self.resolutionX.text())
        self.settings.setValue("filterByVolume", str(self.filterCheckBox.isChecked()).lower())
        self.settings.setValue("filterSize", self.filterSize.text())
        self.settings.setValue("filterByN", str(self.filterNCheckBox.isChecked()).lower())
        self.settings.setValue("filterNSize", self.filterNSize.text())
        self.settings.setValue("imagePath", self.lineWidgetImage.text())

    def restoreDefaultSettings(self):
        reply = QMessageBox.question(
            self, "Restore Defaults",
            "Are you sure you want to restore all settings to their defaults?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        logging.info("Restoring default settings")
        self.settings.clear()

        self.output_folder.clear()
        self.resolutionZ.clear()
        self.resolutionY.clear()
        self.resolutionX.clear()
        self.filterCheckBox.setChecked(False)
        self.filterSize.clear()
        self.filterNCheckBox.setChecked(True)
        self.filterNSize.setText("30")
        self.lineWidgetImage.clear()
        self.clear_image_metadata()
        self.img_added = False
        self.startButton.hide()




    def process(self, path_img):
        logging.info(f"Start analysis: {path_img}")

        stackname = os.path.splitext(os.path.basename(path_img))[0]
        output_path_all = os.path.join(self.output_folder.text(), f'{stackname}_morphology.xlsx')
        output_path_mean = os.path.join(self.output_folder.text(), f'{stackname}_morphology_mean.xlsx')
        output_path_filtered = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered.xlsx')
        output_path_filtered_mean = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered_mean.xlsx')
        N = None
        output_path_filtered_N = None
        output_path_filtered_mean_N = None

        do_filter_by_N = self.filterNCheckBox.isChecked()
        if do_filter_by_N:
            N = self.filterNSize.text()
            if not N.isdigit() or int(N) <= 0:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Please enter a valid positive integer for N.")
                msg.exec()
                logging.error("Invalid input for N. Please enter a valid positive integer.")
                return
            logging.info(f"Filtering for N largest cells: {N}")
            output_path_filtered_N = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered_N_{N}.xlsx')
            output_path_filtered_mean_N = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered_mean_N_{N}.xlsx')

        # check file exist
        if not os.path.exists(path_img):
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText(f"File {path_img} does not exist. Please check the path.")
            msg.exec()
            logging.error(f"File {path_img} does not exist. Please check the path.")
            return

        img_array = sitk.GetArrayFromImage(sitk.ReadImage(path_img))

        # if resolution is not set, use default
        if not self.resolutionZ.text() or not self.resolutionY.text() or not self.resolutionX.text():
            logging.warning("Resolution not set, using default resolution of [1.0, 1.0, 1.0] (Z, Y, X in micrometers)")
            resolution = [1.0, 1.0, 1.0]
            # also qt box
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Resolution not set, using default resolution of [1.0, 1.0, 1.0] (Z, Y, X in micrometers)")
            msg.exec()
        else:
            resolution = [
                float(self.resolutionZ.text()),
                float(self.resolutionY.text()),
                float(self.resolutionX.text())
            ]
        logging.info(f"Using resolution: {resolution} (Z, Y, X in micrometers)")

        df = analyze_stack(img_array, resolution_zyx_um=resolution, calculate_tats_density=False)
        df['stackname'] = stackname
        df.index = stackname + "_cell_" + df.index.astype(str)
        export_dataframe_to_excel_autofit(df, output_path_all)

        df_mean = df.groupby('stackname').mean(numeric_only=True).reset_index()
        export_dataframe_to_excel_autofit(df_mean, output_path_mean)

        doFilter = self.filterCheckBox.isChecked()
        if doFilter:
            filterSize = self.filterSize.text()
            logging.info(f"Filtering by cell volume > {filterSize} pL")
            df_filtered = df[df['cell_volume_pL'] > float(filterSize)]
            export_dataframe_to_excel_autofit(df_filtered, output_path_filtered)

            df_filtered_mean = df_filtered.groupby('stackname').mean(numeric_only=True).reset_index()
            export_dataframe_to_excel_autofit(df_filtered_mean, output_path_filtered_mean)

        if do_filter_by_N:
            logging.info(f"Filtering for N largest cells: {N}")
            df_filtered_N = df.nlargest(int(N), 'cell_volume_pL')
            export_dataframe_to_excel_autofit(df_filtered_N, output_path_filtered_N)

            df_filtered_mean_N = df_filtered_N.groupby('stackname').mean(numeric_only=True).reset_index()
            export_dataframe_to_excel_autofit(df_filtered_mean_N, output_path_filtered_mean_N)
        logging.info("Done")

    def download_test_data_from_web(self):
        msg = QMessageBox()
        msg.setWindowTitle("Downloading Test Data")
        msg.setText("Downloading may take a while. Please be patient.")
        msg.exec()

        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit", "test_morph_analysis")
        path_to_test_data = download_testfile_morph_analysis(path_to_download_directory=tmp_directory, overwrite=False)
        self.set_selected_image(path_to_test_data)

        # set resolution to default values
        self.resolutionZ.setText("0.2")
        self.resolutionY.setText("0.2")
        self.resolutionX.setText("0.2")

        length_um = 84
        width_um = 42
        height_um = 21
        logging.info(f"Setting resolution to {self.resolutionZ.text()} (Z), {self.resolutionY.text()} (Y), {self.resolutionX.text()} (X) in micrometers")
        logging.info(f"Expected dimensions {length_um} (length), {width_um} (width), {height_um} (height) in micrometers")

        self.img_added = True
        self.startButton.show()

        msg = QMessageBox()
        msg.setWindowTitle("Downloaded Test Data")
        msg.setText(f"Test data downloaded to {path_to_test_data}. You can now start the analysis.")
        msg.exec()



def run_GUI():
    app = QApplication([])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ex = GUI_Restoration()
    ex.show()
    app.exec()


if __name__ == '__main__':
    run_GUI()
