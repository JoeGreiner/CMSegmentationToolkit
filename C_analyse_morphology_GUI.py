import os
import logging
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

        self.output_folder_layout.addWidget(self.output_folder_label)
        self.output_folder_layout.addWidget(self.output_folder)
        self.output_folder_layout.addWidget(self.browseButton_output)
        layout.addLayout(self.output_folder_layout)

        self.test_data_layout = QHBoxLayout()
        self.download_test_data_button = QPushButton("Download Test Data", self)
        self.download_test_data_button.clicked.connect(self.download_test_data_from_web)


        self.test_data_layout.addWidget(self.download_test_data_button)
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
        self.filterCheckBox.stateChanged.connect(self.saveSettings)
        self.filterSize.textChanged.connect(self.saveSettings)
        self.filterNCheckBox.stateChanged.connect(self.saveSettings)

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
        for url in e.mimeData().urls():
            if url.isLocalFile():
                path = str(url.toLocalFile())
                self.lineWidgetImage.setText(path)
                self.img_added = True

                self.dropLabelField.setStyleSheet(
                    "background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

                if self.img_added:
                    self.startButton.show()

    def showFileDialogOutputPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.output_folder.setText(dir_path)
            self.settings.setValue("outputPath", dir_path)

    def showFileDialogModelPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.model_folder.setText(dir_path)
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


    def saveSettings(self):
        logging.info("Saving settings")
        self.settings.setValue("outputPath", self.output_folder.text())
        self.settings.setValue("resolutionZ", self.resolutionZ.text())
        self.settings.setValue("resolutionY", self.resolutionY.text())
        self.settings.setValue("resolutionX", self.resolutionX.text())
        self.settings.setValue("filterByVolume", str(self.filterCheckBox.isChecked()).lower())
        self.settings.setValue("filterSize", self.filterSize.text())
        self.settings.setValue("filterNSize", self.filterNSize.text())




    def process(self, path_img):
        logging.info(f"Start analysis: {path_img}")

        stackname = os.path.splitext(os.path.basename(path_img))[0]
        output_path_all = os.path.join(self.output_folder.text(), f'{stackname}_morphology.xlsx')
        output_path_mean = os.path.join(self.output_folder.text(), f'{stackname}_morphology_mean.xlsx')
        output_path_filtered = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered.xlsx')
        output_path_filtered_mean = os.path.join(self.output_folder.text(), f'{stackname}_morphology_filtered_mean.xlsx')

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
        self.lineWidgetImage.setText(path_to_test_data)

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
