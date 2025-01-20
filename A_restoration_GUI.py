from csbdeep.models import CARE
import os
import logging
import tensorflow as tf
import SimpleITK as sitk

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QPushButton, QApplication, QLabel,
                             QHBoxLayout,
                             QMessageBox, QFileDialog, QCheckBox)

from CMSegmentationToolkit.src.fileIO.download_files import download_care_model, download_testfile_restoration
from CMSegmentationToolkit.src.restoration.attenuation_correction import attenuation_correction
from CMSegmentationToolkit.src.restoration.care_restoration import predict_care


class GUI_Restoration(QWidget):
    def __init__(self):
        super(GUI_Restoration, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CM Segmentation Toolkit - CARE Restoration')
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

        ## Model Folder Selection
        self.model_folder_layout = QHBoxLayout()
        self.model_folder_label = QLabel("Model Folder: ")
        self.model_folder = QLineEdit()
        self.model_folder.setPlaceholderText("Model Folder")
        self.browseButton_model = QPushButton("Select Model Folder", self)
        self.browseButton_model.clicked.connect(self.showFileDialogModelPath)
        self.download_model_button = QPushButton("Download Model", self)
        self.download_model_button.clicked.connect(self.download_model_from_web)
        self.download_test_data_button = QPushButton("Download Test Data", self)
        self.download_test_data_button.clicked.connect(self.download_test_data_from_web)

        self.model_folder_layout.addWidget(self.model_folder_label)
        self.model_folder_layout.addWidget(self.model_folder)
        self.model_folder_layout.addWidget(self.browseButton_model)
        self.model_folder_layout.addWidget(self.download_model_button)
        self.model_folder_layout.addWidget(self.download_test_data_button)
        layout.addLayout(self.model_folder_layout)

        ## Drop Folder for TIFs Selection
        self.dropLabelField = QLabel("drop 3D WGA image (ITK-readable: .tif, .nrrd, ...) here\n expected axes: ZYX", self)
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

        # do attenuation correction label and button
        self.attenuationCorrectionLayout = QHBoxLayout()
        self.doAttenuationCorrectionLabel = QLabel("Attenuation Correction:", self)
        self.attenuationCorrectionLayout.addWidget(self.doAttenuationCorrectionLabel)
        self.doAttenuationCorrectionCheckbox = QCheckBox(self)
        self.doAttenuationCorrectionCheckbox.setChecked(True)
        self.attenuationCorrectionLayout.addWidget(self.doAttenuationCorrectionCheckbox)
        layout.addLayout(self.attenuationCorrectionLayout)
        self.doAttenuationCorrectionCheckbox.stateChanged.connect(self.saveSettings)


        ## Start Analysis Button
        self.startButton = QPushButton('Start Restoration', self)
        layout.addWidget(self.startButton)
        self.startButton.hide()
        self.startButton.clicked.connect(self.startProcessing)
        # make it green
        self.startButton.setStyleSheet("background-color: #10c46a; color: white; font-size: 20px;")

        self.setAcceptDrops(True)

        self.loadSettings()


    def widgetIsAdded(self, object):
        # Check if the button is in the layout
        index = self.layout().indexOf(object)
        return index != -1

    def startProcessing(self):
        self.process(path_img=self.lineWidgetImage.text())
        msg = QMessageBox()
        msg.setWindowTitle("Processing Done")
        msg.setText("The restoration process is done! Please manually validate the results.")
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

    def download_model_from_web(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit")

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Model")
        msg.setText("Downloading the model may take a while. Please be patient.")
        msg.exec()

        model_folder = download_care_model(tmp_directory)
        self.model_folder.setText(model_folder)
        if model_folder:
            self.model_folder.setText(model_folder)
            self.settings.setValue("modelPath", model_folder)

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
        self.settings = QSettings("JoeGreiner", "CMSegmentationToolkit")
        model_path = self.settings.value("modelPath")
        if model_path:
            self.model_folder.setText(model_path)
        output_path = self.settings.value("outputPath")
        if output_path:
            self.output_folder.setText(output_path)
        do_attenuation_correction = self.settings.value("doAttenuationCorrection")
        if do_attenuation_correction:
            if do_attenuation_correction == "true":
                self.doAttenuationCorrectionCheckbox.setChecked(True)
            else:
                self.doAttenuationCorrectionCheckbox.setChecked(False)

    def saveSettings(self):
        self.settings.setValue("modelPath", self.model_folder.text())
        self.settings.setValue("outputPath", self.output_folder.text())
        self.settings.setValue("doAttenuationCorrection", self.doAttenuationCorrectionCheckbox.isChecked())

    def process(self, path_img, do_attenuation_correction=True):
        logging.info("Start processing")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        axes = 'ZYX'
        model = CARE(config=None, name='confocal_pinhole_v3', basedir=self.model_folder.text())
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_img)).copy()

        prediction = predict_care(input_array=img, care_model=model, axes=axes)

        if do_attenuation_correction:
            prediction = attenuation_correction(prediction)

        output_path = os.path.join(self.output_folder.text(), os.path.basename(path_img))

        if os.path.exists(output_path):
            filename = os.path.basename(path_img)
            filename, file_extension = os.path.splitext(filename)
            output_path = os.path.join(self.output_folder.text(), filename + "_restored" + file_extension)

        itk_img = sitk.GetImageFromArray(prediction)
        sitk.WriteImage(itk_img, output_path)
        logging.info("Done")

    def download_test_data_from_web(self):

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Test Data")
        msg.setText("Downloading may take a while. Please be patient.")
        msg.exec()


        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit", "test_restoration")
        path_to_test_data = download_testfile_restoration(path_to_download_directory=tmp_directory, overwrite=False)
        self.lineWidgetImage.setText(path_to_test_data)

        self.img_added = True
        self.startButton.show()

        msg = QMessageBox()
        msg.setWindowTitle("Downloaded Test Data")
        msg.setText(f"Test data downloaded to {path_to_test_data}. You can now start the prediction.")
        msg.exec()



def run_GUI():
    app = QApplication([])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ex = GUI_Restoration()
    ex.show()
    app.exec()


if __name__ == '__main__':
    run_GUI()
