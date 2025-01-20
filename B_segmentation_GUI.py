import os
import sys
import logging
import torch
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QDoubleSpinBox,
    QMessageBox,
    QCheckBox,
    QSpinBox
)
from PyQt5.QtGui import QFont
from CMSegmentationToolkit.src.dtws_mc import run_freiburg_mc
from CMSegmentationToolkit.src.fileIO.download_files import download_nnu_model, download_testfile_segmentation


def run_prediction(
    input_files,
    nnunet_results_path,
    output_folder,
    bnd_id,
    mask_id,
    step_size,
    disable_tta,
    beta
):
    os.environ["nnUNet_results"] = nnunet_results_path
    os.environ["nnUNet_raw"] = ""
    os.environ["nnUNet_preprocessed"] = ""

    for input_file in input_files:
        logging.info(f"Running MC segmentation on {input_file}")
        run_freiburg_mc(
            input_file=input_file,
            folder_output=output_folder,
            bnd_dataset_id=bnd_id,
            mask_dataset_id=mask_id,
            betas=[beta],
            step_size=step_size,
            disable_tta=disable_tta
        )
        logging.info(f"Completed MC segmentation on {input_file}")


class DTWSMCGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("JoeGreiner", "CMSegmentationToolkit")
        self.files_to_predict = []
        self.initUI()
        self.loadSettings()


    def initUI(self):
        self.setWindowTitle("DTWSMC Segmentation GUI")
        figure_width = 700
        figure_height = 400
        self.setGeometry(200, 200, figure_width, figure_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        if torch.cuda.is_available():
            logging.info("GPU is available.")
        else:
            logging.info("No GPU detected.")

        # Output folder
        hbox_output = QHBoxLayout()
        layout.addLayout(hbox_output)
        lbl_output = QLabel("Output Folder:")
        hbox_output.addWidget(lbl_output)

        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output Folder")
        hbox_output.addWidget(self.le_output)

        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.select_output_folder)
        hbox_output.addWidget(btn_output)

        # nnUNet Results Folder
        hbox_nnunet = QHBoxLayout()
        layout.addLayout(hbox_nnunet)
        lbl_nnunet = QLabel("nnUNet Results Folder:")
        hbox_nnunet.addWidget(lbl_nnunet)

        self.le_nnunet = QLineEdit()
        self.le_nnunet.setPlaceholderText("nnUNet Results Folder")
        hbox_nnunet.addWidget(self.le_nnunet)

        btn_nnunet_browse = QPushButton("Browse")
        btn_nnunet_browse.clicked.connect(self.select_nnunet_folder)
        hbox_nnunet.addWidget(btn_nnunet_browse)

        btn_nnunet_download = QPushButton("Download Models")
        btn_nnunet_download.clicked.connect(self.download_nnunet_models)
        hbox_nnunet.addWidget(btn_nnunet_download)

        btn_test_data_download = QPushButton("Download Test Data")
        btn_test_data_download.clicked.connect(self.download_test_data)
        hbox_nnunet.addWidget(btn_test_data_download)


        # Model IDs
        hbox_model_ids = QHBoxLayout()
        layout.addLayout(hbox_model_ids)
        lbl_bnd_id = QLabel("Bnd Model ID:")
        self.spin_bnd_id = QSpinBox()
        self.spin_bnd_id.setRange(0, 9999)
        lbl_mask_id = QLabel("Mask Model ID:")
        self.spin_mask_id = QSpinBox()
        self.spin_mask_id.setRange(0, 9999)

        self.spin_bnd_id.valueChanged.connect(self.saveBndID)
        self.spin_mask_id.valueChanged.connect(self.saveMaskID)

        hbox_model_ids.addWidget(lbl_bnd_id)
        hbox_model_ids.addWidget(self.spin_bnd_id)
        hbox_model_ids.addWidget(lbl_mask_id)
        hbox_model_ids.addWidget(self.spin_mask_id)

        hbox_overlap_tta = QHBoxLayout()
        layout.addLayout(hbox_overlap_tta)
        lbl_overlap = QLabel("Step Size:")
        hbox_overlap_tta.addWidget(lbl_overlap)
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 1.0)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setValue(0.5)
        self.spin_overlap.valueChanged.connect(self.saveOverlapVal)

        self.cb_disable_tta = QCheckBox("Disable TTA")
        self.cb_disable_tta.stateChanged.connect(self.saveDisableTTA)

        lbl_beta = QLabel("Beta (for multicut):")
        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setRange(0, 1)
        self.spin_beta.setSingleStep(0.001)
        self.spin_beta.setValue(0.075)
        self.spin_beta.setDecimals(3)
        self.spin_beta.valueChanged.connect(self.saveBetaVal)

        hbox_overlap_tta.addWidget(self.spin_overlap)
        hbox_overlap_tta.addWidget(self.cb_disable_tta)
        hbox_overlap_tta.addWidget(lbl_beta)
        hbox_overlap_tta.addWidget(self.spin_beta)

        self.drop_label = QLabel("drop 3D WGA image (ITK-readable: .tif, .nrrd, ...) here\n expected axes: ZYX", self)

        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFont(QFont("Arial", 16))
        self.drop_label.setStyleSheet("border: 2px dashed #aaa;")
        layout.addWidget(self.drop_label)

        self.text_files = QTextEdit()
        self.text_files.setReadOnly(True)
        layout.addWidget(self.text_files)
        self.text_files.hide()

        self.btn_predict = QPushButton("Start Prediction")
        self.btn_predict.setStyleSheet("background-color: #10c46a; color: white; font-size: 20px;")
        self.btn_predict.clicked.connect(self.start_prediction)
        self.btn_predict.hide()
        layout.addWidget(self.btn_predict)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet(
                "background-color: #bce0ce; border: 2px dashed #888888; color: #10c46a;"
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("border: 2px dashed #aaa; color: black;")

    def dropEvent(self, event):
        self.files_to_predict.clear()
        for url in event.mimeData().urls():
            path = str(url.toLocalFile())
            self.files_to_predict.append(path)
        self.text_files.setText("\n".join(self.files_to_predict))
        self.text_files.show()
        self.btn_predict.show()  # Show button when files are dropped
        self.drop_label.setStyleSheet("border: 2px dashed #aaa; color: black;")

    def select_output_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if dir_path:
            self.le_output.setText(dir_path)
            self.saveOutputPath()

    def select_nnunet_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select nnUNet Results Folder")
        if dir_path:
            self.le_nnunet.setText(dir_path)
            self.saveNnuResultsPath()

    def download_nnunet_models(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit")

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Model")
        msg.setText("Downloading the model may take a while. Please be patient.")
        msg.exec()

        model_folder = download_nnu_model(tmp_directory)
        self.le_nnunet.setText(model_folder)
        if model_folder:
            self.le_nnunet.setText(model_folder)
            self.saveNnuResultsPath()

    def download_test_data(self):
        msg = QMessageBox()
        msg.setWindowTitle("Downloading Test Data")
        msg.setText("Downloading the test data may take a while. Please be patient.")
        msg.exec()

        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit", "test_segmentation")
        path_to_test_data = download_testfile_segmentation(tmp_directory, overwrite=False)
        self.files_to_predict.append(path_to_test_data)
        self.text_files.setText("\n".join(self.files_to_predict))
        self.text_files.show()
        self.btn_predict.show()

        msg = QMessageBox()
        msg.setWindowTitle("Downloaded Test Data")
        msg.setText(f"Test data downloaded to {path_to_test_data}. You can now start the prediction.")
        msg.exec()


    def start_prediction(self):
        if not self.files_to_predict:
            QMessageBox.warning(self, "Warning", "No input files selected.")
            return
        if not self.le_output.text():
            QMessageBox.warning(self, "Warning", "No output folder selected.")
            return

        run_prediction(
            input_files=self.files_to_predict,
            nnunet_results_path=self.le_nnunet.text(),
            output_folder=self.le_output.text(),
            bnd_id=self.spin_bnd_id.value(),
            mask_id=self.spin_mask_id.value(),
            step_size=self.spin_overlap.value(),
            disable_tta=self.cb_disable_tta.isChecked(),
            beta=self.spin_beta.value()
        )
        self.prediction_done()

    def prediction_done(self):
        QMessageBox.information(self, "Finished", "All predictions have completed.")

    def loadSettings(self):
        logging.info("Loading settings")
        for key in self.settings.allKeys():
            logging.info(f"\t{key}: {self.settings.value(key)}")


        output_path = self.settings.value("outputPathSegmentation", "")
        self.le_output.setText(output_path)

        nnu_folder = self.settings.value("nnuResultsPath", "")
        self.le_nnunet.setText(nnu_folder)

        bnd_id = self.settings.value("bndID", 821)  # default
        mask_id = self.settings.value("maskID", 822)  # default
        try:
            self.spin_bnd_id.setValue(int(bnd_id))
            self.spin_mask_id.setValue(int(mask_id))
        except ValueError:
            self.spin_bnd_id.setValue(821)
            self.spin_mask_id.setValue(822)

        overlap_val = self.settings.value("overlapVal", 0.5)
        try:
            self.spin_overlap.setValue(float(overlap_val))
        except ValueError:
            self.spin_overlap.setValue(0.5)

        disable_tta = self.settings.value("disableTTA", False)
        logging.info(f"loading disableTTA: {disable_tta}")
        if disable_tta == "true":
            self.cb_disable_tta.setChecked(True)
        else:
            self.cb_disable_tta.setChecked(False)

        beta_val = self.settings.value("betaVal", 0.075)
        try:
            self.spin_beta.setValue(float(beta_val))
        except ValueError:
            self.spin_beta.setValue(0.075)

    def saveOutputPath(self):
        self.settings.setValue("outputPathSegmentation", self.le_output.text())

    def saveNnuResultsPath(self):
        self.settings.setValue("nnuResultsPath", self.le_nnunet.text())

    def saveBndID(self):
        self.settings.setValue("bndID", self.spin_bnd_id.value())

    def saveMaskID(self):
        self.settings.setValue("maskID", self.spin_mask_id.value())

    def saveOverlapVal(self):
        self.settings.setValue("overlapVal", self.spin_overlap.value())

    def saveDisableTTA(self):
        self.settings.setValue("disableTTA", str(self.cb_disable_tta.isChecked()).lower())

    def saveBetaVal(self):
        self.settings.setValue("betaVal", self.spin_beta.value())

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    gui = DTWSMCGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
