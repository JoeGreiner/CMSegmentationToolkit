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
    beta,
    n_threads,
    min_size,
    # compactness=0,
    folds_bnd,
    folds_mask,
    plan_bnd='nnUNetPlans',
    plan_mask='nnUNetPlans',
    run_bnd_only_ws=False,
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
            disable_tta=disable_tta,
            n_threads=n_threads,
            min_size=min_size,
            # compactness=compactness,
            folds_bnd=folds_bnd,
            folds_mask=folds_mask,
            plan_bnd=plan_bnd,
            plan_mask=plan_mask,
            run_bnd_only_ws=run_bnd_only_ws
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
        self.spin_bnd_id.setValue(821)
        lbl_mask_id = QLabel("Mask Model ID:")
        self.spin_mask_id = QSpinBox()
        self.spin_mask_id.setRange(0, 9999)
        self.spin_mask_id.setValue(822)

        self.spin_bnd_id.valueChanged.connect(self.saveBndID)
        self.spin_mask_id.valueChanged.connect(self.saveMaskID)

        lbl_folds_bnd = QLabel("Folds:")
        self.folds_bnd = QLineEdit()
        self.folds_bnd.setPlaceholderText("0,1,2,3,4")
        self.folds_bnd.setText("0,1,2,3,4")  # default folds
        self.folds_bnd.setToolTip("Comma separated list of folds to use for Bnd model, 0-5 or 'all'")
        self.folds_bnd.textChanged.connect(self.saveFoldsBnd)

        lbl_folds_mask = QLabel("Folds:")
        self.folds_mask = QLineEdit()
        self.folds_mask.setPlaceholderText("0,1,2,3,4")
        self.folds_mask.setText("0,1,2,3,4")  # default folds
        self.folds_mask.setToolTip("Comma separated list of folds to use for Mask model, 0-5 or 'all'")
        self.folds_mask.textChanged.connect(self.saveFoldsMask)

        hbox_model_ids.addWidget(lbl_bnd_id)
        hbox_model_ids.addWidget(self.spin_bnd_id)
        hbox_model_ids.addWidget(lbl_folds_bnd)
        hbox_model_ids.addWidget(self.folds_bnd)
        hbox_model_ids.addWidget(lbl_mask_id)
        hbox_model_ids.addWidget(self.spin_mask_id)
        hbox_model_ids.addWidget(lbl_folds_mask)
        hbox_model_ids.addWidget(self.folds_mask)

        self.planner_hbox = QHBoxLayout()

        self.planner_bnd = QLabel("nnU-Net Plans Bnd:")
        self.qline_bnd_plans = QLineEdit()
        self.qline_bnd_plans.setPlaceholderText("nnUNetPlans")
        self.qline_bnd_plans.setText("nnUNetPlans")  # default plans
        self.qline_bnd_plans.textChanged.connect(self.saveBndPlans)
        self.planner_mask = QLabel("nnU-Net Plans Mask:")
        self.qline_mask_plans = QLineEdit()
        self.qline_mask_plans.setPlaceholderText("nnUNetPlans")
        self.qline_mask_plans.setText("nnUNetPlans")
        self.qline_mask_plans.textChanged.connect(self.saveMaskPlans)

        self.planner_hbox.addWidget(self.planner_bnd)
        self.planner_hbox.addWidget(self.qline_bnd_plans)
        self.planner_hbox.addWidget(self.planner_mask)
        self.planner_hbox.addWidget(self.qline_mask_plans)
        layout.addLayout(self.planner_hbox)

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
        self.spin_beta.setDecimals(3)
        self.spin_beta.setValue(0.075)
        self.spin_beta.valueChanged.connect(self.saveBetaVal)

        hbox_overlap_tta.addWidget(self.spin_overlap)
        hbox_overlap_tta.addWidget(self.cb_disable_tta)
        hbox_overlap_tta.addWidget(lbl_beta)
        hbox_overlap_tta.addWidget(self.spin_beta)

        # limit_cpu_threads integer input
        label_n_threads_layout = QHBoxLayout()
        label_n_threads = QLabel("Number Threads:")
        self.n_threads_spinbox = QSpinBox()
        self.n_threads_spinbox.setRange(1, os.cpu_count())
        self.n_threads_spinbox.setValue(os.cpu_count())

        label_n_threads_layout.addWidget(label_n_threads)
        label_n_threads_layout.addWidget(self.n_threads_spinbox)
        self.n_threads_spinbox.valueChanged.connect(self.saveNThreads)

        self.min_size_label = QLabel("Watershed Min Size:")
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(0, 1000000)
        self.min_size_spinbox.setValue(100)
        self.min_size_spinbox.valueChanged.connect(self.saveMinSize)
        label_n_threads_layout.addWidget(self.min_size_label)
        label_n_threads_layout.addWidget(self.min_size_spinbox)

        # # compactness
        # self.compactness_label = QLabel("Compactness:")
        # self.compactness_spinbox = QDoubleSpinBox()
        # self.compactness_spinbox.setRange(0, 100)
        # self.compactness_spinbox.setSingleStep(0.1)
        # self.compactness_spinbox.setValue(0)
        # self.compactness_spinbox.valueChanged.connect(self.saveCompactness)
        # label_n_threads_layout.addWidget(self.compactness_label)
        # label_n_threads_layout.addWidget(self.compactness_spinbox)

        layout.addLayout(label_n_threads_layout)

        # run additional bnd only watershed
        add_ws_layout = QHBoxLayout()
        # check only
        label_bnd_only_ws = QLabel("Run additional Bnd-only-watershed:")
        self.cb_add_ws = QCheckBox()
        # save
        self.cb_add_ws.stateChanged.connect(self.saveAddBndWS)
        add_ws_layout.addWidget(label_bnd_only_ws)
        add_ws_layout.addWidget(self.cb_add_ws)
        layout.addLayout(add_ws_layout)

        btn_defaults = QPushButton("Restore Defaults")
        btn_defaults.clicked.connect(self.restore_defaults)
        layout.addWidget(btn_defaults)



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
        default_path = self.settings.value("outputPathSegmentation", "")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder", default_path,
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
        if dir_path:
            self.le_output.setText(dir_path)
            self.saveOutputPath()

    def select_nnunet_folder(self):
        default_path = self.settings.value("nnuResultsPath", "")
        dir_path = QFileDialog.getExistingDirectory(self, "Select nnUNet Results Folder", default_path,
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
        if dir_path:
            self.le_nnunet.setText(dir_path)
            self.saveNnuResultsPath()

    def download_nnunet_models(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "CMSegmentationToolkit")

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Model")
        msg.setText("Downloading the model may take a while. Please be patient.")
        msg.exec()

        model_folder = download_nnu_model(tmp_directory, overwrite=False)
        self.le_nnunet.setText(model_folder)
        if model_folder:
            self.le_nnunet.setText(model_folder)
            self.saveNnuResultsPath()

        msg = QMessageBox()
        msg.setWindowTitle("Downloaded Model")
        msg.setText(f"Model downloaded to {model_folder}.")

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

        msg = QMessageBox()
        msg.setWindowTitle("Starting Prediction")
        msg.setText(f"Starting the prediction process for {len(self.files_to_predict)} files.\n")
        msg.exec()

        run_prediction(
            input_files=self.files_to_predict,
            nnunet_results_path=self.le_nnunet.text(),
            output_folder=self.le_output.text(),
            bnd_id=self.spin_bnd_id.value(),
            mask_id=self.spin_mask_id.value(),
            step_size=self.spin_overlap.value(),
            disable_tta=self.cb_disable_tta.isChecked(),
            beta=self.spin_beta.value(),
            n_threads=self.n_threads_spinbox.value(),
            min_size=self.min_size_spinbox.value(),
            # compactness=self.compactness_spinbox.value(),
            folds_bnd=self.folds_bnd.text(),
            folds_mask=self.folds_mask.text(),
            plan_bnd=self.qline_bnd_plans.text(),
            plan_mask=self.qline_mask_plans.text(),
            run_bnd_only_ws=self.cb_add_ws.isChecked()
        )
        self.prediction_done()

    def prediction_done(self):
        msg = QMessageBox()
        msg.setWindowTitle("Prediction Completed")
        msg.setText("The prediction process has been completed successfully.")
        msg.exec()



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

        folds_bnd = self.settings.value("foldsBnd", "0,1,2,3,4")
        self.folds_bnd.setText(folds_bnd)
        folds_mask = self.settings.value("foldsMask", "0,1,2,3,4")
        self.folds_mask.setText(folds_mask)

        bnd_plans = self.settings.value("bndPlans", "nnUNetPlans")
        self.qline_bnd_plans.setText(bnd_plans)
        mask_plans = self.settings.value("maskPlans", "nnUNetPlans")
        self.qline_mask_plans.setText(mask_plans)

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

        n_threads = self.settings.value("nThreads", os.cpu_count())
        self.n_threads_spinbox.setValue(int(n_threads))

        min_size = self.settings.value("minSize", 1000)
        self.min_size_spinbox.setValue(int(min_size))

        add_bnd_ws = self.settings.value("addBndWS", True)
        # cast string to boolean
        add_bnd_ws = add_bnd_ws.lower() == "true"
        self.cb_add_ws.setChecked(add_bnd_ws)

        # compactness = self.settings.value("compactness", 0)
        # self.compactness_spinbox.setValue(float(compactness))

    def restore_defaults(self):
        self.spin_bnd_id.setValue(821)
        self.spin_mask_id.setValue(822)
        self.spin_overlap.setValue(0.5)
        self.cb_disable_tta.setChecked(False)
        self.spin_beta.setValue(0.075)
        self.n_threads_spinbox.setValue(os.cpu_count())
        self.min_size_spinbox.setValue(100)
        # self.compactness_spinbox.setValue(0)
        self.folds_bnd.setText("0,1,2,3,4")
        self.folds_mask.setText("0,1,2,3,4")
        self.qline_bnd_plans.setText("nnUNetPlans")
        self.qline_mask_plans.setText("nnUNetPlans")
        self.cb_add_ws.setChecked(False)


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

    def saveNThreads(self):
        self.settings.setValue("nThreads", self.n_threads_spinbox.value())

    def saveMinSize(self):
        self.settings.setValue("minSize", self.min_size_spinbox.value())

    def saveAddBndWS(self):
        self.settings.setValue("addBndWS", self.cb_add_ws.isChecked())

    # def saveCompactness(self):
    #     self.settings.setValue("compactness", self.compactness_spinbox.value())

    def saveFoldsBnd(self):
        self.settings.setValue("foldsBnd", self.folds_bnd.text())

    def saveFoldsMask(self):
        self.settings.setValue("foldsMask", self.folds_mask.text())

    def saveBndPlans(self):
        self.settings.setValue("bndPlans", self.qline_bnd_plans.text())

    def saveMaskPlans(self):
        self.settings.setValue("maskPlans", self.qline_mask_plans.text())


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    gui = DTWSMCGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
