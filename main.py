import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QDialog, QFileDialog, QMessageBox
)

from main_window_ui import Ui_MainWindow
from clearing_ui import Ui_Clearing_Dialog
from masking_ui import Ui_Masking_Dialog
from infra import (
    read_file, check_available_columns, 
    simple_double_view, clearing_double_view, 
    masking_double_view, final_view, branch_double_view
)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.button_chose_file.clicked.connect(self.open_file)
        self.ui.button_view_chosen.clicked.connect(self.first_preview)

        self.ui.button_clearing.clicked.connect(self.open_clearing_dialog)
        self.ui.button_view_clearing.clicked.connect(self.view_clearing_result)

        self.ui.button_masking.clicked.connect(self.open_masking_dialog)
        self.ui.button_view_masking.clicked.connect(self.view_masking_result)

        self.ui.enter_mpcs.valueChanged.connect(self.mpcs_changed)
        self.ui.enter_mags.valueChanged.connect(self.mags_changed)
        self.ui.buttons_distance.accepted.connect(self.write_distance)
        self.ui.buttons_distance.rejected.connect(self.empty_distance)

        self.ui.enter_coef_1.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.enter_coef_2.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.enter_b_minus_v.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.buttons_reddening.accepted.connect(self.write_reddening)
        self.ui.buttons_reddening.rejected.connect(self.empty_reddening)

        self.ui.button_final_view.clicked.connect(self.view_final)

        # second page
        self.ui.button_view_spatial.clicked.connect(self.view_branch)
        self.ui.button_save_1.clicked.connect(self.save_1)

    def open_file(self):
        widget = QWidget(parent=self)
        self.file_path, _ = QFileDialog.getOpenFileName(widget, 'Open File')
        if self.file_path:
            print(self.file_path)
            try:
                self.data = read_file(self.file_path)
                self.ui.label_filename.setText(self.file_path)
                self.ui.button_view_chosen.setEnabled(True)
                self.ui.group_clearing.setEnabled(True)

            except pd.errors.ParserError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Invalid file")
                msg.setInformativeText('Chose .csv file with proper column names. Mandatory column names: ["x", "y", "mag_v", "err_v", "mag_i", "err_i"]')
                msg.setWindowTitle("Error")
                msg.exec_()

    def first_preview(self):
        simple_double_view(self.data)

    def view_clearing_result(self):
        self.clearing_res.show()

    def view_masking_result(self):
        self.masking_res.show()

    def open_clearing_dialog(self):
        window = QDialog(parent=self)
        window.ui = Ui_Clearing_Dialog()
        window.ui.setupUi(window)

        columns_availability = check_available_columns(self.data)
        window.ui.check_type.setChecked(columns_availability['type'])
        window.ui.check_type.setEnabled(columns_availability['type'])
        window.ui.check_mag.setChecked(columns_availability['mag'])
        window.ui.check_mag.setEnabled(columns_availability['mag'])
        window.ui.check_snr.setChecked(columns_availability['snr'])
        window.ui.check_snr.setEnabled(columns_availability['snr'])
        window.ui.check_sharp.setChecked(columns_availability['sharp'])
        window.ui.check_sharp.setEnabled(columns_availability['sharp'])
        window.ui.check_flag.setChecked(columns_availability['flag'])
        window.ui.check_flag.setEnabled(columns_availability['flag'])
        window.ui.check_crowd.setChecked(columns_availability['crowd'])
        window.ui.check_crowd.setEnabled(columns_availability['crowd'])

        def mark_clean_rows():
            marking = (self.data['mag_v'] == self.data['mag_v'])

            if window.ui.check_type.isChecked():
                marking *= (self.data['type'] <= window.ui.enter_type.value())
            if window.ui.check_mag.isChecked():
                marking *= ((self.data['mag_v'] < window.ui.enter_mag.value()) & 
                            (self.data['mag_i'] < window.ui.enter_mag.value()))
            if window.ui.check_snr.isChecked():
                marking *= ((self.data['snr_v'] >= window.ui.enter_snr.value()) & 
                            (self.data['snr_i'] >= window.ui.enter_snr.value()))
            if window.ui.check_sharp.isChecked():
                marking *= ((self.data['sharp_v'] + self.data['sharp_i'])**2 <= window.ui.enter_sharp.value())
            if window.ui.check_flag.isChecked():
                marking *= ((self.data['flag_v'] <= window.ui.enter_flag.value()) & 
                            (self.data['flag_i'] <= window.ui.enter_flag.value()))
            if window.ui.check_crowd.isChecked():
                marking *= ((self.data['crowd_v'] + self.data['crowd_i']) <= window.ui.enter_crowd.value())
            return marking

        def preview():
            marking = mark_clean_rows()
            clean_data = self.data[marking]
            dirty_data = self.data[~marking]
            fig = clearing_double_view(clean=clean_data, dirty=dirty_data)
            fig.show()

        def saving():
            marking = mark_clean_rows()
            clean_data = self.data[marking]
            dirty_data = self.data[~marking]
            self.clearing_res = clearing_double_view(clean=clean_data, dirty=dirty_data)
            self.data = self.data[marking]
            self.ui.button_view_clearing.setEnabled(True)
            self.ui.group_masking.setEnabled(True)

        window.ui.button_preview.clicked.connect(preview)
        window.ui.buttonBox.accepted.connect(saving)
        # window.ui.buttonBox.rejected.connect()
        window.show()

    def open_masking_dialog(self):
        window = QDialog(parent=self)
        window.ui = Ui_Masking_Dialog()
        window.ui.setupUi(window)

        x_min = self.data['x'].min()
        x_max = self.data['x'].max()
        y_min = self.data['y'].min()
        y_max = self.data['y'].max()

        window.ui.enter_x_left.setRange(x_min, x_max)
        window.ui.enter_x_right.setRange(x_min, x_max)
        window.ui.enter_x_left.setValue(x_min)
        window.ui.enter_x_right.setValue(x_max)

        window.ui.enter_y_bottom.setRange(y_min, y_max)
        window.ui.enter_y_top.setRange(y_min, y_max)
        window.ui.enter_y_bottom.setValue(y_min)
        window.ui.enter_y_top.setValue(y_max)

        def apply_mask():
            x_left = window.ui.enter_x_left.value()
            x_right = window.ui.enter_x_right.value()
            x_left, x_right = sorted([x_left, x_right])
            y_bottom = window.ui.enter_y_bottom.value()
            y_top = window.ui.enter_y_top.value()
            y_bottom, y_top = sorted([y_bottom, y_top])
            borders = [x_left, x_right, y_bottom, y_top]

            eps = 0.1
            masking = ((self.data['x'] > x_left - eps) & 
                       (self.data['x'] < x_right + eps) &
                       (self.data['y'] > y_bottom - eps) &
                       (self.data['y'] < y_top + eps))
            return borders, masking

        def preview():
            borders, masking = apply_mask()
            fig = masking_double_view(self.data, masking, borders)
            fig.show()

        def saving():
            borders, masking = apply_mask()
            self.masking_res = masking_double_view(self.data, masking, borders)
            self.data = self.data[masking]
            self.ui.button_view_masking.setEnabled(True)
            self.ui.group_distance.setEnabled(True)

        window.ui.button_preview.clicked.connect(preview)
        window.ui.buttonBox.accepted.connect(saving)
        # window.ui.buttonBox.rejected.connect()
        window.show()

    def mpcs_changed(self, value):
        if (value > 0):
            dist_mags = 5 * np.log10(value * 1e5)
        else:
            dist_mags = 0
        self.ui.enter_mags.blockSignals(True)
        self.ui.enter_mags.setValue(dist_mags)
        self.ui.enter_mags.blockSignals(False)

    def mags_changed(self, value):
        dist_mpcs = 10**(value / 5) / 1e5
        self.ui.enter_mpcs.blockSignals(True)
        self.ui.enter_mpcs.setValue(dist_mpcs)
        self.ui.enter_mpcs.blockSignals(False)

    def write_distance(self):
        self.dist = self.ui.enter_mags.value()
        self.ui.group_reddening.setEnabled(True)

    def empty_distance(self):
        self.ui.enter_mags.setValue(0)
        self.ui.enter_mpcs.setValue(0)

    def write_reddening(self):
        if (self.ui.tabs_reddening.currentIndex() == 0):
            self.redshift = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_1.value()
            self.absorbtion = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_2.value()
        else:
            self.redshift = self.ui.enter_redshift.value()
            self.absorbtion = self.ui.enter_absorbtion.value()
        self.ui.button_final_view.setEnabled(True)
        # self.ui.button_next_tab.setEnabled(True)

    def empty_reddening(self):
            self.ui.enter_b_minus_v.setValue(0.0)
            self.ui.enter_coef_1.setValue(2.742)
            self.ui.enter_coef_2.setValue(1.505)
            self.ui.enter_redshift.setValue(0.0)
            self.ui.enter_absorbtion.setValue(0.0)
            self.ui.label_redshift.setText('na')
            self.ui.label_absorbtion.setText('na')
    
    def recalculate_reddening_labels(self):
        redshift = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_1.value()
        absorbtion = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_2.value()
        self.ui.label_redshift.setText(f'{redshift:1.4}')
        self.ui.label_absorbtion.setText(f'{absorbtion:1.4}')
    
    def view_final(self):
        fig = final_view(self.data, self.dist, self.redshift, self.absorbtion)
        fig.show()
    
    def view_branch(self):
        params = {'dist':self.dist, 
        'redshift':self.redshift, 'absorbtion':self.absorbtion, 
        'vi_left':self.ui.enter_vi_left.value(),
        'vi_right':self.ui.enter_vi_right.value(), 
        'i_left':self.ui.enter_i_left.value(), 
        'i_right':self.ui.enter_i_right.value(), 
        'p_chosen':self.ui.enter_p.value(),
        'd_minus':self.ui.enter_d_minus.value(), 
        'd_plus':self.ui.enter_d_plus.value()}
        try:
            fig = branch_double_view(self.data, params)
            fig.show()
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Invalid borders")
            msg.setInformativeText('At least 1 star should be in chosen diapason.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def save_1(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
