import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import json

from PyQt5.QtCore import (
    QThread, pyqtSignal
)
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QDialog, QFileDialog, QMessageBox
)

from src.main_window_ui import Ui_MainWindow
from src.clearing_ui import Ui_Clearing_Dialog
from src.masking_ui import Ui_Masking_Dialog

from src.infra import (
    read_file, 
    check_available_columns, 
    create_pdf_out_of_figures
)
from src.simple_charts import (   
    get_overview_chart, 
    gat_clearing_chart, 
    get_masking_chart, 
    get_abs_mag_chart
)
from src.branch_approximation import (
    branch_two_step_analythis_support_functions, 
    calculate_branch_double_chart
)
from src.denisty_approximation import (
    get_density_chart, 
    density_choosing_region
)
from src.monte_carlo import (
    plot_histogrm_3x3, 
    iterate_over_n_experiments, 
    plot_monte_carlo_results
)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.main_tabs.setTabEnabled(1, False)
        self.ui.main_tabs.setTabEnabled(2, False)

        # first page
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

        self.ui.button_final_view.clicked.connect(self.view_page1_final)

        # second page
        self.ui.button_view_spatial.clicked.connect(self.view_branch)
        self.ui.button_save_1.clicked.connect(self.save_branch_approx)

        # third page
        self.ui.button_view_density.clicked.connect(self.view_density)
        self.ui.button_save_2.clicked.connect(self.calculate_and_save_density_manager) 

    def open_file(self):
        widget = QWidget(parent=self)
        self.file_path, _ = QFileDialog.getOpenFileName(widget, 'Open File')
        if self.file_path:
            try:
                self.data = read_file(self.file_path)
                self.data_raw = self.data.copy()
                self.ui.label_filename.setText(self.file_path)
                self.ui.button_view_chosen.setEnabled(True)
                self.ui.group_clearing.setEnabled(True)

                self.ui.button_view_clearing.setEnabled(False)
                self.ui.group_masking.setEnabled(False)
                self.ui.button_view_masking.setEnabled(False)
                self.ui.group_distance.setEnabled(False)
                self.ui.group_reddening.setEnabled(False)
                self.ui.button_final_view.setEnabled(False)
                self.ui.check_add_kde_1.setEnabled(False)
                
                plt.close('all')

            except pd.errors.ParserError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Invalid file")
                msg.setInformativeText('Chose .csv file with proper column names. Mandatory column names: ["x", "y", "mag_v", "err_v", "mag_i", "err_i"]')
                msg.setWindowTitle("Error")
                msg.exec_()

    def first_preview(self):
        fig_raw = get_overview_chart(self.data)
        fig_raw.suptitle('Raw data')
        fig_raw.show()

    ##########################################################################

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
            fig = gat_clearing_chart(clean=clean_data, dirty=dirty_data)
            fig.show()

        def saving():
            marking = mark_clean_rows()
            clean_data = self.data[marking]
            dirty_data = self.data[~marking]
            self.fig_clear = gat_clearing_chart(clean=clean_data, dirty=dirty_data)
            self.data = self.data[marking]
            self.ui.button_view_clearing.setEnabled(True)
            self.ui.group_masking.setEnabled(True)
            self.ui.group_distance.setEnabled(True)

        window.ui.button_preview.clicked.connect(preview)
        window.ui.buttonBox.accepted.connect(saving)
        # window.ui.buttonBox.rejected.connect()
        window.show()

    def view_clearing_result(self):
        self.fig_clear.show()

    ##########################################################################

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
            fig = get_masking_chart(self.data, masking, borders)
            fig.show()

        def saving():
            borders, masking = apply_mask()
            self.fig_mask = get_masking_chart(self.data, masking, borders)
            self.data = self.data[masking]
            self.ui.button_view_masking.setEnabled(True)
            self.ui.group_distance.setEnabled(True)

        window.ui.button_preview.clicked.connect(preview)
        window.ui.buttonBox.accepted.connect(saving)
        # window.ui.buttonBox.rejected.connect()
        window.show()

    def view_masking_result(self):
        self.fig_mask.show()

    ##########################################################################

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
        self.dist_in_mpcs = self.ui.enter_mpcs.value()
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
        self.ui.check_add_kde_1.setEnabled(True)
        self.ui.main_tabs.setTabEnabled(1, True)
        self.ui.main_tabs.setTabEnabled(2, True)

    def empty_reddening(self):
            self.ui.enter_b_minus_v.setValue(0.0)
            self.ui.enter_coef_1.setValue(2.742)
            self.ui.enter_coef_2.setValue(1.505)
            self.ui.enter_redshift.setValue(0.0)
            self.ui.enter_absorbtion.setValue(0.0)
            self.ui.label_redshift.setText('0.0')
            self.ui.label_absorbtion.setText('0.0')
    
    def recalculate_reddening_labels(self):
        redshift = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_1.value()
        absorbtion = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_2.value()
        self.ui.label_redshift.setText(f'{redshift:1.4}')
        self.ui.label_absorbtion.setText(f'{absorbtion:1.4}')
    
    def view_page1_final(self):
        add_kde = self.ui.check_add_kde_1.checkState() == 2
        fig = get_abs_mag_chart(self.data, self.dist, self.redshift, self.absorbtion, add_kde)
        fig.show()
    
    ##########################################################################
    
    def pack_all_branch_approx_parameters_in_dict(self):
        params = {
            'dist':self.dist, 
            'redshift':self.redshift,
            'absorbtion':self.absorbtion, 
            'i_level':self.ui.enter_level.value(),
            'vi_left':self.ui.enter_vi_left.value(),
            'vi_right':self.ui.enter_vi_right.value(), 
            'i_left':self.ui.enter_i_left.value(), 
            'i_right':self.ui.enter_i_right.value(), 
            'p_chosen':self.ui.enter_p.value(),
            'd_minus':self.ui.enter_d_minus.value(), 
            'd_plus':self.ui.enter_d_plus.value()
        }
        return params

    def view_branch(self):
        params = self.pack_all_branch_approx_parameters_in_dict()
        try:
            chosen_bool, inliers_bool, f_approx, f_std = branch_two_step_analythis_support_functions(self.data, params)
            fig = calculate_branch_double_chart(self.data, params, chosen_bool, inliers_bool, f_approx, f_std)
            fig.show()
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Invalid borders")
            msg.setInformativeText('At least 1 star should be in chosen diapason.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def save_branch_approx(self):
        fig_raw_overview = get_overview_chart(self.data_raw, point_size=2)
        fig_raw_overview.suptitle('Raw data')

        fig_new_overview = get_overview_chart(self.data, point_size=2)
        fig_new_overview.suptitle('Cleaned data')

        fig_absmag = get_abs_mag_chart(
            self.data, self.dist, self.redshift, self.absorbtion, add_kde=False, point_size=2
        )

        params = self.pack_all_branch_approx_parameters_in_dict()
        chosen_bool, inliers_bool, f_approx, f_std = branch_two_step_analythis_support_functions(self.data, params)
        
        fig_result = calculate_branch_double_chart(self.data, params, chosen_bool, inliers_bool, f_approx, f_std)
        
        d_m = params['d_minus']
        d_p = params['d_plus']
        i_mag = params['i_level']

        estimate = f_approx(i_mag)
        estimate_high = f_approx(i_mag+d_p) - f_std(i_mag+d_p)
        estimate_low = f_approx(i_mag-d_m) + f_std(i_mag-d_m)

        data = {
            'method' : 'approximation of a branch by a parabola',
            'filename' : self.file_path,
            'I mag level' : params['i_level'],
            '(V-I) color estimate' : estimate,
            '(V-I) color estimate low' : estimate_low,
            '(V-I) color estimate high' : estimate_high,
            'distance [in mag]' : params['dist'],
            'distance low' : params['dist'] - d_m,
            'distance high' : params['dist'] + d_p,
            'distance [in mpcs]' : self.dist_in_mpcs,
            'absorbtion (I)' : params['absorbtion'],
            'redshift (V-I)' : params['redshift'],
            'paramters' : {
                'vi_left': params['vi_left'],
                'vi_right': params['vi_right'], 
                'i_top': params['i_left'], 
                'i_bottom': params['i_right'], 
                'p_chosen': params['p_chosen'],
            }
        }

        filename, _ = QFileDialog.getSaveFileName(self, 'Save JSON', 'output')
        with open(filename, "w") as out_file:
            json.dump(data, out_file, indent=4)

        figures = [fig_raw_overview, fig_new_overview, fig_absmag, fig_result]
        filename, _ = QFileDialog.getSaveFileName(self, 'Save PDF', 'output')

        pdf = create_pdf_out_of_figures(figures)
        pdf.output(filename)
        for fig in figures:
            plt.close(fig)

    ##########################################################################

    def pack_all_density_parameters_in_dict(self):
        params = {
            'dist':self.dist, 
            'redshift':self.redshift,
            'absorbtion':self.absorbtion, 
            'i_level':self.ui.enter_level_2.value(),
            'd_minus':self.ui.enter_d_minus_2.value(), 
            'd_plus':self.ui.enter_d_plus_2.value(),
            'vi_left':self.ui.enter_vi_left_2.value(),
            'vi_right':self.ui.enter_vi_right_2.value(),
            's_scaler':self.ui.enter_s.value(),
        }
        chosen_bool, i_level_low, i_level_high, mean_i_error, mean_color_error = density_choosing_region(self.data, params)
        params['chosen_bool'] = chosen_bool
        params['i_level_low'] = i_level_low
        params['i_level_high'] = i_level_high
        params['mean_i_error'] = mean_i_error
        params['mean_color_error'] = mean_color_error
        return params

    def view_density(self):
        params = self.pack_all_density_parameters_in_dict()
        try:
            fig = get_density_chart(self.data, params, smoothing_bw=0.1)
            fig.show()
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Invalid borders")
            msg.setInformativeText('At least 1 star should be in chosen diapason.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def calculate_and_save_density_manager(self):
        self.thread = CalculateAndSaveDensity(self)
        self.thread.started.connect(self.show_processing)
        self.thread.finished.connect(self.show_done)
        self.thread.result_signal.connect(self.save_results)
        self.thread.start()

    def show_processing(self):
        self.ui.button_save_2.setText("Processing ⏱")
        self.ui.button_save_2.setEnabled(False)

    def show_done(self):
        self.ui.button_save_2.setText('Calculate and save (json + pdf) [may take some time ⏱]')
        self.ui.button_save_2.setEnabled(True)

    def save_results(self, pdf, data):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save JSON', 'output')
        if filename != '':
            with open(filename, "w") as out_file:
                json.dump(data, out_file, indent=4)

        filename, _ = QFileDialog.getSaveFileName(self, 'Save PDF', 'output')
        if filename != '':
            pdf.output(filename)

    def calculate_density_approach(self):
        params = self.pack_all_density_parameters_in_dict()

        fig_raw_overview = get_overview_chart(self.data_raw, point_size=2)
        fig_raw_overview.suptitle('Raw data')

        fig_new_overview = get_overview_chart(self.data, point_size=2)
        fig_new_overview.suptitle('Cleaned data')

        fig_absmag = get_abs_mag_chart(
            self.data, params['dist'], params['redshift'], params['absorbtion'], 
            add_kde=False, point_size=2)
    
        number_of_mc_experiments = 1000
        smoothing_bw = 0.1

        fig_zoom_density = get_density_chart(self.data, params, smoothing_bw)
        results = iterate_over_n_experiments(
            self.data, params, 
            number_of_mc_experiments, smoothing_bw
        )
        
        color_mean = results.mean()
        color_error = ((results - color_mean)**2).mean()**0.5

        fig_mc_visualsation = plot_histogrm_3x3(self.data, params, smoothing_bw)

        fig_result = plot_monte_carlo_results(results, color_mean, color_error, number_of_mc_experiments)

        data = {
            'method' : 'finding the maximum density',
            'filename' : self.file_path,
            'I mag level' : params['i_level'],
            '(V-I) color estimate' : color_mean,
            '(V-I) color estimate std' : color_error,
            'distance [in mag]' : params['dist'],
            'distance low' : params['dist'] - params['d_minus'],
            'distance high' : params['dist'] + params['d_plus'],
            'distance [in mpcs]' : self.dist_in_mpcs,
            'absorbtion (I)' : params['absorbtion'],
            'redshift (V-I)' : params['redshift'],
            'paramters' : {
                'number of m-c experiments' : number_of_mc_experiments,
                'espilon to smooth bw' : smoothing_bw,
                'vi_left' : params['vi_left'],
                'vi_right' : params['vi_right'],
                'i_level_low' : params['i_level_low'],
                'i_level_high' : params['i_level_high'],
                's_scaler' : params['s_scaler'],
                'mean_i_error' : params['mean_i_error'],
            }
        }

        figures = [fig_raw_overview, fig_new_overview, fig_absmag, 
                   fig_zoom_density, fig_mc_visualsation, fig_result]
        pdf = create_pdf_out_of_figures(figures)
        for fig in figures:
            plt.close(fig)

        return pdf, data


class CalculateAndSaveDensity(QThread):
    result_signal = pyqtSignal(FPDF, dict)

    def __init__(self, parent):
        super(CalculateAndSaveDensity, self).__init__()
        self.window = parent
        
    def run(self):
        pdf, data = self.window.calculate_density_approach()
        self.result_signal.emit(pdf, data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
