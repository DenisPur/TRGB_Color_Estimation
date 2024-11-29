import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from fpdf import FPDF

from PyQt5.QtCore import QThread, pyqtSignal, QLocale

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QDialog, QFileDialog, QMessageBox)

from src.main_window_ui import Ui_MainWindow
from src.clearing_ui import Ui_Dialog as Ui_Clearing_Dialog
from src.masking_ui import Ui_Dialog as Ui_Masking_Dialog

from src.infra import (
    read_file, check_available_columns, mask_based_on_cells_density, 
    create_pdf_out_of_figures, get_kde)
from src.main_charts import ( 
    overview_cmd_field_chart, clearing_chart, rectangular_masking_chart, 
    cells_hist_chart, cells_masking_chart, abs_mag_cmd_chart)
from src.branch_approximation import (
    marking_and_approximating, branch_approximation_graph)
from src.denisty_approximation import (
    slice_density_graph, choosing_low_density_regions)
from src.monte_carlo import (
    plot_histogrm_3x3, iterate_over_n_experiments, plot_monte_carlo_results)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setLocale(QLocale("en_US"))

        # UI set up
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.main_tabs.setTabEnabled(2, False)
        self.ui.main_tabs.setTabEnabled(1, False)

        # first tab
        self.ui.button_chose_file.clicked.connect(self.open_file)
        self.ui.button_reload.clicked.connect(self.reload_file)

        self.ui.button_clearing.clicked.connect(self.open_clearing_dialog)
        self.ui.button_view_clearing.clicked.connect(self.view_clearing_result)
        self.ui.button_masking.clicked.connect(self.open_masking_dialog)
        self.ui.button_view_masking.clicked.connect(self.view_masking_result)
        self.ui.button_view_cmd_field.clicked.connect(self.view_cmd_field)
        self.ui.button_export_csv.clicked.connect(self.export_cmd)

        self.ui.enter_mpcs.valueChanged.connect(self.mpcs_changed)
        self.ui.enter_mags.valueChanged.connect(self.mags_changed)
        self.ui.enter_coef_1.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.enter_coef_2.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.enter_b_minus_v.valueChanged.connect(self.recalculate_reddening_labels)
        self.ui.button_view_abs_cmd.clicked.connect(self.view_chart_manager)

        # second tab
        self.ui.button_view_spatial.clicked.connect(self.view_branch)
        self.ui.button_save_1.clicked.connect(self.save_branch_approx)

        # third tab
        self.ui.button_view_density.clicked.connect(self.view_density)
        self.ui.button_save_2.clicked.connect(self.calculate_and_save_density_manager) 

        # var
        self.mask_used = False
        self.dist = 0
        self.dist_in_mpcs = 0
        self.input_folder = None
        self.output_folder = None

        # pyqt5 bug fix
        self.ui.group_view_export_changes.setEnabled(False)
        self.ui.group_view_cmd_abs.setEnabled(False)


    def set_input_folder(self, folder: str):
        self.input_folder = folder

    def set_output_folder(self, folder: str):
        self.output_folder = folder

    def save_json(self, data: dict):
        if self.output_folder is not None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save JSON', self.output_folder + '/' + self.filename + '.json')
        else:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save JSON')

        if filename != '':
            with open(filename, "w") as out_file:
                json.dump(data, out_file, indent=4)

    def save_pdf(self, pdf: FPDF):
        if self.output_folder is not None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save PDF', self.output_folder + '/' + self.filename + '.pdf')
        else:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save PDF')
    
        if filename != '':
            pdf.output(filename)


    def open_file(self):
        widget = QWidget(parent=self)

        if self.input_folder is not None:
            self.file_path, _ = QFileDialog.getOpenFileName(widget, 'Open File', self.input_folder)
        else:
            self.file_path, _ = QFileDialog.getOpenFileName(widget, 'Open File')
        
        if self.file_path:
            try:
                self.load_file(self.file_path)
                self.filename = self.file_path.split('/')[-1].removesuffix('.csv')
            except pd.errors.ParserError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Invalid file")
                msg.setInformativeText('Chose .csv file with proper column names. Mandatory column names: ["x", "y", "mag_v", "err_v", "mag_i", "err_i"]')
                msg.setWindowTitle("Error")
                msg.exec_()

    def reload_file(self):
        self.load_file(self.file_path)

    def load_file(self, filename):
        self.data = read_file(filename)
        self.data_raw = self.data.copy()
        self.ui.label_filename.setText(filename)

        self.ui.group_clearing.setEnabled(True)
        self.ui.group_masking.setEnabled(True)
        self.ui.group_view_export_changes.setEnabled(True)
        self.ui.group_distance.setEnabled(True)
        self.ui.group_reddening.setEnabled(True)
        self.ui.group_view_cmd_abs.setEnabled(True)
        self.ui.button_reload.setEnabled(True)

        self.ui.button_view_clearing.setEnabled(False)
        self.ui.button_view_masking.setEnabled(False)

        self.ui.main_tabs.setTabEnabled(1, True)
        self.ui.main_tabs.setTabEnabled(2, True)

        self.mask_used = False
        self.boundries_for_overview = self.save_boundries_for_overview_chart()
        plt.close('all')

    def save_boundries_for_overview_chart(self) -> list[float]:
        color_low = np.percentile(self.data['color_vi'].values, 1.5)
        color_high = np.percentile(self.data['color_vi'].values, 98.5)

        x_ax_low = self.data['x'].values.min() - 0.05
        x_ax_high = self.data['x'].values.max() + 0.05
        y_ax_low = self.data['y'].values.min() - 0.05
        y_ax_high = self.data['y'].values.max() + 0.05
        return [color_low, color_high, x_ax_low, x_ax_high, y_ax_low, y_ax_high]


    def view_cmd_field(self):
        fig_raw = overview_cmd_field_chart(self.data, self.boundries_for_overview)
        fig_raw.show()

    def export_cmd(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Export CSV')
        if filename != '':
            with open(filename, "w") as out_file:
                self.data.to_csv(filename, index=None)


    def open_clearing_dialog(self):
        window = QDialog(parent=self)
        window.ui = Ui_Clearing_Dialog()
        window.ui.setupUi(window)
        
        columns_availability = check_available_columns(self.data)
        checkers_list = [
            window.ui.check_type, window.ui.check_mag, window.ui.check_snr,
            window.ui.check_sharp, window.ui.check_flag, window.ui.check_crowd]
        column_names = ['type', 'mag', 'snr', 'sharp', 'flag', 'crowd']
        for checker, column in zip(checkers_list, column_names):
            checker.setChecked(columns_availability[column])
            checker.setEnabled(columns_availability[column])

        def mark_clean_rows() -> pd.Series:
            marking = (self.data['mag_v'] == self.data['mag_v']) # bruh

            if window.ui.check_type.isChecked():
                marking *= (self.data['type'] <= window.ui.enter_type.value())
            if window.ui.check_mag.isChecked():
                marking *= ((self.data['mag_v'] < window.ui.enter_mag.value()) 
                            & (self.data['mag_i'] < window.ui.enter_mag.value()))
            if window.ui.check_snr.isChecked():
                marking *= ((self.data['snr_v'] >= window.ui.enter_snr.value())
                            & (self.data['snr_i'] >= window.ui.enter_snr.value()))
            if window.ui.check_sharp.isChecked():
                marking *= ((self.data['sharp_v'] + self.data['sharp_i'])**2 
                            <= window.ui.enter_sharp.value())
            if window.ui.check_flag.isChecked():
                marking *= ((self.data['flag_v'] <= window.ui.enter_flag.value())
                            & (self.data['flag_i'] <= window.ui.enter_flag.value()))
            if window.ui.check_crowd.isChecked():
                marking *= ((self.data['crowd_v'] + self.data['crowd_i']) 
                            <= window.ui.enter_crowd.value())
            return marking

        def preview():
            marking = mark_clean_rows()
            clean_data = self.data[marking]
            dirty_data = self.data[~marking]
            fig = clearing_chart(clean=clean_data, dirty=dirty_data)
            fig.show()

        def saving():
            marking = mark_clean_rows()
            clean_data = self.data[marking]
            dirty_data = self.data[~marking]
            self.fig_clear = clearing_chart(clean=clean_data, dirty=dirty_data)
            self.data = self.data[marking]
            self.ui.button_view_clearing.setEnabled(True)
            self.ui.group_masking.setEnabled(True)
            self.ui.group_distance.setEnabled(True)

        window.ui.button_preview.clicked.connect(preview)
        window.ui.buttonBox.accepted.connect(saving)
        window.show()

    def view_clearing_result(self):
        self.fig_clear.show()


    def open_masking_dialog(self):
        window = QDialog(parent=self)
        window.ui = Ui_Masking_Dialog()
        window.ui.setupUi(window)

        x_min, x_max = self.data['x'].min(), self.data['x'].max()
        y_min, y_max = self.data['y'].min(), self.data['y'].max()

        window.ui.enter_x_left.setRange(x_min, x_max)
        window.ui.enter_x_right.setRange(x_min, x_max)
        window.ui.enter_x_left.setValue(x_min)
        window.ui.enter_x_right.setValue(x_max)

        window.ui.enter_y_bottom.setRange(y_min, y_max)
        window.ui.enter_y_top.setRange(y_min, y_max)
        window.ui.enter_y_bottom.setValue(y_min)
        window.ui.enter_y_top.setValue(y_max)

        def apply_rectangular_mask() -> tuple[list[float], pd.Series]:
            x_left = window.ui.enter_x_left.value()
            x_right = window.ui.enter_x_right.value()
            x_left, x_right = sorted([x_left, x_right])
            y_bottom = window.ui.enter_y_bottom.value()
            y_top = window.ui.enter_y_top.value()
            y_bottom, y_top = sorted([y_bottom, y_top])
            borders = [x_left, x_right, y_bottom, y_top]
            take_external = window.ui.check_inverse_rect_select.checkState() == 2

            eps = 0.1
            masking = ((self.data['x'] > x_left - eps) 
                       & (self.data['x'] < x_right + eps)
                       & (self.data['y'] > y_bottom - eps)
                       & (self.data['y'] < y_top + eps))
            if take_external:
                masking = ~ masking
            return borders, masking

        def view_chosen_rectangle():
            borders, masking = apply_rectangular_mask()
            fig = rectangular_masking_chart(self.data, masking, borders)
            fig.show()

        def preview_cells():
            number_of_cells = window.ui.enter_number_of_cells.value()
            fig = cells_hist_chart(self.data, number_of_cells)
            fig.show()

        def view_chosen_cells():
            number_of_cells = window.ui.enter_number_of_cells.value()
            threshold = window.ui.enter_threshold.value() / 100
            take_external = window.ui.check_inverse_dens_select.checkState() == 2
            mask = mask_based_on_cells_density(self.data, number_of_cells, threshold)
            if take_external:
                mask = ~ mask
            fig = cells_masking_chart(self.data, number_of_cells, mask)
            fig.show()

        def saving():
            self.mask_used = True
            if window.ui.tabWidget.currentIndex() == 0:
                borders, mask = apply_rectangular_mask()
                self.fig_mask = rectangular_masking_chart(self.data, mask, borders)
            else:
                number_of_cells = window.ui.enter_number_of_cells.value()
                threshold = window.ui.enter_threshold.value() / 100
                take_external = window.ui.check_inverse_dens_select.checkState() == 2
                mask = mask_based_on_cells_density(self.data, number_of_cells, threshold) 
                if take_external:
                    mask = ~ mask
                self.fig_mask = cells_masking_chart(self.data, number_of_cells, mask)

            self.data = self.data[mask]
            self.ui.button_view_masking.setEnabled(True)
            self.ui.group_distance.setEnabled(True)

        window.ui.slider_threshold.valueChanged.connect(window.ui.enter_threshold.setValue)
        window.ui.enter_threshold.valueChanged.connect(window.ui.slider_threshold.setValue)

        window.ui.button_preview_rect.clicked.connect(view_chosen_rectangle)
        window.ui.button_preview_cells.clicked.connect(preview_cells)
        window.ui.button_preview_dens.clicked.connect(view_chosen_cells)

        window.ui.buttonBox.accepted.connect(saving)
        window.show()

    def view_masking_result(self):
        self.fig_mask.show()


    def mpcs_changed(self, value: float):
        if (value > 0):
            dist_mags = 5 * np.log10(value * 1e5)
        else:
            dist_mags = 0
        self.dist_in_mpcs = value
        self.dist = dist_mags
        
        self.ui.enter_mags.blockSignals(True)
        self.ui.enter_mags.setValue(dist_mags)
        self.ui.enter_mags.blockSignals(False)
        
    def mags_changed(self, value: float):
        dist_mpcs = 10**(value / 5) / 1e5
        self.dist = value
        self.dist_in_mpcs = dist_mpcs
        
        self.ui.enter_mpcs.blockSignals(True)
        self.ui.enter_mpcs.setValue(dist_mpcs)
        self.ui.enter_mpcs.blockSignals(False)


    def recalculate_reddening_labels(self):
        extinction = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_1.value()
        absorbtion = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_2.value()
        self.ui.label_extinction.setText(f'{extinction:1.4}')
        self.ui.label_absorbtion.setText(f'{absorbtion:1.4}')
    
    def get_extinction_absorbtion(self) -> tuple[float, float]:
        if (self.ui.tabs_reddening.currentIndex() == 0):
            extinction = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_1.value()
            absorbtion = self.ui.enter_b_minus_v.value() * self.ui.enter_coef_2.value()
        else:
            extinction = self.ui.enter_extinction.value()
            absorbtion = self.ui.enter_absorbtion.value()
        return extinction, absorbtion
    
    
    def view_chart_manager(self):
        add_kde = self.ui.check_add_kde.checkState() == 2
        extinction, absorbtion = self.get_extinction_absorbtion()
        if add_kde:
            self.thread = CalculateKDE(self.data, self.dist, extinction, absorbtion)
            self.thread.started.connect(self.show_abs_cmd_chart_processing)
            self.thread.finished.connect(self.show_abs_cmd_chart_done)
            self.thread.result_signal.connect(self.view_abs_cmd_chart)
            self.thread.start()
        else: 
            self.view_abs_cmd_chart(kde=None)

    def view_abs_cmd_chart(self, kde):
        extinction, absorbtion = self.get_extinction_absorbtion()
        self.data['abs_color_vi'] = self.data['color_vi'] - extinction
        self.data['abs_mag_i'] = self.data['mag_i'] - self.dist - absorbtion
        fig = abs_mag_cmd_chart(self.data, kde=kde)
        fig.show()       

    def show_abs_cmd_chart_processing(self):
        self.ui.button_view_abs_cmd.setText("Processing ⏱")
        self.ui.check_add_kde.setEnabled(False)
        self.ui.button_view_abs_cmd.setEnabled(False)

    def show_abs_cmd_chart_done(self):
        self.ui.button_view_abs_cmd.setText('View in absolute magnitudes')
        self.ui.check_add_kde.setEnabled(True)
        self.ui.button_view_abs_cmd.setEnabled(True)

    
    def pack_all_branch_approx_parameters_in_dict(self):
        extinction, absorbtion = self.get_extinction_absorbtion()
        params = {
            'dist':self.dist, 
            'extinction':extinction,
            'absorbtion':absorbtion, 
            'i_level':self.ui.enter_level.value(),
            'vi_left':self.ui.enter_vi_left.value(),
            'vi_right':self.ui.enter_vi_right.value(), 
            'i_left':self.ui.enter_i_left.value(), 
            'i_right':self.ui.enter_i_right.value(), 
            'p_chosen':self.ui.enter_p.value(),
            'd_minus':self.ui.enter_d_minus.value(), 
            'd_plus':self.ui.enter_d_plus.value()}
        return params

    def view_branch(self):
        params = self.pack_all_branch_approx_parameters_in_dict()
        try:
            chosen_bool, inliers_bool, f_approx, f_std = marking_and_approximating(self.data, params)
            fig = branch_approximation_graph(self.data, params, chosen_bool, inliers_bool, f_approx, f_std)
            fig.show()
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Invalid borders")
            msg.setInformativeText('At least 1 star should be in chosen diapason.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def save_branch_approx(self):
        params = self.pack_all_branch_approx_parameters_in_dict()
        
        fig_raw_overview = overview_cmd_field_chart(self.data_raw, self.boundries_for_overview)
        fig_raw_overview.suptitle('Raw data')
        fig_new_overview = overview_cmd_field_chart(self.data, self.boundries_for_overview)
        fig_new_overview.suptitle('Cleaned data')

        self.data['abs_color_vi'] = self.data['color_vi'] - params['extinction']
        self.data['abs_mag_i'] = self.data['mag_i'] - self.dist - params['absorbtion']
        fig_absmag_1 = abs_mag_cmd_chart(self.data, kde=False)
        kde = get_kde(self.data, self.dist, params['extinction'], params['absorbtion'])
        fig_absmag_2 = abs_mag_cmd_chart(self.data, kde=kde)

        chosen_bool, inliers_bool, f_approx, f_std = marking_and_approximating(self.data, params)
        fig_result = branch_approximation_graph(self.data, params, chosen_bool, inliers_bool, f_approx, f_std)
        
        d_m = params['d_minus']
        d_p = params['d_plus']
        i_mag = params['i_level']
        estimate = f_approx(i_mag)
        estimate_high = f_approx(i_mag+d_p) - f_std(i_mag+d_p)
        estimate_low = f_approx(i_mag-d_m) + f_std(i_mag-d_m)
        output_data = {
            'method' : 'approximation of a branch by a parabola',
            'filename' : self.file_path,
            'I mag level' : params['i_level'],
            'mask used' : self.mask_used,
            '(V-I) color estimate' : estimate,
            '(V-I) color estimate low' : estimate_low,
            '(V-I) color estimate high' : estimate_high,
            'distance [in mag]' : params['dist'],
            'distance low' : params['dist'] - d_m,
            'distance high' : params['dist'] + d_p,
            'distance [in mpcs]' : self.dist_in_mpcs,
            'absorbtion (I)' : params['absorbtion'],
            'extinction (V-I)' : params['extinction'],
            'paramters' : {
                'vi_left': params['vi_left'],
                'vi_right': params['vi_right'], 
                'i_top': params['i_left'], 
                'i_bottom': params['i_right'], 
                'p_chosen': params['p_chosen'],
            }
        }

        self.save_json(output_data)

        figures = [fig_raw_overview, fig_new_overview, fig_absmag_1, fig_absmag_2, fig_result]
        pdf = create_pdf_out_of_figures(figures)
        for fig in figures:
            plt.close(fig)
            
        self.save_pdf(pdf)


    def pack_all_density_parameters_in_dict(self):
        extinction, absorbtion = self.get_extinction_absorbtion()
        params = {
            'dist':self.dist, 
            'extinction':extinction,
            'absorbtion':absorbtion, 
            'i_level':self.ui.enter_level_2.value(),
            'd_minus':self.ui.enter_d_minus_2.value(), 
            'd_plus':self.ui.enter_d_plus_2.value(),
            'vi_left':self.ui.enter_vi_left_2.value(),
            'vi_right':self.ui.enter_vi_right_2.value(),
            's_scaler':self.ui.enter_s.value()}
        chosen_bool, i_level_low, i_level_high, mean_i_error, mean_color_error = choosing_low_density_regions(self.data, params)
        params = {
            **params,
            'chosen_bool': chosen_bool,
            'i_level_low': i_level_low,
            'i_level_high': i_level_high,
            'mean_i_error': mean_i_error,
            'mean_color_error': mean_color_error}
        return params

    def view_density(self):
        params = self.pack_all_density_parameters_in_dict()
        try:
            fig = slice_density_graph(self.data, params, smoothing_bw=0.1)
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
        self.thread.started.connect(self.show_density_processing)
        self.thread.finished.connect(self.show_density_done)
        self.thread.result_signal.connect(self.save_density_results)
        self.thread.start()

    def show_density_processing(self):
        self.ui.button_save_2.setText("Processing ⏱")
        self.ui.button_save_2.setEnabled(False)

    def show_density_done(self):
        self.ui.button_save_2.setText('Calculate and save (json + pdf) [may take some time ⏱]')
        self.ui.button_save_2.setEnabled(True)

    def save_density_results(self, pdf: FPDF, data: dict):
        self.save_json(data)
        self.save_pdf(pdf)

    def calculate_density_approach(self) -> tuple[FPDF, dict]:
        params = self.pack_all_density_parameters_in_dict()

        fig_raw_overview = overview_cmd_field_chart(self.data_raw, self.boundries_for_overview)
        fig_raw_overview.suptitle('Raw data')

        fig_new_overview = overview_cmd_field_chart(self.data, self.boundries_for_overview)
        fig_new_overview.suptitle('Cleaned data')

        self.data['abs_color_vi'] = self.data['color_vi'] - params['extinction']
        self.data['abs_mag_i'] = self.data['mag_i'] - self.dist - params['absorbtion']
        fig_absmag_1 = abs_mag_cmd_chart(self.data, kde=False)
        kde = get_kde(self.data, self.dist, params['extinction'], params['absorbtion'])
        fig_absmag_2 = abs_mag_cmd_chart(self.data, kde=kde)
    
        number_of_mc_experiments = self.ui.enter_n_exp.value()
        smoothing_bw = self.ui.enter_eps.value()

        fig_zoom_density = slice_density_graph(self.data, params, smoothing_bw)
        results, num_of_stars = iterate_over_n_experiments(
            self.data, params, 
            number_of_mc_experiments, smoothing_bw)
        
        color_mean = results.mean()
        color_error = ((results - color_mean)**2).mean()**0.5

        fig_mc_visualsation = plot_histogrm_3x3(self.data, params, smoothing_bw)
        fig_result = plot_monte_carlo_results(results, color_mean, color_error, number_of_mc_experiments)

        output_data = {
            'method' : 'finding the maximum density',
            'filename' : self.file_path,
            'I mag level' : params['i_level'],
            'mask used' : self.mask_used,
            '(V-I) color estimate' : color_mean,
            '(V-I) color estimate std' : color_error,
            'distance [in mag]' : params['dist'],
            'distance low' : params['dist'] - params['d_minus'],
            'distance high' : params['dist'] + params['d_plus'],
            'distance [in mpcs]' : self.dist_in_mpcs,
            'absorbtion (I)' : params['absorbtion'],
            'extinction (V-I)' : params['extinction'],
            'paramters' : {
                'mean number of stars' : num_of_stars,
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

        figures = [fig_raw_overview, fig_new_overview, fig_absmag_1, fig_absmag_2, 
                   fig_zoom_density, fig_mc_visualsation, fig_result]
        pdf = create_pdf_out_of_figures(figures)
        for fig in figures:
            plt.close(fig)

        return pdf, output_data


class CalculateAndSaveDensity(QThread):
    result_signal = pyqtSignal(FPDF, dict)

    def __init__(self, parent: MainWindow):
        super(CalculateAndSaveDensity, self).__init__()
        self.window = parent
        
    def run(self):
        pdf, data = self.window.calculate_density_approach()
        self.result_signal.emit(pdf, data)


class CalculateKDE(QThread):
    result_signal = pyqtSignal(tuple)

    def __init__(self, 
            data: dict, dist:float, 
            extinction: float, absorbtion: float):
        super(CalculateKDE, self).__init__()
        self.data = data
        self.dist = dist 
        self.extinction = extinction 
        self.absorbtion = absorbtion        

    def run(self):
        kde = get_kde(self.data, self.dist, self.extinction, self.absorbtion)
        self.result_signal.emit(kde)
