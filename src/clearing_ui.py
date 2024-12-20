# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_files/clearing_criteria.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(360, 360)
        Dialog.setWindowTitle("Photometry cleaning criteria")
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.text_instructions = QtWidgets.QTextBrowser(Dialog)
        self.text_instructions.setObjectName("text_instructions")
        self.verticalLayout.addWidget(self.text_instructions)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.enter_type = QtWidgets.QSpinBox(Dialog)
        self.enter_type.setProperty("value", 4)
        self.enter_type.setObjectName("enter_type")
        self.gridLayout.addWidget(self.enter_type, 0, 3, 1, 1)
        self.label_type = QtWidgets.QLabel(Dialog)
        self.label_type.setObjectName("label_type")
        self.gridLayout.addWidget(self.label_type, 0, 2, 1, 1)

        self.enter_mag = QtWidgets.QDoubleSpinBox(Dialog)
        self.enter_mag.setDecimals(1)
        self.enter_mag.setProperty("value", 50.0)
        self.enter_mag.setObjectName("enter_mag")
        self.gridLayout.addWidget(self.enter_mag, 1, 3, 1, 1)
        self.label_mag = QtWidgets.QLabel(Dialog)
        self.label_mag.setEnabled(True)
        self.label_mag.setObjectName("label_mag")
        self.gridLayout.addWidget(self.label_mag, 1, 2, 1, 1)

        self.enter_snr = QtWidgets.QDoubleSpinBox(Dialog)
        self.enter_snr.setDecimals(1)
        self.enter_snr.setMaximum(1000.0)
        self.enter_snr.setSingleStep(0.1)
        self.enter_snr.setProperty("value", 4.0)
        self.enter_snr.setObjectName("enter_snr")
        self.gridLayout.addWidget(self.enter_snr, 2, 3, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.label_snr = QtWidgets.QLabel(Dialog)
        self.label_snr.setObjectName("label_snr")
        self.gridLayout.addWidget(self.label_snr, 2, 2, 1, 1)

        self.enter_sharp = QtWidgets.QDoubleSpinBox(Dialog)
        self.enter_sharp.setDecimals(3)
        self.enter_sharp.setSingleStep(0.001)
        self.enter_sharp.setProperty("value", 0.075)
        self.enter_sharp.setObjectName("enter_sharp")
        self.gridLayout.addWidget(self.enter_sharp, 3, 3, 1, 1)
        self.label_sharp = QtWidgets.QLabel(Dialog)
        self.label_sharp.setObjectName("label_sharp")
        self.gridLayout.addWidget(self.label_sharp, 3, 2, 1, 1)

        self.enter_flag = QtWidgets.QSpinBox(Dialog)
        self.enter_flag.setProperty("value", 2)
        self.enter_flag.setObjectName("enter_flag")
        self.gridLayout.addWidget(self.enter_flag, 4, 3, 1, 1)
        self.label_flag = QtWidgets.QLabel(Dialog)
        self.label_flag.setEnabled(True)
        self.label_flag.setObjectName("label_flag")
        self.gridLayout.addWidget(self.label_flag, 4, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 4, 1, 1)

        self.enter_crowd = QtWidgets.QDoubleSpinBox(Dialog)
        self.enter_crowd.setDecimals(1)
        self.enter_crowd.setSingleStep(0.1)
        self.enter_crowd.setProperty("value", 0.8)
        self.enter_crowd.setObjectName("enter_crowd")
        self.gridLayout.addWidget(self.enter_crowd, 5, 3, 1, 1)
        self.label_crowd = QtWidgets.QLabel(Dialog)
        self.label_crowd.setObjectName("label_crowd")
        self.gridLayout.addWidget(self.label_crowd, 5, 2, 1, 1)

        self.check_type = QtWidgets.QCheckBox(Dialog)
        self.check_type.setEnabled(True)
        self.check_type.setText("")
        self.check_type.setChecked(True)
        self.check_type.setObjectName("check_type")
        self.gridLayout.addWidget(self.check_type, 0, 1, 1, 1)

        self.check_mag = QtWidgets.QCheckBox(Dialog)
        self.check_mag.setText("")
        self.check_mag.setChecked(True)
        self.check_mag.setObjectName("check_mag")
        self.gridLayout.addWidget(self.check_mag, 1, 1, 1, 1)

        self.check_snr = QtWidgets.QCheckBox(Dialog)
        self.check_snr.setText("")
        self.check_snr.setChecked(True)
        self.check_snr.setObjectName("check_snr")
        self.gridLayout.addWidget(self.check_snr, 2, 1, 1, 1)

        self.check_sharp = QtWidgets.QCheckBox(Dialog)
        self.check_sharp.setText("")
        self.check_sharp.setChecked(True)
        self.check_sharp.setObjectName("check_sharp")
        self.gridLayout.addWidget(self.check_sharp, 3, 1, 1, 1)

        self.check_flag = QtWidgets.QCheckBox(Dialog)
        self.check_flag.setText("")
        self.check_flag.setChecked(True)
        self.check_flag.setObjectName("check_flag")
        self.gridLayout.addWidget(self.check_flag, 4, 1, 1, 1)

        self.check_crowd = QtWidgets.QCheckBox(Dialog)
        self.check_crowd.setEnabled(True)
        self.check_crowd.setText("")
        self.check_crowd.setCheckable(True)
        self.check_crowd.setChecked(True)
        self.check_crowd.setObjectName("check_crowd")
        self.gridLayout.addWidget(self.check_crowd, 5, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.button_preview = QtWidgets.QPushButton(Dialog)
        self.button_preview.setObjectName("button_preview")
        self.verticalLayout.addWidget(self.button_preview)
        
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        self.check_type.toggled['bool'].connect(self.enter_type.setEnabled) # type: ignore
        self.check_mag.toggled['bool'].connect(self.enter_mag.setEnabled) # type: ignore
        self.check_snr.toggled['bool'].connect(self.enter_snr.setEnabled) # type: ignore
        self.check_sharp.toggled['bool'].connect(self.enter_sharp.setEnabled) # type: ignore
        self.check_flag.toggled['bool'].connect(self.enter_flag.setEnabled) # type: ignore
        self.check_crowd.toggled['bool'].connect(self.enter_crowd.setEnabled) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        self.text_instructions.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Cantarell\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Если какие-то столбцы отсутсвуют в файле, то они буду недоступны для редактирования.</p></body></html>"))
        self.label_mag.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">mag <span style=\" vertical-align:sub;\">V</span> &amp; mag <span style=\" vertical-align:sub;\">I</span> &lt; </p></body></html>"))
        self.label_type.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">type ≤ </p></body></html>"))
        self.label_crowd.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">crowd <span style=\" vertical-align:sub;\">V</span> + crowd <span style=\" vertical-align:sub;\">I</span> ≤ </p></body></html>"))
        self.label_flag.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">flag <span style=\" vertical-align:sub;\">V</span> &amp; flag <span style=\" vertical-align:sub;\">I</span> ≤ </p></body></html>"))
        self.label_snr.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">s/n ratio ≥ </p></body></html>"))
        self.label_sharp.setText(_translate("Dialog", "<html><head/><body><p align=\"right\">( sharp <span style=\" vertical-align:sub;\">V</span> + sharp <span style=\" vertical-align:sub;\">I </span>)<span style=\" vertical-align:super;\">2</span> ≤ </p></body></html>"))
        self.button_preview.setText(_translate("Dialog", "Preview"))
