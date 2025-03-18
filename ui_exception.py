# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'exception.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_exceptWindow(object):
    def setupUi(self, exceptWindow):
        exceptWindow.setObjectName("exceptWindow")
        exceptWindow.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(exceptWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(exceptWindow)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(exceptWindow)
        QtCore.QMetaObject.connectSlotsByName(exceptWindow)

    def retranslateUi(self, exceptWindow):
        _translate = QtCore.QCoreApplication.translate
        exceptWindow.setWindowTitle(_translate("exceptWindow", "Ошибка"))
        self.label.setText(_translate("exceptWindow", "TextLabel"))

