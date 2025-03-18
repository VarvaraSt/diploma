# -*- coding: utf-8 -*-

import sys  # sys нужен для передачи argv в QApplication

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTableWidgetItem, \
    QFileDialog, QDialog, QHeaderView

import suda_alg
import ui_exception
import ui_main
import ui_noise


class TabularData:
    def __init__(self, df):
        self.df = df
        self.changed_columns = set()

    def get_unchanged_columns(self):
        return self.df[set(self.df) - self.changed_columns]

    def delete_column(self, col_name):
        self.df = self.df.drop(col_name, axis=1)

    def add_column(self, col_data, col_name):
        self.df[col_name] = col_data

    def add_noise(self, col_name, is_round=False, std=0.2, option=1, start=-10, end=10):
        self.changed_columns.add(col_name)
        if option == 1:
            noise = np.random.normal(0, scale=std * self.df[col_name].std(ddof=0), size=self.df.shape[0])
            self.df[col_name] = self.df[col_name] + noise
        elif option == 2:
            self.df[col_name] = self.df[col_name] * (1 + np.random.normal(0, 0.1, self.df.shape[0]))
        elif option == 3:
            noise = np.random.uniform(start, end, size=self.df.shape[0])
            self.df[col_name] = self.df[col_name] + noise

        if is_round:
            self.df[col_name] = np.round(self.df[col_name])

    def set_sensitive_attribute(self, col_name):
        self.changed_columns.add(col_name)


class ExceptWindow(QtWidgets.QDialog, ui_exception.Ui_exceptWindow):
    def __init__(self, parent=None, txt=''):
        super(ExceptWindow, self).__init__(parent)
        parent.setEnabled(False)
        self.Parent = parent
        self.setupUi(self)
        self.label.setText(txt)

    def closeEvent(self, event):
        self.Parent.setEnabled(True)


class AddNoiseWindow(QtWidgets.QDialog, ui_noise.Ui_AddNoise):
    def __init__(self, parent=None):
        super(AddNoiseWindow, self).__init__(parent)
        # parent.setEnabled(False)
        self.Parent = parent
        self.setupUi(self)
        self.change_option(1)
        self.btn1.clicked.connect(lambda state: self.change_option(1))
        self.btn2.clicked.connect(lambda state: self.change_option(2))
        self.btn3.clicked.connect(lambda state: self.change_option(3))
        self.option = 1
        self.buttonBox.accepted.connect(self.add_noise)

    def add_noise(self):
        self.Parent.add_noise(self.get_options())

    def get_options(self):
        return {
            'method': self.option,
            'int': bool(self.roundCheckBox.checkState()),
            'std': self.stdSpinBox.value(),
            'start': self.startSpinBox.value(),
            'end': self.endSpinBox.value(),
        }

    def change_option(self, n):
        self.option = n
        self.widget1.hide()
        self.widget2.hide()
        self.widget3.hide()
        if n == 1:
            self.widget1.show()
        elif n == 2:
            self.widget2.show()
        elif n == 3:
            self.widget3.show()

    def closeEvent(self, event):
        pass
        # self.Parent.setEnabled(True)


class mainWindow(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.data = None
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.deleteCol.clicked.connect(self.delete_columns)
        self.uniqueBtn.clicked.connect(self.show_unique_rows)
        self.noise.triggered.connect(self.open_noise_window)
        self.openAction.triggered.connect(self.openFile)
        self.sensitiveBtn.clicked.connect(self.set_sensitive_attribute)

        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # для быстрого тестирования
        # self.load_excel_data("example_800_csv.csv")

    def set_sensitive_attribute(self):
        try:
            selected = self.tableWidget.selectedRanges()
            n_rows = self.tableWidget.rowCount()
            for sel in selected:
                if sel.rowCount() != n_rows:
                    self.open_except_window("Выберите столбцы!")
                    print("Выберите столбцы!")
                    return
            columns = set()
            for index in self.tableWidget.selectedIndexes():
                columns.add(index.column())
            self.update_table(self.data.df)
            for col in columns:
                name = self.tableWidget.horizontalHeaderItem(col).text()
                self.highlight_columns([name], (219, 158, 211))
                self.data.set_sensitive_attribute(name)
        except Exception as ex:
            print(ex)
            self.open_except_window(str(ex))

    def get_table_data(self):
        data = []
        n_rows = self.tableWidget.rowCount()
        n_cols = self.tableWidget.columnCount()
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                row.append(self.tableWidget.item(i, j).text())
            data += [row]
        col_names = [self.tableWidget.horizontalHeaderItem(i).text() for i in range(n_cols)]
        df = pd.DataFrame(data, columns=col_names)
        return df

    def highlight_unique_vals(self, unique_vals):
        try:
            n_rows = self.tableWidget.rowCount()
            color = QColor(240, 202, 108, 127)
            for row in range(n_rows):
                if unique_vals[row]:
                    vals = unique_vals[row][1:-1].split(',')
                    for val in vals:
                        val = val.lstrip().lstrip('\'').rstrip('\'')
                        col_n = self.get_column_index_by_name(val)
                        if col_n != -1:
                            self.tableWidget.item(row, col_n).setBackground(color)
        except Exception as ex:
            print(ex)
            self.open_except_window(str(ex))

    def add_column(self, column_data, column_name):
        self.data.add_column(column_data, column_name)
        col_n = self.tableWidget.columnCount()
        self.tableWidget.setColumnCount(col_n + 1)
        self.tableWidget.setHorizontalHeaderItem(col_n, QTableWidgetItem(column_name))
        for row_index, row_val in enumerate(column_data):
            self.tableWidget.setItem(row_index, col_n, QTableWidgetItem(str(row_val)))

    def show_unique_rows(self):
        unique_vals = suda_alg.suda2(self.data.get_unchanged_columns())
        # self.add_column(unique_vals, "Уникальные значения")
        self.update_table(self.data.df)
        self.highlight_unique_vals(unique_vals)

    def get_column_index_by_name(self, col_name):
        n_cols = self.tableWidget.columnCount()
        for i in range(n_cols):
            if self.tableWidget.horizontalHeaderItem(i).text() == col_name:
                return i
        return -1

    def highlight_columns(self, column_names, color=(0, 0, 0)):
        n_rows = self.tableWidget.rowCount()
        color = QColor(color[0], color[1], color[2], 127)
        for col in column_names:
            col_n = self.get_column_index_by_name(col)
            for row in range(n_rows):
                self.tableWidget.item(row, col_n).setBackground(color)
            self.tableWidget.horizontalHeaderItem(col_n).setBackground(color)

    def add_noise(self, options=None):
        selected = self.tableWidget.selectedRanges()
        n_rows = self.tableWidget.rowCount()
        for sel in selected:
            if sel.rowCount() != n_rows:
                self.open_except_window("Выберите столбцы!")
                print("Выберите столбцы!")
                return
        columns = set()
        for index in self.tableWidget.selectedIndexes():
            columns.add(index.column())

        for col in columns:
            self.data.add_noise(
                self.tableWidget.horizontalHeaderItem(col).text(),
                is_round=options['int'],
                std=options['std'],
                option=options['method'],
                start=options['start'],
                end=options['end'],
            )
        self.update_table(self.data.df)

    def delete_columns(self):
        selected = self.tableWidget.selectedRanges()
        n_rows = self.tableWidget.rowCount()
        for sel in selected:
            if sel.rowCount() != n_rows:
                print("Выберите столбцы!")
                self.open_except_window("Выберите столбцы!")
                return
        columns = set()
        for index in self.tableWidget.selectedIndexes():
            columns.add(index.column())

        for col in sorted(columns, reverse=True):
            self.data.delete_column(self.tableWidget.horizontalHeaderItem(col).text())
            self.tableWidget.removeColumn(col)

    def update_table(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for row in df.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0,.2f}'.format(value).rstrip('0').rstrip('.')
                tableItem = QTableWidgetItem(str(value))
                self.tableWidget.setItem(row[0], col_index, tableItem)

        self.highlight_columns(self.data.changed_columns, (36, 255, 175))

    def load_excel_data(self, excel_file_dir, worksheet_name="Worksheet"):
        # df = pd.read_excel(excel_file_dir, worksheet_name)
        df = pd.read_csv(excel_file_dir)
        if df.size == 0:
            return

        df.fillna('', inplace=True)
        self.data = TabularData(df)
        self.update_table(df)

    def open_noise_window(self):
        window = AddNoiseWindow(self)
        window.show()

    def save(self):
        with open(self.path[:-4] + "_2.csv", 'w') as f:
            f.write(str(self.alpha))
            f.write("\n")
            f.write(str(self.beta))
            f.write("\n")
            f.write(str(self.demand))
            f.write("\n")
            f.write(str(self.limit))
            f.write("\n")
            f.write(str(self.losses))
            f.write("\n")
            f.write(str(self.bandwidth))
            f.write("\n")

    def openFile(self):
        path = self.FileDialog()
        if path != "":
            try:
                self.load_excel_data(path)
            except Exception as ex:
                print(ex)
                self.open_except_window("Неверный формат файла")

    def open_except_window(self, txt=""):
        window = ExceptWindow(self, txt=txt)
        window.show()

    def FileDialog(self, directory='', forOpen=True, fmt='', isFolder=False):
        dialog = QFileDialog()

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        if isFolder:
            dialog.setFileMode(QFileDialog.DirectoryOnly)
        else:
            dialog.setFileMode(QFileDialog.AnyFile)

        dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

        if fmt != '' and isFolder is False:
            dialog.setDefaultSuffix(fmt)
            dialog.setNameFilters([f'{fmt} (*.{fmt})'])

        if directory != '':
            dialog.setDirectory(str(directory))

        if dialog.exec_() == QDialog.Accepted:

            path = dialog.selectedFiles()[0]  # returns a list
            return path
        else:
            return ''


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = mainWindow()  # Создаём объект класса MyApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
