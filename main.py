# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
import ast
import random
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
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(float)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype("boolean")
            else:
                df[col] = df[col].astype("object")
        self.df = df
        self.changed_columns = set()
        self.sensitive_columns = set()
        self.unique_values = None

    def get_unchanged_columns(self):
        return self.df[set(self.df) - self.changed_columns]

    def delete_column(self, col_name):
        self.changed_columns.discard(col_name)
        self.sensitive_columns.discard(col_name)
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
        self.sensitive_columns.add(col_name)

    def set_changed_column(self, col_name):
        self.changed_columns.add(col_name)

    def set_unique_values(self, unique_values):
        self.unique_values = unique_values

    def suppress_unique_values(self, col_name):
        def safe_parse(val):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return []
            except:
                return []

        if self.unique_values is None:
            return
        for idx in self.df.index:
            allowed = safe_parse(self.unique_values.loc[idx])
            if col_name in allowed:
                self.df.at[idx, col_name] = np.nan

    def microaggregate(self, col_name, round_digits=0):
        group_cols = [col for col in self.df.columns if col != col_name and col not in self.sensitive_columns]

        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            agg_func = lambda x: round(x.mean(), round_digits)
        else:
            agg_func = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

        aggregated = self.df.groupby(group_cols, dropna=False)[col_name].transform(agg_func)
        self.df[col_name] = aggregated

    def post_randomize(self, col_name, top_n=3, seed=None):
        if seed is not None:
            random.seed(seed)
        top_values = self.df[col_name].value_counts().nlargest(top_n).index.tolist()

        def safe_parse(val):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return []

        for idx in self.df.index:
            allowed_cols = safe_parse(self.unique_values.loc[idx])

            if col_name in allowed_cols:
                replacement = random.choice(top_values)
                self.df.at[idx, col_name] = replacement

    def generalize(self, col_name, df_hierarchy):
        replacement_dict = dict(zip(df_hierarchy.iloc[:, 0], df_hierarchy.iloc[:, 1]))
        self.df[col_name] = self.df[col_name].replace(replacement_dict)

    def check_l_diversity(self, l=1):
        if l == 1 or not len(self.sensitive_columns):
            return set()
        quasi_columns = [col for col in self.df.columns if col not in self.sensitive_columns]

        grouped = self.df.groupby(quasi_columns, group_keys=False)
        violating_indices = set()
        for _, group in grouped:
            unique_sensitive = group[self.sensitive_columns].drop_duplicates()
            if len(unique_sensitive) < l:
                violating_indices.update(group.index)

        return violating_indices

    def save(self, path):
        self.df.to_csv(path, index=False)


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
        self.hierarchy_data = None
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.deleteCol.clicked.connect(self.delete_columns)
        self.uniqueBtn.clicked.connect(self.show_unique_rows)
        self.noise.triggered.connect(self.open_noise_window)
        self.openAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveFile)
        self.sensitiveBtn.clicked.connect(self.set_sensitive_attribute)
        self.suppresion.triggered.connect(self.suppress_values)
        self.microaggregation.triggered.connect(self.microaggregate)
        self.randomization.triggered.connect(self.post_randomize)
        self.generalization.triggered.connect(self.generalize)

        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # для быстрого тестирования
        # self.load_excel_data("example_800_csv.csv")

    def set_sensitive_attribute(self):
        try:
            columns = self.get_selected_columns()
            self.update_table(self.data.df)
            for col in columns:
                name = self.tableWidget.horizontalHeaderItem(col).text()
                self.highlight_columns([name], (219, 158, 211))
                self.data.set_sensitive_attribute(name)
        except Exception as ex:
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

    def highlight_unique_vals(self, unique_vals, violations):
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
                if row in violations:
                    for col_name in self.data.sensitive_columns:
                        col_n = self.get_column_index_by_name(col_name)
                        if col_n != -1:
                            self.tableWidget.item(row, col_n).setBackground(color)
            self.data.set_unique_values(unique_vals)
        except Exception as ex:
            self.open_except_window(str(ex))

    def add_column(self, column_data, column_name):
        try:
            self.data.add_column(column_data, column_name)
            col_n = self.tableWidget.columnCount()
            self.tableWidget.setColumnCount(col_n + 1)
            self.tableWidget.setHorizontalHeaderItem(col_n, QTableWidgetItem(column_name))
            for row_index, row_val in enumerate(column_data):
                self.tableWidget.setItem(row_index, col_n, QTableWidgetItem(str(row_val)))
        except Exception as ex:
            self.open_except_window(str(ex))

    def show_unique_rows(self):
        unique_vals = suda_alg.suda2(self.data.get_unchanged_columns(), k=self.k_val.value() - 1)
        violations = self.data.check_l_diversity(l=self.l_val.value())
        self.update_table(self.data.df)
        if unique_vals is not None:
            self.highlight_unique_vals(unique_vals, violations)

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

    def get_selected_columns(self):
        selected = self.tableWidget.selectedRanges()
        n_rows = self.tableWidget.rowCount()
        if not len(selected):
            self.open_except_window("Выберите столбцы!")
            return
        for sel in selected:
            if sel.rowCount() != n_rows:
                self.open_except_window("Выберите столбцы!")
                print("Выберите столбцы!")
                return
        columns = set()
        for index in self.tableWidget.selectedIndexes():
            columns.add(index.column())
        return columns

    def add_noise(self, options=None):
        try:
            columns = self.get_selected_columns()

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
        except Exception as ex:
            self.open_except_window(str(ex))

    def suppress_values(self):
        try:
            columns = self.get_selected_columns()

            for col in columns:
                self.data.suppress_unique_values(self.tableWidget.horizontalHeaderItem(col).text())
            self.update_table(self.data.df)
        except Exception as ex:
            self.open_except_window(str(ex))

    def microaggregate(self):
        try:
            columns = self.get_selected_columns()

            for col in columns:
                self.data.microaggregate(self.tableWidget.horizontalHeaderItem(col).text())
            self.update_table(self.data.df)
        except Exception as ex:
            self.open_except_window(str(ex))

    def post_randomize(self):
        try:
            columns = self.get_selected_columns()

            for col in columns:
                self.data.post_randomize(self.tableWidget.horizontalHeaderItem(col).text())
            self.update_table(self.data.df)
        except Exception as ex:
            self.open_except_window(str(ex))

    def generalize(self):
        try:
            self.open_hierarchy_file()
            if self.hierarchy_data is None:
                self.open_except_window("Ошибка чтения файла!")
                return

            columns = self.get_selected_columns()

            for col in columns:
                self.data.generalize(self.tableWidget.horizontalHeaderItem(col).text(), self.hierarchy_data.df)
            self.update_table(self.data.df)
        except Exception as ex:
            self.open_except_window(str(ex))

    def delete_columns(self):
        try:
            columns = self.get_selected_columns()

            for col in sorted(columns, reverse=True):
                self.data.delete_column(self.tableWidget.horizontalHeaderItem(col).text())
                self.tableWidget.removeColumn(col)
        except Exception as ex:
            self.open_except_window(str(ex))

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

    def open_hierarchy_file(self):
        path = self.FileDialog()
        if path != "":
            try:
                df = pd.read_csv(path)
                if df.size == 0:
                    return

                self.hierarchy_data = TabularData(df)
            except Exception as ex:
                print(ex)
                self.open_except_window("Неверный формат файла")
        else:
            self.open_except_window("Неверный формат файла")

    def openFile(self):
        path = self.FileDialog()
        if path != "":
            try:
                self.load_excel_data(path)
            except Exception as ex:
                print(ex)
                self.open_except_window("Неверный формат файла")

    def saveFile(self):
        path = self.FileDialog(forOpen=False, fmt='csv')
        if path != "":
            try:
                self.data.save(path)
            except Exception as ex:
                print(ex)
                self.open_except_window("Ошибка сохранения файла")

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
