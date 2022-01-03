import sys
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = None

    def initUI(self):
        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_inputs_tab(), "input")
        self.tabs.addTab(self.create_run_tab(), "optimize")
        self.tabs.addTab(self.create_results_tab(), "result")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.setWindowTitle("SAS")
        self.resize(1000, 500)
        self.show()

    def create_inputs_tab(self):
        tab_input = QWidget()

        self.button_file_i = QPushButton("Open File", self)
        self.button_file_i.clicked.connect(self.open_file)
        self.label_file_i1 = QLabel("file path : ")
        self.label_file_i2 = QLabel()

        self.table_data = QTableWidget()
        self.table_data.setColumnCount(6)
        self.table_data.setHorizontalHeaderLabels(
            ["Plate Welding", "Front-side SAW", "Turn-over", "Rear-side SAW", "Longitudina Attachment", "Longitudinal Welding"])
        self.table_data.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(self.button_file_i)
        hbox.addWidget(self.label_file_i1)
        hbox.addWidget(self.label_file_i2)
        hbox.addStretch(2)
        vbox.addLayout(hbox)
        vbox.addWidget(self.table_data)

        tab_input.setLayout(vbox)

        return tab_input

    def open_file(self):
        file_name = QFileDialog.getOpenFileName(self)
        self.label_file2.setText(file_name[0])

        self.data = pd.read_excel(file_name[0], engine="openpyxl")

        idx = 0
        for i, row in self.data.iterrows():
            self.table_data.insertRow(idx)
            for j, pt in row.iteritems():
                self.table_data.setItem(idx, j, QTableWidgetItem(str(pt)))
            idx += 1

    def create_run_tab(self):
        tab_run = QWidget()

        self.button_file_r = QPushButton("Open File", self)
        self.button_file_r.clicked.connect(self.open_file)
        self.label_file_r1 = QLabel("file path : ")
        self.label_file_r2 = QLabel()

        self.button_opt = QPushButton(" " * 50 + "Optimize" + " " * 50, self)
        self.button_opt.clicked.connect(self.optimize)

        self.label_opt1 = QLabel("number of samples : ")
        self.input_opt1 = QLineEdit(self)
        self.label_opt2 = QLabel("temperature : ")
        self.input_opt2 = QLineEdit(self)

        self.button_eval = QPushButton(" " * 50 + "Evaluation" + " " * 50, self)
        self.button_eval.clicked.connect(self.evaluate)

        self.label_eval1 = QLabel("error in processing time : ")
        self.input_eval1 = QLineEdit(self)
        self.label_eval2 = QLabel("number of iterations : ")
        self.input_eval2 = QLineEdit(self)
        self.label_eval3 = QLabel("random seed : ")
        self.input_eval3 = QLineEdit(self)

        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(False)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.button_file_r)
        hbox1.addWidget(self.label_file_r1)
        hbox1.addWidget(self.label_file_r2)

        grid = QGridLayout()
        grid.addWidget(self.button_opt, 0, 0, 1, 2)
        grid.addWidget(self.label_opt1, 1, 0)
        grid.addWidget(self.input_opt1, 1, 1)
        grid.addWidget(self.label_opt2, 2, 0)
        grid.addWidget(self.input_opt2, 2, 1)
        grid.addWidget(self.button_eval, 3, 0, 1, 2)
        grid.addWidget(self.label_eval1, 4, 0)
        grid.addWidget(self.input_eval1, 4, 1)
        grid.addWidget(self.label_eval2, 5, 0)
        grid.addWidget(self.input_eval2, 5, 1)
        grid.addWidget(self.label_eval3, 6, 0)
        grid.addWidget(self.input_eval3, 6, 1)
        grid.addWidget(self.log, 0, 2, 7, 1)

        vbox1 = QVBoxLayout()
        vbox1.addLayout(hbox1)
        vbox1.addLayout(grid)

        tab_run.setLayout(vbox1)

        return tab_run

    def optimize(self):
        pass

    def evaluate(self):
        pass

    def create_results_tab(self):
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())