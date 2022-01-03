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
        self.resize(1000, 800)
        self.show()

    def create_inputs_tab(self):
        tab_input = QWidget()

        self.button_file = QPushButton("Open File", self)
        self.button_file.clicked.connect(self.open_file)

        self.label_file1 = QLabel("file path : ")
        self.label_file2 = QLabel()

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(self.button_file)
        hbox.addWidget(self.label_file1)
        hbox.addWidget(self.label_file2)
        hbox.addStretch(2)
        vbox.addLayout(hbox)
        vbox.addWidget(self.table_data)

        tab_input.setLayout(vbox)

        return tab_input

    def open_file(self):
        file_name = QFileDialog.getOpenFileName(self)
        self.label_file2.setText(file_name[0])

        self.data = pd.read_excel(file_name[0], engine="openpyxl")

        self.table_data = QTableWidget()
        self.table_data.setColumnCount(len(self.data.columns))
        self.table_data.setRowCount(len(self.data.index))
        # self.table_data.setHorizontalHeaderLabels(
        #     ["Project No.", "Location Code", "Activity Code", "Start Date", "Finish Date", "Duration"])
        self.table_data.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        idx = 0
        for i, row in self.data.iterrows():
            self.table_data.insertRow(idx)
            for j, pt in row.iteritems():
                self.table_data.setItem(idx, j, QTableWidgetItem(str(pt)))
            idx += 1

    def create_run_tab(self):
        pass

    def create_results_tab(self):
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())