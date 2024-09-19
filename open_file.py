import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog


def open_file():
    app = QApplication(sys.argv)
    widget = QWidget()
    file_path, _ = QFileDialog.getOpenFileName(widget, 'Open File')
    if file_path:
        with open(file_path, 'r') as f:
            data = f.read()
            print(data)
    sys.exit(app.exec_())


if __name__ == '__main__':
    open_file()
