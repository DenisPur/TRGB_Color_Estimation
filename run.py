import sys
from PyQt5.QtWidgets import QApplication
from src.main import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    # window.set_input_folder('/home/your_input_folder')
    # window.set_output_folder('/home/your_output_folder')
    window.show()
    sys.exit(app.exec())
