import os
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

try:
    import PyQt5
    pyqt_path = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins', 'platforms')
    if not os.path.exists(plugin_path):
        plugin_path = os.path.join(pyqt_path, 'plugins', 'platforms')
    
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
except Exception as e:
    print(f"Warning: Could not set Qt platform plugin path: {e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()