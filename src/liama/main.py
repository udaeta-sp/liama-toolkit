"""Entry point for LIAMA Toolkit."""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from PyQt6.QtWidgets import QApplication

from .mainwindow import MainWindow
from .utils.theme import QSS


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LIAMA Toolkit")
    app.setStyleSheet(QSS)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
