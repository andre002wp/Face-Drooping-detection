from sys import executable, argv
from PyQt5.QtWidgets import QFileDialog, QApplication

def openFiles(directory='./'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename"""
    # run this exact file in a separate process, and grab the result
    app = QApplication([directory])
    if directory is None: directory ='./'
    fname = QFileDialog.getOpenFileName(None, "Select data file...", 
                directory, filter="All files (*)")
    return fname[0]

aa = openFiles()
print(aa)