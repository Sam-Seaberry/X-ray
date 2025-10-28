from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QScrollArea, QVBoxLayout, QMenu, QFileDialog,
    QAbstractItemView, QLineEdit
)
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtGui import QPixmap, QImage, QAction, QFileSystemModel, QDropEvent, QDragEnterEvent, QDrag
from PyQt6.QtCore import Qt, QObject, QEvent,QModelIndex, pyqtSlot, QTimer, QMimeData
import numpy as np
import libeff
import logging
import sys
import os


logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logGUI.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
logger.addHandler(ch)

# ======================================================== #
# Class used to allow drag and drop from treeview to paths #
# ======================================================== #       
class LineEditDropFilter(QObject):
    def __init__(self, lineedit):
        super().__init__(lineedit)
        self.lineedit = lineedit
        self.lineedit.setAcceptDrops(True)
        lineedit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.lineedit:
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasText():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.Drop:
                if event.mimeData().hasText():
                    self.lineedit.setText(event.mimeData().text())
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(obj, event)

class ClickableImage(QLabel):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.full_pixmap = pixmap  # Store the full-resolution image
        self.current_scale = 1.0   # Keep track of zoom level

        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # Enable mouse tracking for smoother interactions
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        # Check modifier if you only want zoom on Ctrl+Scroll (optional)
        # if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
        #     return super().wheelEvent(event)

        angle = event.angleDelta().y()
        if angle > 0:
            self.current_scale *= 1.1  # Zoom in
        else:
            self.current_scale /= 1.1  # Zoom out

        # Limit zoom range
        self.current_scale = max(0.1, min(5.0, self.current_scale))

        # Apply scaling
        new_width = int(self.full_pixmap.width() * self.current_scale)
        new_height = int(self.full_pixmap.height() * self.current_scale)
        scaled_pixmap = self.full_pixmap.scaled(
            new_width, new_height, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        save_action = QAction("Save Image As...", self)
        save_action.triggered.connect(self.save_image)
        menu.addAction(save_action)
        menu.exec(self.mapToGlobal(pos))

    def save_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if filename:
            self.full_pixmap.save(filename)

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


def numpy_to_qpixmap(array: np.ndarray) -> QPixmap:
    """Convert a 2D NumPy array to a QPixmap for display."""
    norm_array = (array - np.min(array)) / (np.ptp(array)) * 255
    norm_array = norm_array.astype(np.uint8)

    height, width = norm_array.shape
    image = QImage(norm_array.data, width, height, width, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        logger.info("GUI Started")

        self.projectFolder = None
        self.mouse = 0 

        self.darkList = []
        self.flatList = []
        self.imageList = []

        self.allResults = []
        self.allDualResults = []

        self.resultImages = []
        self.resultReg = []
        self.resultInstensity = []
        self.resultPhase = []
        self.resultDark = []
        self.resultDual = []

        self.resultDualImages = []
        self.resultDualReg = []
        self.resultDualInstensity = []
        self.resultDualPhase = []
        self.resultDualDark = []
        self.resultDual = []

        self.display = 0
        self.displayTwo = 0

        self.ui = uic.loadUi('./res/UI/MainGUI.ui', self)

        self.ui.runButton.clicked.connect(lambda: self.run())

        self.ui.regButton.clicked.connect(lambda: self.setDisplay(0))
        self.ui.intensityButton.clicked.connect(lambda: self.setDisplay(1))
        self.ui.phaseButton.clicked.connect(lambda: self.setDisplay(2))
        self.ui.darkButton.clicked.connect(lambda: self.setDisplay(3))

        self.ui.threeImageButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.threeImagePage))
        self.ui.dualImageButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.twoImagePage))

        self.ui.runButton2.clicked.connect(lambda: self.runDual())

        self.ui.regButton2.clicked.connect(lambda: self.setDisplayTwo(0))
        self.ui.phaseButton2.clicked.connect(lambda: self.setDisplayTwo(2))
        self.ui.darkButton2.clicked.connect(lambda: self.setDisplayTwo(1))
        self.ui.thicknessButton.clicked.connect(lambda: self.setDisplayTwo(3))

        self.ui.setFolderDirButton.clicked.connect(lambda: self.openFolder())

        self.lineEditDropFilter = LineEditDropFilter(self.ui.threeImagePathOne)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.darkCorrectionOne)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.flatCorrectionOne)

        self.lineEditDropFilter = LineEditDropFilter(self.ui.threeImagePathTwo)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.darkCorrectionTwo)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.flatCorrectionTwo)

        self.lineEditDropFilter = LineEditDropFilter(self.ui.threeImagePathThree)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.darkCorrectionThree)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.flatCorrectionThree)

        self.lineEditDropFilter = LineEditDropFilter(self.ui.imageOnePath)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.darkCorrectionOneDual)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.flatCorrectionOneDual)

        self.lineEditDropFilter = LineEditDropFilter(self.ui.imageTwoPath)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.darkCorrectionTwoDual)
        self.lineEditDropFilter = LineEditDropFilter(self.ui.flatCorrectionTwoDual)



        self.show()

    # ================================================= #
    # Function used to load a project fdlder into GUI   #
    # ================================================= #
    def openFolder(self):
        self.projectFolder = (QFileDialog.getExistingDirectory(self, 'Open File', os.path.expanduser("~")))
        logger.info(f"Opened Folder: {self.projectFolder}")
        self.ui.projectDirDisplay.setText(str(self.projectFolder))
        self.addproject()

    def addproject(self, folder = None):
        if folder is not None:
            path = folder
        else:
            path = self.projectFolder
        self.ui.projectView.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        
        
        self.filelist = QFileSystemModel(self)
        self.filelist.setReadOnly(False)
        root_pth = self.filelist.setRootPath(path)
        self.filter = QtCore.QSortFilterProxyModel(self.ui.projectView)
        
        self.filter.setSourceModel(self.filelist)
        
        self.filter.setFilterRegularExpression(QtCore.QRegularExpression())
        self.ui.projectView.setModel(self.filter)
        self.ui.projectView.setColumnWidth(0, 300)
        self.ui.projectView.setRootIndex(self.filter.mapFromSource(root_pth))

        self.ui.projectView.setDragEnabled(True)
        self.ui.projectView.setAcceptDrops(False)
        self.ui.projectView.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)

        self.ui.projectView.clicked.connect(self.on_treeView_clicked)

    # ================================================================ #
    # Function used to allow users to drag from tree view into paths   #
    # ================================================================ #
    @pyqtSlot(QModelIndex)
    def on_treeView_clicked(self, index):
        index = self.filter.mapToSource(index)
        filePath = self.filelist.index(index.row(), 0, index.parent())

        drag = QDrag(self.ui.projectView)
        mimeData = QMimeData()
        if len(mimeData.objectName()) > 2:
            mimeData.setText(filePath)  # Send file path as text
            drag.setMimeData(mimeData)

            drag.exec(Qt.DropAction.CopyAction)
        
    def setDisplay(self, num):
        self.display = num
        self.showImages()

    def setDisplayTwo(self, num):
        self.displayTwo = num
        self.showDualImages()

    def runDual(self):
        self.ui.runtimeInfo.setText("Running Dual Image...")
        QApplication.processEvents() 
        logger.info("Running Dual Image")
        self.createImagePathLists()

        distances = [float(self.ui.imageOneDistance_2.text()),
                    float(self.ui.imageTwoDistance_2.text())]

        try:
            reg, dark, phase = libeff.dualPhaseDarkField(self.imageList, self.darkList, self.flatList, 
                                      distances, int(self.ui.stdStrideVal.text()), 
                                      int(self.ui.columbCutSize.text()),
                                      self.ui.highPass.isChecked(), 
                                      self.ui.edgeTaper.isChecked(),
                                      int(self.ui.highPassVal.text()))
            thickness = libeff.dualDistanceEffs(self.imageList, self.darkList, self.flatList,distances)
        except OSError as e:
            logger.error(f"Error encountered while running dual image phase and dark field retreval {e}")
            self.ui.runtimeInfo.setText(f"Error encountered while running dual image phase and dark field retreval {e}")
            return
        
        self.allDualResults = [reg, dark, phase, thickness]

        logger.info(f"Dual Distance exited sucessfully")
        self.ui.runtimeInfo.setText(f"Complete")

        self.showDualImages()
            


    def run(self):

        self.ui.runtimeInfo.setText("Running")
        QApplication.processEvents() 
        logger.info("Run Requested")

        self.createImagePathLists()

        self.resultImages = []
        self.resultReg = []
        self.resultInstensity = []
        self.resultPhase = []
        self.resultDark = []
        self.resultDual = []

        logger.info("Cleared Last Results")

        padValue = 0

        if self.ui.zeroPad.isChecked():
            padValue = 0
        elif self.ui.reflectPad.isChecked():
            padValue = 1
        elif self.ui.symetricPad.isChecked():
            padValue = 2
        elif self.ui.wrappedPad.isChecked():
            padValue = 3
        

        if self.ui.tikCheckOne.isChecked() and (self.ui.tikCheckTwo.isChecked() or self.ui.tikCheckThree.isChecked()):
            self.ui.runtimeInfo.setText("Please Only Select A Single Tikov Parameter To Sweep")
            return

        elif self.ui.tikCheckTwo.isChecked() and (self.ui.tikCheckOne.isChecked() or self.ui.tikCheckThree.isChecked()):
            self.ui.runtimeInfo.setText("Please Only Select A Single Tikov Parameter To Sweep")
            return

        elif self.ui.tikCheckThree.isChecked() and (self.ui.tikCheckOne.isChecked() or self.ui.tikCheckTwo.isChecked()):
            self.ui.runtimeInfo.setText("Please Only Select A Single Tikov Parameter To Sweep")
            return

        else:


            if len(self.ui.distanceOne.text()) == 0:
                self.ui.runtimeInfo.setText("Enter Distance 1")
                return

            elif len(self.ui.distanceTwo.text()) == 0:
                self.ui.runtimeInfo.setText("Enter Distance 2")
                return
            
            elif len(self.ui.distanceThree.text()) == 0:
                self.ui.runtimeInfo.setText("Enter Distance 3")
                return
            
            else:
                distances = [float(self.ui.distanceOne.text()),
                            float(self.ui.distanceTwo.text()),
                            float(self.ui.distanceThree.text())]
                
                tik_regs = [float(self.ui.tikParamOne.text()), 
                            float(self.ui.tikParamTwo.text()), 
                            float(self.ui.tikParamThree.text())]

                logger.info("Running X-Ray Effects")

                try:
                    
                    for i in range(0,int(self.ui.sweepReps.text())):
                        if self.ui.alignCheck.isChecked():
                            self.resultImages = libeff.xray_effects(self.imageList, self.darkList, self.flatList, distances, tik_regs, padValue, True)
                        else:
                            self.resultImages = libeff.xray_effects(self.imageList, self.darkList, self.flatList, distances, tik_regs, padValue)
                        
                        self.resultReg.append(self.resultImages[1])
                        self.resultInstensity.append(self.resultImages[2])
                        self.resultPhase.append(self.resultImages[3])
                        self.resultDark.append(self.resultImages[4])

                        if self.ui.tikCheckOne.isChecked():
                            tik_regs[0] = tik_regs[0] + (float(self.ui.sweepMaxVal.text()) - float(self.ui.tikParamOne.text())) / int(self.ui.sweepReps.text())

                        elif self.ui.tikCheckTwo.isChecked():
                            tik_regs[1] = tik_regs[1] + (float(self.ui.sweepMaxVal.text()) - float(self.ui.tikParamTwo.text())) / int(self.ui.sweepReps.text())

                        elif self.ui.tikCheckThree.isChecked():
                            tik_regs[2] = tik_regs[2] + (float(self.ui.sweepMaxVal.text()) - float(self.ui.tikParamThree.text())) / int(self.ui.sweepReps.text())


                    
                    self.allResults = [self.resultReg, self.resultInstensity, self.resultPhase, self.resultDark]
                    self.showImages
                    self.ui.runtimeInfo.setText("Complete")
                except OSError as e:
                    logger.error("Error {e} \n Running X-Ray Effects")
                    self.ui.runtimeInfo.setText("Error While Running - See Logging Info")
                    self.ui.runtimeInfo.setStyleSheet("color: red;") 
    
    def createImagePathLists(self):
        if self.ui.stackedWidget.currentWidget().objectName() == 'threeImagePage':
            self.darkList = (self.ui.darkCorrectionOne.text(), self.ui.darkCorrectionTwo.text(), self.ui.darkCorrectionThree.text())
            self.flatList = (self.ui.flatCorrectionOne.text(), self.ui.flatCorrectionTwo.text(), self.ui.flatCorrectionThree.text())
            self.imageList = (self.ui.threeImagePathOne.text(), self.ui.threeImagePathTwo.text(), self.ui.threeImagePathThree.text())
        else:
            self.darkList = (self.ui.darkCorrectionOneDual.text(), self.ui.darkCorrectionTwoDual.text())
            self.flatList = (self.ui.flatCorrectionOneDual.text(), self.ui.flatCorrectionTwoDual.text())
            self.imageList = (self.ui.imageOnePath.text(), self.ui.imageTwoPath.text())


    def showImages(self):
        layout = QHBoxLayout()
        for res in self.allResults[self.display]:
            pixmap = numpy_to_qpixmap(res)

            image_label = ClickableImage(pixmap)
            image_label.setPixmap(pixmap.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation))
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            vbox = QVBoxLayout()
            vbox.addWidget(image_label)

            layout.addLayout(vbox)

        container = QWidget()
        container.setLayout(layout)
        self.imageScroll.setWidget(container)

    def showDualImages(self):
        layout = QHBoxLayout()
        
        pixmap = numpy_to_qpixmap(self.allDualResults[self.displayTwo])

        image_label = ClickableImage(pixmap)
        image_label.setPixmap(pixmap.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(image_label)

        layout.addLayout(vbox)

        container = QWidget()
        container.setLayout(layout)
        self.imageScroll2.setWidget(container)


    def eventFilter(self, source, event):
        #if source is self.edit
        #print(self.dragdata)
        
        if event.type() == QEvent.Type.DragEnter:
            event.acceptProposedAction()
            source.setText(self.dragdata)
            print(self.dragdata)
            #source.dropEvent(self)
            self.dragdata = None
            
            #source.setText(self.dragdata)
            return QWidget.eventFilter(self, source, event)
        return False
    
    

            


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()

app.exec()
# Terminating tensorflow process when window closed
window.close()