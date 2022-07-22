import rclpy
import sys
from rclpy.node import Node
from std_msgs.msg import String
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt


class ConveyorNode(Node):
    def __init__(self) -> None:
        super().__init__('controller')
        self.pubMotor = self.create_publisher(String, 'motor_line', 10)

    def setMotorParameters(self, motor_param: str) -> None:
        msg = String()
        msg.data = motor_param
        self.pubMotor.publish(msg)


class GUI(QWidget):
    direction: str = 'F'
    speed: int = 0
    lastPacket: str = 'None'

    def __init__(self) -> None:
        super().__init__()
        self.params = ConveyorNode()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 640, 480)
        self.setWindowTitle('Conveyor')

        sendButton = QPushButton('Send', self)
        sendButton.clicked.connect(self.sendPacket)
        sendButton.resize(100, 50)
        sendButton.move(500, 400)

        self.packetLabel = QLabel(f'Last package sent: {self.lastPacket}             ', self)
        self.packetLabel.move(50, 400)

        self.speedBlock(180, 300)
        self.show()

    def speedBlock(self, x, y):
        slider = QSlider(Qt.Horizontal, self)
        slider.setGeometry(x + 20, y - 5, 300, 30)
        slider.setMinimum(0)
        slider.setMaximum(250)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(10)
        slider.setPageStep(10)
        slider.valueChanged.connect(self.setSpeed)

        minLabel = QLabel('0', self)
        minLabel.move(x, y)

        maxLabel = QLabel('250', self)
        maxLabel.move(x + 330, y)

        self.currentLabel = QLabel('000', self)
        self.currentLabel.move(x + 170, y + 30)

        self.directionLabel = QLabel('FORWARD    ', self)
        self.directionLabel.move(x + 145, y - 30)

        directionButton = QPushButton('Change \n direction', self)
        directionButton.clicked.connect(self.setDirection)
        directionButton.resize(80, 80)
        directionButton.move(x - 90, y - 30)

    def sendPacket(self):
        self.lastPacket = f'M {self.direction} {str(self.speed)}'
        self.params.setMotorParameters(f'{self.direction} {str(self.speed)}')
        self.packetLabel.setText(f'Last package sent: {self.lastPacket}')
        print(f'{self.direction} {str(self.speed)}')

    def setSpeed(self, value):
        self.speed = value
        self.currentLabel.setText(str(value))

    def setDirection(self):
        if self.direction == 'F':
            self.direction = 'B'
            self.directionLabel.setText('BACKWARD')
        elif self.direction == 'B':
            self.direction = 'F'
            self.directionLabel.setText('FORWARD')

def main():
    rclpy.init()
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
