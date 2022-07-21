import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from blessed import Terminal


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('controller')
        self.pubMotor = self.create_publisher(String, 'motor_line', 10)

    def setMotorParameters(self, motor_param: str) -> None:
        msg = String()
        msg.data = motor_param
        self.pubMotor.publish(msg)


class GUI:
    direction: str = 'F'
    speed: int = 100
    lastPacket: str = 'None'

    def __init__(self) -> None:
        self.term = Terminal()
        self.params = ControllerNode()
        print(self.term.home + self.term.clear)
        with self.term.location(x=self.term.width // 2 - 4, y=0):
            print(self.term.bold + self.term.green + 'CONVEYOR')
        with self.term.location(x=0, y=2):
            print(self.term.orange + self.term.bold + 'ENTER - send the packet')
        with self.term.location(x=0, y=3):
            print(self.term.orange + self.term.bold + 'L/R arrows - change speed')
        with self.term.location(x=0, y=4):
            print(self.term.orange + self.term.bold + 'U/D arrows - change direction')
        with self.term.location(x=0, y=5):
            print(self.term.orange + self.term.bold + 'ESCAPE - stop program')
        with self.term.location(x=0, y=20):
            print(self.term.green + self.term.bold + 'Last package sent: ')

    def run(self) -> None:
        with self.term.cbreak():
            val = self.term.inkey(timeout=3)
            while val.name != 'KEY_ESCAPE':
                val = self.term.inkey(timeout=3)
                if val.name == 'KEY_LEFT' and self.speed - 10 >= 0:
                    self.speed -= 10
                if val.name == 'KEY_RIGHT' and self.speed + 10 <= 250:
                    self.speed += 10
                if val.name == 'KEY_UP':
                    self.direction = 'F'
                if val.name == 'KEY_DOWN':
                    self.direction = 'B'
                if val.name == 'KEY_ENTER':
                    self.lastPacket = f'M {self.direction} {str(self.speed)}'
                    self.params.setMotorParameters(f'{self.direction} {str(self.speed)}')
                self.speedBlock(self.term.width // 2 - 17, 10)
                with self.term.location(x=20, y=20):
                    print(self.term.green + self.term.bold + self.term.clear_eol + self.lastPacket)
            print(self.term.clear)

    def speedBlock(self, x: int, y: int) -> None:
        with self.term.location(x=0, y=y + 1):
            print(self.term.clear_eol)
        with self.term.location(x=x, y=y + 1):
            print(self.term.bold + self.term.on_blue + '  0  ')
        with self.term.location(x=x + 30, y=y + 1):
            print(self.term.bold + self.term.on_blue + ' 250 ')
        with self.term.location(x=x + 16, y=y + 2):
            print(str(self.speed) + self.term.clear_eol)
        with self.term.location(x=x + 5, y=y + 1):
            print(self.term.on_yellow + ' ' * (self.speed // 10))
        with self.term.location(x=x + 13, y=y):
            if self.direction == 'F':
                print(self.term.clear_eol + 'FORWARD')
            if self.direction == 'B':
                print(self.term.clear_eol + 'BACKWARD')


def main():
    rclpy.init()
    gui = GUI()
    gui.run()


if __name__ == '__main__':
    main()
