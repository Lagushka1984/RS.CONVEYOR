import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MotorNode(Node):
    direction: str = 'F'
    speed: str = '0'

    def __init__(self) -> None:
        super().__init__('motor')
        self.pub = self.create_publisher(String, 'udp', 10)
        self.timer = self.create_timer(1, self.timer_callback)
        self.sub = self.create_subscription(String, 'motor_line', self.listener_callback, 10)
        self.sub
        self.last_msg = String()

    def timer_callback(self) -> None:
        while len(self.speed) != 3:
            self.speed = '0' + self.speed
        msg = String()
        msg.data = f'M {self.direction} {self.speed}'
        if self.last_msg.data != msg.data:
            self.pub.publish(msg)
            self.last_msg.data = msg.data
            self.get_logger().info(f'Sent: {msg.data}')

    def listener_callback(self, msg) -> None:
        self.direction, self.speed = msg.data.split()


def main(args=None):
    rclpy.init(args=args)
    motorNode = MotorNode()
    rclpy.spin(motorNode)
    motorNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
