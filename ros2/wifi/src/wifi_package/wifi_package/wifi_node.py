import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from socket import socket, AF_INET, SOCK_DGRAM


class UDP:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.server = socket(AF_INET, SOCK_DGRAM)

    def post(self, data: str) -> None:
        self.server.sendto(data.encode('utf-8'), self.addr)

    def get(self) -> str:
        data, _ = self.server.recvfrom(1024)
        return data.decode('utf-8')

    def close(self) -> None:
        self.server.close()


class WifiNode(Node):
    def __init__(self) -> None:
        super().__init__('server')
        self.sub = self.create_subscription(String, 'udp', self.listener_callback, 10)

        self.host = '192.168.0.200'
        self.port = 4444
        self.server = UDP(self.host, self.port)

    def listener_callback(self, msg) -> None:
        self.server.post(msg.data)
        self.get_logger().info(f'Sent: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    wifiNode = WifiNode()
    rclpy.spin(wifiNode)
    wifiNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
