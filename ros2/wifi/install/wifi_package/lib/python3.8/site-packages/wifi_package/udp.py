from socket import socket, AF_INET, SOCK_DGRAM


class UDP:
    def __int__(self, host: str, port: str) -> None:
        self.addr = (host, port)
        self.server = socket(AF_INET, SOCK_DGRAM)
        self.server.bind(self.addr)

    def post(self, data: str) -> None:
        self.server.sendto(data.encode('utf-8'), self.addr)

    def get(self) -> str:
        data, _ = self.server.recvfrom(1024)
        return data.decode('utf-8')

    def close(self) -> None:
        self.server.close()
