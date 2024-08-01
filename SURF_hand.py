#! /usr/bin/env python3
import rospy

import socket

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8080))  # Use appropriate IP and port
    server_socket.listen(1)
    print('Server is listening...')

    while True:
        conn, addr = server_socket.accept()
        print(f'Connected by {addr}')
        data = conn.recv(1024)
        if not data:
            break
        print(f'Received data: {data.decode()}')
        conn.sendall(b'Data received')

    conn.close()


if __name__ == "__main__":
    start_server()