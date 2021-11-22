import socket
import os
import time

if __name__ == '__main__':
    host = socket.gethostname()
    port = 24002  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))

    # receive data stream. it won't accept data packet greater than 1024 bytes
    data = conn.recv(10240).decode()
    print("from connected user: " + str(data))
    print(len(data))
    if (len(data) == 28):
        print("Receiv Depth Images")
        p = os.system("python test_simpleclient.py")
        print("Depth record finished")
        print("Start Reconstruction")
        q = os.system("python test_tsdf.py")
        print("Reconstruction finished")
    conn.close()




