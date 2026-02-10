import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)

sock.connect("tcp://192.168.18.4:5555")

sock.send_string("Hola desde Python")
print(sock.recv_string())
