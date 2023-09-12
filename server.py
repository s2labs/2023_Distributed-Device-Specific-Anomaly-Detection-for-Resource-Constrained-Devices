import socket
import json

# Local Import
import anomaly_detector

HOST = ''  # Symbolic name meaning all available interfaces
PORT = 50007  # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024).decode()
                if not data: break

                # Read request and filter for parameters
                try:
                    parameters = json.loads(data)
                    print(f"Received request with parameters: {parameters}")
                    layer = parameters["layer"]
                    neurons = parameters["neurons"]
                    data = parameters["data"]
                    device = parameters["device"]

                except ValueError:
                    print(f'Invalid JSON request with parameters: {data}')
                    conn.sendall(str.encode(f'Invalid request with parameters: {data}'))
                except KeyError as e:
                    print(f'Invalid request with wrong parameter fields for: {e} in {data}')
                    conn.sendall(str.encode(f'Invalid request with wrong parameter fields: {data}'))

                # Process request
                result = anomaly_detector.process_dataset(layer, neurons, data,
                                                          device)  # , True) if it runs on Pi with local model

                # Reply to request
                conn.sendall(str.encode(''.join([str(x) for x in result])))
