import socket
import json
import sys
import getopt


def main(argv):
    """
    Main function parsing the inputs and calling the send_request function
    """
    # Default parameters if not given
    layer = '3'
    neurons = '10'
    ratio = '1'
    device = 'default'
    host = '10.42.0.1'
    port = 50007

    try:
        opts, args = getopt.getopt(argv, "hl:n:r:d:s:p:",
                                   ["layer=", "neurons=", "ratio=", "device=", "server=", "port="])
    except getopt.GetoptError:
        print(
            'Error: client.py -l <num_layers> -n <amount_neurons> -r <ratio_training_data> -d <device> -s <server_ip> -p <server_port>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'client.py -l <num_layers> -n <amount_neurons> -r <ratio_training_data> -d <device> -s <server_ip> -p <server_port>')
            sys.exit()
        elif opt in ("-l", "--layer"):
            layer = arg
        elif opt in ("-n", "--neurons"):
            neurons = arg
        elif opt in ("-r", "--ratio"):
            ratio = arg
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-s", "--server"):
            host = arg
        elif opt in ("-p", "--port"):
            port = arg
        else:
            assert False, "unhandled option"
    send_request(layer, neurons, ratio, device, host, port)


def send_request(layer: str, neurons: str, ratio: str, device: str, host: str, port: str) -> float:
    """
    Send parameterized request to prediction server
    :param layer: Amount of layers
    :param neurons: Amount of neurons of first hidden layer
    :param ratio: Ratio of training data
    :param device: Device ID for device to be predicted
    :param host: IP of prediction server
    :param port: Port of prediction server
    :return: Duration of prediction on prediction server
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        msg = json.dumps({"layer": layer, "neurons": neurons, "data": ratio, "device": device}, sort_keys=True)
        print(f'Sent request: {msg}')
        s.sendall(str.encode(msg))
        data = s.recv(1024)
    print('Received', repr(data))

    returnvalues = repr(data)[2:-1]
    dur = float(returnvalues.split(';')[0])
    print(dur)

    return dur


if __name__ == "__main__":
    main(sys.argv[1:])
