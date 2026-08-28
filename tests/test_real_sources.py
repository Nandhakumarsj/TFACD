import json
import socket
import threading
import time

from tfacd.streaming.sources import SocketStreamSource


def test_socket_stream_source(tmp_path):
    port = 19999
    source = SocketStreamSource(host="127.0.0.1", port=port, timeout_seconds=1.0, max_records=2)

    def client_thread():
        time.sleep(0.2)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            s.sendall(json.dumps({"ip.src_host": "192.168.1.10", "frame.len": 100}).encode() + b"\n")
            s.sendall(json.dumps({"ip.src_host": "192.168.1.11", "frame.len": 200}).encode() + b"\n")
            s.close()
        except Exception:
            pass

    t = threading.Thread(target=client_thread)
    t.start()

    records = list(source.records())
    t.join()

    assert len(records) == 2
    assert records[0]["ip.src_host"] == "192.168.1.10"
    assert records[1]["ip.src_host"] == "192.168.1.11"
