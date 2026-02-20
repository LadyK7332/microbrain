# core_sub.py
import zmq
import json

ADDR = "tcp://127.0.0.1:5555"

def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(ADDR)
    sock.setsockopt(zmq.SUBSCRIBE, b"percept/")  # subscribe to percept/* only

    print(f"[core] listening on {ADDR}")
    while True:
        topic, raw = sock.recv_multipart()
        event = json.loads(raw.decode("utf-8"))
        print(f"\n[core] topic={topic.decode()}")
        print(json.dumps(event, indent=2))

if __name__ == "__main__":
    main()