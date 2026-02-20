# vision_pub.py
import time
import uuid
import zmq
import json

ADDR = "tcp://127.0.0.1:5555"
TOPIC = b"percept/vision/features"  # matches MicroBrain-style topics

def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(ADDR)

    print(f"[vision] publishing on {ADDR}")
    seq = 0
    time.sleep(0.2)  # let SUB connect (PUB/SUB needs a beat)

    while True:
        seq += 1

        event = {
            "topic": TOPIC.decode("utf-8"),
            "payload": {
                "seq": seq,
                "motion": 0.12,
                "objects": [{"label": "cat", "conf": 0.84}],
                "note": "fake features (no video yet)",
            },
            "timestamp": time.time(),
            "source": "vision-service-1",
            "correlation_id": uuid.uuid4().hex,
            "meta": {
                "ttl_ms": 750,
                "confidence": 0.84,
            },
        }

        sock.send_multipart([TOPIC, json.dumps(event).encode("utf-8")])
        time.sleep(0.5)

if __name__ == "__main__":
    main()