import asyncio
import json
import time

import websockets

SERVER = "ws://10.146.11.202:8000/ws"
GROUP_ID = "rishee"
JOIN_TOKEN = "e9199364c71c4997"

META = {
    "train_loss": 2.0516929234529018,
    "train_accuracy": 0.29690346083788705,
    "local_steps": 70,
    "epoch_metrics": [
        {"epoch": 1, "loss": 2.226641477565731, "accuracy": 0.2887067395264117, "samples": 1098},
        {"epoch": 2, "loss": 1.8767443693400732, "accuracy": 0.30510018214936246, "samples": 1098},
    ],
}


async def main() -> None:
    client_id = f"manual_{int(time.time())}"

    async with websockets.connect(SERVER, ping_interval=10, ping_timeout=5) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "register",
                    "client_id": client_id,
                    "group_id": GROUP_ID,
                    "join_token": JOIN_TOKEN,
                    "data_metadata": {"modality": "vision", "samples": None},
                    "capabilities": {"has_gpu": False, "device": "cpu"},
                }
            )
        )
        reg = json.loads(await ws.recv())
        print("register:", reg)

        await ws.send(
            json.dumps(
                {
                    "type": "metrics",
                    "client_id": client_id,
                    "group_id": GROUP_ID,
                    "meta": META,
                }
            )
        )
        ack = json.loads(await ws.recv())
        print("metrics_ack:", ack)

        await ws.send(
            json.dumps(
                {
                    "type": "update",
                    "update": {
                        "client_id": client_id,
                        "client_version": 0,
                        "local_updates": "",
                        "update_type": "delta",
                        "local_dataset_size": 1,
                        "meta": META,
                    },
                }
            )
        )
        upd_ack = json.loads(await ws.recv())
        print("update_ack:", upd_ack)


if __name__ == "__main__":
    asyncio.run(main())
