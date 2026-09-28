# Send a batch of test images to /predict so the Grafana dashboard has data.
#   python -m scripts.load_test --n 50
# a few requests are deliberately bad (wrong type, corrupt file) to show up as 4xx
import argparse
import glob
import random
from concurrent.futures import ThreadPoolExecutor

import httpx

from src.config import TEST_IMGS


def send(client, url, path):
    if path == "bad-type":
        files = {"file": ("notes.txt", b"hello", "text/plain")}
    elif path == "corrupt":
        files = {"file": ("broken.png", b"not really a png", "image/png")}
    else:
        files = {"file": (path, open(path, "rb").read(), "image/png")}
    return client.post(f"{url}/predict", files=files).status_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    images = glob.glob(f"{TEST_IMGS}/*.png")
    jobs = random.sample(images, min(args.n, len(images)))
    jobs += ["bad-type", "corrupt"] * max(1, args.n // 20)
    random.shuffle(jobs)

    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(lambda p: send(client, args.url, p), jobs))

    for code in sorted(set(codes)):
        print(f"{code}: {codes.count(code)}")


if __name__ == "__main__":
    main()
