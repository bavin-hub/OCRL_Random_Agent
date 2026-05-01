import time

num = 10
for i in range(1, num + 1):
    print(f"\rctr: {i}/{num}", end="", flush=True)
    time.sleep(0.5)
