import time

def print_message_for_15_seconds():
    for i in range(15):
        print(f"Message {i+1}", flush=True)
        time.sleep(1)

if __name__=="__main__":
    print_message_for_15_seconds()
