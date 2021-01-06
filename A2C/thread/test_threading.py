# %%
import threading
# %%
import time
# %%
def thread_delay(thread_name, delay):
    count = 0
    print(thread_name, 'start:', time.time())
    while count < 3:
        time.sleep(delay)
        count += 1
        print(thread_name, 'doning',count, time.time())
    print(thread_name, 'done:', time.time())
# %%
t1 = threading.Thread(target=thread_delay, args=('t1', 1))
t2 = threading.Thread(target=thread_delay, args=('t2', 3))
t1.start()
t2.start()
# %%
t1.join()
t2.join()
# %%
