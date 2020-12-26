from tqdm import tqdm
from time import sleep

EPOCHS = 4

    
bar = tqdm(range(0,100,10),desc='here')

for i in bar:
    sleep(.1)

    postfix = f"Loss:{23},Accuracy:{1}"
    bar.set_postfix_str(postfix)