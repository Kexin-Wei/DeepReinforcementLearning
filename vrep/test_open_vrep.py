# open coppeliaSim from python
import glob
import time
import subprocess
import concurrent.futures

#subprocess.run("ls",shell=True)


def open_coppelia(port,hide=False):
    file = glob.glob("*.ttt")
    assert len(file) == 1, "There should be one and only one *.ttt file in the directory"    
    model_file = file[0]
    
    port_control = f" -gREMOTEAPISERVERSERVICE_{port}_FALSE_TRUE"#https://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
    command = f"coppeliaSim.sh {model_file}" + port_control
    if hide:
        command += " -h"
    
    #https://stackoverflow.com/questions/4996852/how-to-just-call-a-command-and-not-get-its-output
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p


def main():
    start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ports = [20000, 20001]
        ps = executor.map(open_coppelia,ports,[False]*2)     
                   
    end = time.perf_counter()
    print(f"Done with running time: {end-start:.2f}")
    
    time.sleep(20)
    
if __name__ == "__main__":
    #thread_open_coppelia(2000)
    main() 