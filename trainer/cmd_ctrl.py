"""
Control network parameter from command line!
"""
import threading
from trainer.base_gantrainer import BaseGANTrainer
import pprint

try: input = raw_input
except NameError: pass

class CMDControl(object):
    """
    Play with networks using dynamic parameters~
    """
    def __init__(self, Trainer):
        self.Trainer = Trainer
        self.thr = threading.Thread(target=CMDControl.worker, args=(self.Trainer,))
    
    def start_thread(self):
        self.thr.setDaemon(True)
        self.thr.start()
    
    @staticmethod
    def modify_tflags(Trainer):
        try:
            print("=> Enter TFLAGS key:")
            key = input().strip()
            print("=> Entered : %s" % key)
            print("=> Enter new value (int or float):")
            val = input()
            if key.find("iter") > -1:
                val = int(val)
            elif key.find("weight") > -1 or key.find("rate") > -1:
                val = float(val)

            Trainer.TFLAGS[key] = val
        except ValueError:
            print("=> Key not found or datatype misunderstood")
        except:
            print("=> Fatal error")
    
    @staticmethod
    def sendcmd_train(Trainer):
        print("=> Enter command for controlling trainer")
        cmd = input()
        
        if cmd.find("print") > -1:
            # print command
            print("=> Enter attributes that want to print")
            arg = input()
            if arg.find("global_iter"):
                print(Trainer.global_iter)
            else:
                print("=> Command not recognized")
        elif cmd.find("req") > -1:
            # request command
            # print("=> Enter request")
            # arg = input()
            #if arg.find("int_sum") > -1:
            print("=> Request interval summary")
            Trainer.req_inter_sum = True
            #else:
            #    print("=> Command not recognized")
        else:
            print("=> Command not recognized")

    @staticmethod
    def worker(Trainer):
        """
        Modify TFLAGS
        """
        print("=> Command line control is online.")

        while True:
            if Trainer.finished:
                print("=> Training finished, exit cmd tool")
                return 0
        
            try:
                print("=> Enter command")
                cmd = input()
                if cmd.find("print") == 0:
                    print("=> Print current TFLAGS")
                    pprint.pprint(Trainer.TFLAGS)
                elif cmd.find("exit") == 0:
                    print("=> Exit command line tool")
                    return 0
                elif cmd.find("tflags") == 0: #try to modify TFLAGS
                    CMDControl.modify_tflags(Trainer)
                elif cmd.find("train") == 0:
                    CMDControl.sendcmd_train(Trainer)
                else:
                    print("=> Command not recognized")
                    
            except SyntaxError:
                print("\n\n")
            except KeyboardInterrupt:
                return 0