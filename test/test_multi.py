import threading

class A(object):
    def __init__(self):
        self.a = 1
        self.b = 2
    
    def p(self):
        print("A: %d %d" % (self.a, self.b))
    
def modA(obj):
    while True:
        try:
            cmd = raw_input()
            if cmd.find("p") == 0:
                obj.p()
            elif cmd.find("e") == 0:
                return 0
            else:
                print("Input a, b: ")
                a1 = int(input())
                b1 = int(input())
                obj.a = a1
                obj.b = b1
        except SyntaxError:
            print("\n\n")
        except KeyboardInterrupt:
            return 0

def printA(obj):
    while True:
        try:
            threading._sleep(1)
            obj.p()
        except KeyboardInterrupt:
            return 0

def main():
    o = A()
    threads = []
    t1 = threading.Thread(target=modA, args=(o,))
    #t2 = threading.Thread(target=printA, args=(o,))
    threads = [t1]

    for t in threads:
        t.setDaemon(True)
        t.start()
    try:
        t.join()
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    main()