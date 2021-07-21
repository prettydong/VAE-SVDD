from matplotlib import pyplot as plt
import time


def draw(t, name="mygraph.png"):
    t = t.cpu().detach().numpy()
    plt.imshow(t)
    plt.savefig(name)


class TimeRecorder:
    def __init__(self):
        self.init_time = time.time()
        self.start_time = None

    def start_record(self):
        if not self.start_time:
            self.start_time = time.time()
        else:
            raise Exception("bie XJB start")

    def end_record_return(self):

        last_time = time.time()-self.start_time
        self.start_time = None
        return last_time

    def get_time_from_init(self):
        return time.time() - self.init_time


def test():
    a = TimeRecorder()
    a.start_record()
    time.sleep(1)
    print(a.end_record_return())


if __name__ == '__main__':
    test()
