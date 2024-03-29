
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 日志输出位置
LOG_PATH = 'output/log'


# 日志记录器
class Logger:
    def __init__(self, epochs: int, log_path: str = LOG_PATH):
        self.epochs = epochs
        self.loss_sum = []
        self.loss_dis = []
        self.loss_gen = []
        self.accuracy = []
        self.path = f'{log_path}/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(f'{self.path}/train_log.txt', 'w+') as file:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"train at {cur_time}\n")
            file.close()

    # 添加一次训练时的迭代中数据,并输出到日志中
    def log_loss(self, epoch: int, gen_loss: float, dis_loss: float, used_time: float):
        self.loss_gen.append(gen_loss)
        self.loss_dis.append(dis_loss)
        self.loss_sum.append(gen_loss + dis_loss)

        log_str = f"Epoch: {epoch:6} - loss:{gen_loss + dis_loss:>12.8f} [gen:{gen_loss:>12.8f}, dis:{dis_loss:>12.8f}] - time:{int(used_time / 3600):4.0f}:{int(used_time) / 60 % 60:02.0f}:{used_time - int(used_time / 60) * 60:05.2f}"
        print(log_str)
        with open(f'{self.path}/train_log.txt', 'a') as file:
            file.writelines(log_str + "\n")

    # 添加一次训练时的迭代中准确度数据,并输出到日志中
    def log_accuracy(self, epoch:int, accuracy: float):
        self.accuracy.append(accuracy)

        log_str = f"Epoch: {epoch:6} - acc:{accuracy}"
        print(log_str)
        with open(f'{self.path}/train_log.txt', 'a') as file:
            file.writelines(log_str + "\n")

    # 输出将所有迭代中损失变化的折线图
    def output_loss_change_figure(self):
        x = list(range(0, self.epochs + 1))

        # plt.plot(x, self.loss_sum, 's-', color='r', label="sum loss")
        plt.plot(x, self.loss_dis, 'o-', color='g', linewidth=1, markersize='1', label="discriminator loss")
        plt.plot(x, self.loss_gen, 'o-', color='b', linewidth=1, markersize='1', label="generator loss")

        plt.xlabel("iter time")
        plt.ylabel("loss")
        plt.legend(loc="best")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        plt.savefig(f'{self.path}loss_change.png')
        plt.close()

    # 输出将所有迭代中准确率变化的散点图
    def output_acc_change_figure(self):
        x = list(range(0, self.epochs + 1))

        plt.plot(x, self.accuracy, 'o', color='g', linewidth=1, markersize='1')

        plt.xlabel("iter time")
        plt.ylabel("accuracy")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        plt.savefig(f'{self.path}acc_change.png')
        plt.close()
