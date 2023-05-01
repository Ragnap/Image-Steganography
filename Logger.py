import os
import matplotlib.pyplot as plt
from datetime import datetime

# 日志输出位置
LOG_PATH = 'output/log'


# 日志记录器
class Logger:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.loss_sum = []
        self.loss_dis = []
        self.loss_gen = []

        path = f'{LOG_PATH}/'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{LOG_PATH}/train_log.txt', 'w+') as file:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"train at {cur_time}\n")
            file.close()

    # 添加一次训练时的迭代中数据,并输出到日志中
    def log_loss(self, epoch: int, gen_loss: float, dis_loss: float):
        self.loss_gen.append(gen_loss)
        self.loss_dis.append(dis_loss)
        self.loss_sum.append(gen_loss + dis_loss)
        log_str = f"Epoch: {epoch:6} - loss:{gen_loss + dis_loss:>12.3f} [gen:{gen_loss:>12.3f}, dis:{dis_loss:>12.3f}]"
        print(log_str)
        with open(f'{LOG_PATH}/train_log.txt', 'a') as file:

            file.writelines(log_str + "\n")

    # 输出将所有迭代中损失变化的折线图
    def output_loss_change_figure(self):
        x = list(range(0, self.epochs + 1))

        # plt.plot(x, self.loss_sum, 's-', color='r', label="sum loss")
        plt.plot(x, self.loss_dis, 'o-', color='g', label="discriminator loss")
        plt.plot(x, self.loss_gen, 'o-', color='b', label="generator loss")

        plt.xlabel("iter time")
        plt.ylabel("loss")
        plt.legend(loc="best")

        path = f'{LOG_PATH}/'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(f'{path}loss_change.png')
