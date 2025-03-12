import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("./output/pemsd7-m/train_loss.pickle", "rb") as f:
        train_loss_pemsd7 = pickle.load(f)

        train_loss = train_loss_pemsd7['train loss']
        copy_loss = train_loss_pemsd7['copy loss']

        plt.plot(train_loss, marker='o', linewidth=2, markersize=2, label='train loss')
        plt.plot(copy_loss, marker='o', linewidth=2, markersize=2, label='copy loss')
        plt.grid(True)
        plt.xlabel('# epoch')
        plt.ylabel('MAE')
        plt.title('PEMSD7M MAE Loss')
        plt.legend()
        plt.savefig('./output/pemsd7m-mae-loss.png')