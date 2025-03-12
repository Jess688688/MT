import pickle
import random

def perform_random_query(random_size):
    with open("cifar100_partition1.pkl", 'rb') as f:
        data = pickle.load(f)

    data_size = len(data['train_data'])

    random_indices = random.sample(range(data_size), random_size)

    raw_x_train = [data['train_data'][i] for i in random_indices]
    y_train = [data['train_labels'][i] for i in random_indices]

    raw_x_test, y_test = data['test_data'], data['test_labels']

    save_data = {
        "train_data": raw_x_train,
        "train_labels": y_train,
        "test_data": raw_x_test,
        "test_labels": y_test
    }

    with open("random_query.pkl", "wb") as f:
        pickle.dump(save_data, f)

    print("data is saved to random_query.pkl")

if __name__ == "__main__":
    perform_random_query(5000)
