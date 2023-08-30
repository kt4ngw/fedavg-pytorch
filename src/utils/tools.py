import numpy as np




def get_each_client_data_index(train_labels, client_num):
    np.random.seed(42)
    train_labels_num = len(train_labels)
    each_client_label_num = train_labels_num / client_num
    each_client_label_index = [[] for _ in range(client_num)]

    current_index = 0
    for i in range(client_num):
        end_index = int((i + 1) * each_client_label_num)
        client_indices = list(range(current_index, end_index))
        each_client_label_index[i] = client_indices
        current_index = end_index

    return each_client_label_index


