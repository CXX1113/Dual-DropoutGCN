import numpy as np
from sklearn.model_selection import KFold
import scipy.sparse as sp
import torch


def data_loader(dataset, not_computational):
    if dataset == 'SL':
        with open("../data/sl/List_Proteins_in_SL.txt", "r") as inf:
            gene_name = [line.rstrip() for line in inf]
            id_mapping = dict(zip(gene_name, range(len(set(gene_name)))))

        inter_pairs = []
        if not_computational:
            n_filter = 0
            computational_pairs = []
            with open("../data/sl/computational_pairs.txt", "r") as inf:
                for line in inf:
                    name1, name2 = line.rstrip().split()
                    computational_pairs.append({name1, name2})
            with open("../data/sl/SL_Human_Approved.txt", "r") as inf:
                for line in inf:
                    name1, name2, _ = line.rstrip().split()
                    if {name1, name2} not in computational_pairs:
                        inter_pairs.append((id_mapping[name1], id_mapping[name2]))
                    else:
                        n_filter += 1
            print("There are a total of {} SL pairs, and {} computational pairs are filtered.".format(len(inter_pairs) + n_filter, n_filter))

        else:
            with open("../data/sl/SL_Human_Approved.txt", "r") as inf:
                for line in inf:
                    name1, name2, _ = line.rstrip().split()
                    inter_pairs.append((id_mapping[name1], id_mapping[name2]))

        inter_pairs = np.array(inter_pairs, dtype=np.int32)

        return inter_pairs, len(id_mapping)


def feature_loader(num_node):
    identity_matrix = torch.eye(num_node)
    is_sparse_feat = True
    return identity_matrix, is_sparse_feat


def split_graph(kfold, pairs, num_node, seed):
    if kfold is not None:
        prng = np.random.RandomState(seed)
        kf = KFold(n_splits=kfold, random_state=prng, shuffle=True)

        graph_train_kfold = []
        graph_test_kfold = []
        for train_indices, test_indices in kf.split(pairs):
            graph_train = np.zeros((num_node, num_node))
            graph_test = np.zeros((num_node, num_node))

            pair_x_train, pair_y_train = pairs[train_indices, 0], pairs[train_indices, 1]
            graph_train[pair_x_train, pair_y_train] = 1
            graph_train[pair_y_train, pair_x_train] = 1

            pair_x_test, pair_y_test = pairs[test_indices, 0], pairs[test_indices, 1]
            graph_test[pair_x_test, pair_y_test] = 1
            graph_test[pair_y_test, pair_x_test] = 1

            graph_train_kfold.append(graph_train)
            graph_test_kfold.append(graph_test)

        return graph_train_kfold, graph_test_kfold

    else:
        graph_train = np.zeros((num_node, num_node))
        pair_x_train, pair_y_train = pairs[:, 0], pairs[:, 1]
        graph_train[pair_x_train, pair_y_train] = 1
        graph_train[pair_y_train, pair_x_train] = 1

        return graph_train


def normalize_mat(mat, normal_dim):
    # adj = sp.coo_matrix(adj)
    if normal_dim == 'Row&Column':
        # adj_ = mat + sp.eye(mat.shape[0])
        rowsum = np.array(mat.sum(1))
        inv = np.power(rowsum, -0.5).flatten()
        inv[np.isinf(inv)] = 0.
        degree_mat_inv_sqrt = sp.diags(inv)
        # D^{-0.5}AD^{-0.5}
        mat_normalized = mat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return mat_normalized

    elif normal_dim == 'Row':
        rowsum = np.array(mat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mat_normalized = r_mat_inv.dot(mat)
        return mat_normalized


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
