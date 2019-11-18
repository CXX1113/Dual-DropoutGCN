from preprocess import *
from model import *
from objective import *
from evaluate import *
import time

# Hyperparameter
EPOCH = 2000
SEED = 123
KFold = 5
EVAL_INTER = 50
LR = 0.01
CONFIDENCE = 0.95

DATASET = 'SL'
DROPOUT = 0.5
INIT_TYPE = 'Kaiming'
USE_BIAS = False
LOOP_FEAT2 = False
NORMAL_DIM = 'Row&Column'
KERNAL_SIZE1 = 512
KERNAL_SIZE2 = 256
RHO = 1.
NOT_COMPUTATIONAL = False  # use SynLethDB-NonPred or not.
TOLERANCE_EPOCH = 1000
STOP_THRESHOLD = 1e-5
POS_THRESHOLD = 0.987

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    inter_pairs, num_node = data_loader(DATASET, not_computational=NOT_COMPUTATIONAL)
    graph_train_kfold, graph_test_kfold = split_graph(KFold, inter_pairs, num_node, SEED)

    feature1, is_sparse_feat1 = feature_loader(num_node)
    nfeat = feature1.shape[1]

    auc_kfold = []
    aupr_kfold = []
    f1_kfold = []

    for i in range(KFold):
        print("Using {} th fold dataset.".format(i+1))
        graph_train = graph_train_kfold[i]
        graph_test = graph_test_kfold[i]

        adj_norm = normalize_mat(sp.coo_matrix(graph_train) + sp.eye(num_node), NORMAL_DIM)
        adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(adj_norm)

        adj_traget = torch.FloatTensor(graph_train + np.eye(num_node))
        if LOOP_FEAT2:
            feature2 = torch.FloatTensor(graph_train + np.eye(num_node))
            is_sparse_feat2 = True
        else:
            feature2 = torch.FloatTensor(graph_train)
            is_sparse_feat2 = True

        model = GraphAutoEncoder(nfeat, KERNAL_SIZE1, KERNAL_SIZE2, DROPOUT, INIT_TYPE, USE_BIAS, is_sparse_feat1, is_sparse_feat2)
        obj = ObjectiveFunction(adj_traget)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
        evaluator = Evaluator(graph_train, graph_test, POS_THRESHOLD)
        obj_test = ObjectiveFunction(torch.FloatTensor(graph_test))

        for j in range(EPOCH):
            model.train()
            optimizer.zero_grad()

            reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
            loss = obj.cal_loss(reconstruct_adj_logit1, reconstruct_adj_logit2, RHO)
            loss.backward()

            optimizer.step()

            need_early_stop_check = j > TOLERANCE_EPOCH and abs((loss.item() - last_loss) / last_loss) < STOP_THRESHOLD
            if (j % EVAL_INTER == 0) or need_early_stop_check or j+1 >= EPOCH:
                t = time.time()
                model.eval()
                with torch.no_grad():
                    reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
                    test_loss = obj_test.cal_loss(reconstruct_adj_logit1, reconstruct_adj_logit2, RHO)
                    reconstruct_adj1 = torch.sigmoid(reconstruct_adj_logit1)
                    reconstruct_adj2 = torch.sigmoid(reconstruct_adj_logit2)
                    auc_test, aupr_test, f1_test = evaluator.eval(reconstruct_adj1, reconstruct_adj2, RHO)

                    print(
                        "Epoch:", '%04d' % (j + 1),
                        "train_loss=", "{:0>9.5f}".format(loss.item()),
                        "test_loss=", "{:0>9.5f}".format(test_loss.item()),
                        "test_auc=", "{:.5f}".format(auc_test),
                        "test_aupr=", "{:.5f}".format(aupr_test),
                        "test_f1=", "{:.5f}".format(f1_test),
                        "time=", "{:.2f}".format(time.time() - t))
                if need_early_stop_check or j+1 >= EPOCH:
                    auc_kfold.append(auc_test)
                    aupr_kfold.append(aupr_test)
                    f1_kfold.append(f1_test)
                    if need_early_stop_check:
                        print("Early stopping...")
                    else:
                        print("Arrived at the last Epoch...")
                    break

            last_loss = loss.item()

    print("\nOptimization Finished!")
    mean_auc, bound_auc = cal_confidence_interval(auc_kfold, confidence=CONFIDENCE)
    mean_aupr, bound_aupr = cal_confidence_interval(aupr_kfold, confidence=CONFIDENCE)
    mean_f1, bound_f1 = cal_confidence_interval(f1_kfold, confidence=CONFIDENCE)
    print("### Confidence Interval over pairs(Confidence={}):\n"
          "auc_mean ={:0>7.5f}, auc_bound ={:0>7.5f}\n"
          "aupr_mean={:0>7.5f}, aupr_bound={:0>7.5f}\n"
          "f1_mean={:0>7.5f}, f1_bound={:0>7.5f}\n".format(CONFIDENCE, mean_auc, bound_auc, mean_aupr, bound_aupr, mean_f1, bound_f1))


if __name__ == "__main__":
    main()
