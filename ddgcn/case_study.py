from preprocess import *
from model import *
from objective import *
from evaluate import *
import pandas as pd

# Hyperparameter
EPOCH = 2000
SEED = 123

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
TOP_K = 10000  # Case Study

torch.manual_seed(SEED)
np.random.seed(SEED)


def case_study(dataset, unknown_pairs_id, unknown_pairs_scores, top_k, genes_degree):
    genes_degree = genes_degree.tolist()
    if dataset == 'SL':
        with open("../data/sl/List_Proteins_in_SL.txt", "r") as inf:
            gene_name = [line.rstrip() for line in inf]
            gene_mapping = dict(zip(range(len(set(gene_name))), gene_name))

        if top_k == 'all':
            sl_probs, pairs_id = unknown_pairs_scores.topk(len(unknown_pairs_scores))
        else:
            sl_probs, pairs_id = unknown_pairs_scores.topk(top_k)
        study_result = [[i+1, gene_mapping[unknown_pairs_id[pairs_id[i]][0]], genes_degree[unknown_pairs_id[pairs_id[i]][0]], gene_mapping[unknown_pairs_id[pairs_id[i]][1]], genes_degree[unknown_pairs_id[pairs_id[i]][1]], prob] for i, prob in enumerate(sl_probs.tolist())]
        frame = pd.DataFrame(data=study_result, columns=['Rank', 'Gene 1', 'Gene 1 Degree', 'Gene 2', 'Gene 2 Degree', 'Predicted likelihood'])
        frame.to_csv('./case_study_k={}.csv'.format(top_k), index=False)
        print("Case Study Finished!")


def main():
    inter_pairs, num_node = data_loader(DATASET, not_computational=NOT_COMPUTATIONAL)
    graph_train = split_graph(None, inter_pairs, num_node, None)

    genes_degree = torch.FloatTensor(graph_train).sum(dim=1)

    feature1, is_sparse_feat1 = feature_loader(num_node)
    nfeat = feature1.shape[1]

    adj_norm = normalize_mat(sp.coo_matrix(graph_train) + sp.eye(num_node), NORMAL_DIM)
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(adj_norm)

    adj_traget = torch.FloatTensor(graph_train + np.eye(num_node))

    if LOOP_FEAT2:
        feature2 = torch.FloatTensor(graph_train + np.eye(num_node))
        is_sparse_feat2 = True
    else:
        feature2 = torch.FloatTensor(graph_train)
        is_sparse_feat2 = True

    evaluator = Evaluator(graph_train)
    model = GraphAutoEncoder(nfeat, KERNAL_SIZE1, KERNAL_SIZE2, DROPOUT, INIT_TYPE, USE_BIAS, is_sparse_feat1,
                             is_sparse_feat2)
    obj = ObjectiveFunction(adj_traget)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

    print("Optimization Begin.")
    for j in range(EPOCH):
        model.train()
        optimizer.zero_grad()

        reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
        loss = obj.cal_loss(reconstruct_adj_logit1, reconstruct_adj_logit2, RHO)
        loss.backward()

        optimizer.step()
        need_early_stop_check = j > TOLERANCE_EPOCH and abs((loss.item() - last_loss) / last_loss) < STOP_THRESHOLD
        if need_early_stop_check or j + 1 >= EPOCH:
            if need_early_stop_check:
                print("Early stopping in {}th epoch...".format(j))
            else:
                print("Arrived at the last Epoch...")
            model.eval()
            with torch.no_grad():
                reconstruct_adj_logit1, reconstruct_adj_logit2 = model(feature1, feature2, adj_norm)
                reconstruct_adj1 = torch.sigmoid(reconstruct_adj_logit1)
                reconstruct_adj2 = torch.sigmoid(reconstruct_adj_logit2)
                unknown_pairs_id, unknown_pairs_scores = evaluator.unknown_pairs_scores(reconstruct_adj1,
                                                                                        reconstruct_adj2, RHO)
                case_study('SL', unknown_pairs_id, unknown_pairs_scores, TOP_K, genes_degree)
            break

        last_loss = loss.item()

    print("\nOptimization Finished!")


if __name__ == "__main__":
    main()

