import itertools

from down_task_model import *
from sklearn.metrics import roc_auc_score


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


def gnn_main(train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g):
    model = GraphSAGE(train_g.ndata["feat"].shape[1], 512)
    # You can replace DotPredictor with MLPPredictor.
    # pred = MLPPredictor(16)
    pred = MLPPredictor(512)
    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.01
    )

    # ----------- 4. training -------------------------------- #
    all_logits = []
    for e in range(400):
        # forward
        model.train()
        h = model(train_g, train_g.ndata["feat"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos_score = pred(train_pos_g, h)
            neg_score = pred(train_neg_g, h)
            # print("AUC", compute_auc(pos_score, neg_score))
        if e % 5 == 0:
            print("In epoch {}, loss: {}, train_auc: {}".format(e, loss, compute_auc(pos_score, neg_score)))

    # ----------- 5. check results ------------------------ #
    # from sklearn.metrics import roc_auc_score
    model.eval()
    with torch.no_grad():
        pos_score = pred(val_pos_g, h)
        neg_score = pred(val_neg_g, h)
        print("val AUC", compute_auc(pos_score, neg_score))
    model.eval()
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("Test AUC", compute_auc(pos_score, neg_score))

    # Thumbnail credits: Link Prediction with Neo4j, Mark Needham
    # sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
