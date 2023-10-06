
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from utils import to_etype_name
import os

# def set_seed(seed_num):
#     th.manual_seed(seed_num)
#     th.cuda.manual_seed(seed_num)
#     th.cuda.manual_seed_all(seed_num)
#     np.random.seed(seed_num)
#     # random.seed(seed_num)
#     th.backends.cudnn.benchmark = False
#     th.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed_num)
#
#
# # seed_num = 3407
# set_seed(123)  # 27 114514

class GCMCGraphConv(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(
            self, in_feats, out_feats, weight=True, device=None, dropout_rate=0.0
    ):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """

        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used

            cj = graph.srcdata["cj"]
            ci = graph.dstdata["ci"]
            # print(cj.shape)
            # print(ci.shape)
            # print(self.weight.shape)
            # exit()
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)
            # print(feat.shape)
            # exit()
            feat = feat * self.dropout(cj)
            graph.srcdata["h"] = feat
            graph.update_all(
                fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
            )
            rst = graph.dstdata["h"]
            rst = rst * ci

        return rst


class GCMCBlock(nn.Module):

    def __init__(self, in_feats, hidden_feats, out_feats, weight=True, device=None, dropout_rate=0.0):
        super(GCMCBlock, self).__init__()
        self.gcmc_conv1 = GCMCGraphConv(in_feats=in_feats, out_feats=out_feats, weight=weight, device=device,
                                        dropout_rate=dropout_rate)
        self.gcmc_conv2 = GCMCGraphConv(in_feats=hidden_feats, out_feats=out_feats, weight=True, device=device,
                                        dropout_rate=dropout_rate)

    def forward(self, graph, feat, weight=None):
        h = self.gcmc_conv1(graph, feat, weight)
        h = self.gcmc_conv2(graph, h, weight)
        return h


class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both gene nodes and cell nodes and the parameters
    are not shared unless ``share_gene_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    gene_in_units : int
        Size of gene input feature
    cell_in_units : int
        Size of cell input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output gene and cell features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_gene_item_param : bool, optional
        If true, gene node and cell node share the same set of parameters.
        Require ``gene_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(
            self,
            rating_vals,
            gene_in_units,  # 4762
            cell_in_units,  # 847
            msg_units,
            out_units,
            dropout_rate=0.0,
            agg="stack",  # or 'sum'
            agg_act=None,
            out_act=None,
            share_gene_item_param=False,
            device=None,
    ):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_gene_item_param = share_gene_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        # self.ufc = nn.Sequential(
        #     nn.Linear(msg_units, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, out_units)
        # )
        # print('msg', msg_units, out_units)
        if share_gene_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
            # self.ifc = nn.Sequential(
            #     nn.Linear(msg_units, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, out_units)
            # )
        if agg == "stack":
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        subConv = {}
        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            origin_rating = rating
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            if share_gene_item_param and gene_in_units == cell_in_units:
                self.W_r[rating] = nn.Parameter(
                    th.randn(gene_in_units, msg_units)
                )
                self.W_r["rev-%s" % rating] = self.W_r[rating]
                subConv[rating] = GCMCGraphConv(
                    gene_in_units,
                    msg_units,
                    weight=False,
                    device=device,
                    dropout_rate=dropout_rate,
                )
                subConv[rev_rating] = GCMCGraphConv(
                    gene_in_units,
                    msg_units,
                    weight=False,
                    device=device,
                    dropout_rate=dropout_rate,
                )
            else:
                self.W_r = None
                # if origin_rating < len(rating_vals) - 1:
                subConv[rating] = GCMCGraphConv(
                    gene_in_units,
                    # gene_in_units,
                    msg_units,
                    weight=True,
                    device=device,
                    dropout_rate=dropout_rate,
                )
                subConv[rev_rating] = GCMCGraphConv(
                    cell_in_units,
                    # cell_in_units,
                    msg_units,
                    weight=True,
                    device=device,
                    dropout_rate=dropout_rate,
                )
                # elif origin_rating == len(rating_vals) - 1:
                #     subConv[rating] = GCMCGraphConv(
                #         gene_in_units,
                #         # gene_in_units,
                #         msg_units,
                #         weight=True,
                #         device=device,
                #         dropout_rate=dropout_rate,
                #     )
                #     subConv[rev_rating] = GCMCGraphConv(
                #         gene_in_units,
                #         # cell_in_units,
                #         msg_units,
                #         weight=True,
                #         device=device,
                #         dropout_rate=dropout_rate,
                #     )
                # else:
                #     subConv[rating] = GCMCGraphConv(
                #         cell_in_units,
                #         # gene_in_units,
                #         msg_units,
                #         weight=True,
                #         device=device,
                #         dropout_rate=dropout_rate,
                #     )
                #     subConv[rev_rating] = GCMCGraphConv(
                #         cell_in_units,
                #         msg_units,
                #         weight=True,
                #         device=device,
                #         dropout_rate=dropout_rate,
                #     )
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        # self.conv1 = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_gene_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        """Forward function

        Parameters
        ----------
        graph : DGLGraph
            gene-cell rating graph. It should contain two node types: "gene"
            and "cell" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            gene features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            cell features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New gene features
        new_ifeat : torch.Tensor
            New cell features
        """
        in_feats = {"gene": ufeat, "cell": ifeat}

        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            mod_args[rating] = (
                self.W_r[rating] if self.W_r is not None else None,
            )
            mod_args[rev_rating] = (
                self.W_r[rev_rating] if self.W_r is not None else None,
            )
        # mod_args['uu'] = (
        #     self.W_r['uu'] if self.W_r is not None else None,
        # )
        # mod_args['uu1'] = (
        #     self.W_r['uu1'] if self.W_r is not None else None,
        # )
        # mod_args['mm'] = (
        #     self.W_r['mm'] if self.W_r is not None else None,
        # )
        # mod_args['mm1'] = (
        #     self.W_r['mm1'] if self.W_r is not None else None,
        # )
        # print(mod_args)
        # exit()
        # print(in_feats)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        # print(out_feats['gene'].shape, out_feats['cell'].shape)
        # exit()

        ufeat = out_feats["gene"]
        ifeat = out_feats["cell"]
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)


# Define a Heterograph Conv model


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, 1)

    def apply_edges(self, edges):
        x = th.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        # gene embedding
        self.gene_embedding = nn.Embedding(4762, 256)
        # cell embedding
        self.cell_embedding = nn.Embedding(847, 256)
        self.inputs = {'gene': self.gene_embedding.weight, 'cell': self.cell_embedding.weight}
        self.pred = HeteroDotProductPredictor()
        self.pred1 = HeteroMLPPredictor(256, len(rel_names))

    def forward(self, enc_graph, dec_graph):
        # 输入是节点的特征字典
        h = self.conv1(enc_graph, self.inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv2(enc_graph, h)
        # return self.pred(dec_graph,h, dec_graph.etypes[0]), h['gene'], h['cell']
        return self.pred1(dec_graph, h), h['gene'], h['cell']


class BiDecoder(nn.Module):
    r"""Bi-linear decoder.

    Given a bipartite graph G, for each edge (i, j) ~ G, compute the likelihood
    of it being class r by:

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    in_units : int
        Size of input gene and cell features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self, in_units=256, num_basis=1, dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ParameterList(
            nn.Parameter(th.randn(in_units, in_units)) for _ in range(num_basis)
        )
        self.W1 = nn.Linear(in_units * 2, in_units, bias=True)
        self.W2 = nn.Linear(in_units, 1, bias=True)
        # self.fc = nn.Linear(in_units, num_classes, bias=False)
        # self.ac = nn.Softmax(dim=1)
        self.combine_basis = nn.Linear(self._num_basis, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = th.cat([h_u], 1)
        # score = self.W2(F.relu(score))
        return {'cl_sr': score}

    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLGraph
            "Flattened" gene-cell graph with only one edge type.
        ufeat : th.Tensor
            gene embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            cell embeddings. Shape: (|V_m|, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each gene-cell edge.
        """
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes["cell"].data["h"] = ifeat
            basis_out = []
            class_out = []
            for i in range(self._num_basis):
                graph.nodes["gene"].data["h"] = ufeat  # @ self.Ps[i]
                graph.apply_edges(fn.u_dot_v("h", "h", "sr"))
                # graph.apply_edges(self.apply_edges)
                return graph.edata['sr']
                # basis_out.append(graph.edata["sr"])
                class_out.append(graph.edata["cl_sr"])
            # out = th.cat(basis_out, dim=1)
            out = th.cat(class_out, dim=1)
            out = self.W2(out)
            # out = self.combine_basis(out)
            # out2 = self.fc(th.cat(class_out, dim=1))
            # out2 = self.ac(out2)
        return out


class DenseBiDecoder(nn.Module):
    r"""Dense bi-linear decoder.

    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        Size of input gene and cell features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """

    def __init__(self, in_units, num_classes, num_basis=2, dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ufeat, ifeat):
        """Forward function.

        Compute logits for each pair ``(ufeat[i], ifeat[i])``.

        Parameters
        ----------
        ufeat : th.Tensor
            gene embeddings. Shape: (B, D)
        ifeat : th.Tensor
            cell embeddings. Shape: (B, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each gene-cell edge. Shape: (B, num_classes)
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum("ai,bij,aj->ab", ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.GraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2 = dglnn.GraphConv(
            in_feats=hid_feats, out_feats=out_feats, )
        self.conv3 = dglnn.GraphConv(
            in_feats=out_feats, out_feats=out_feats)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        return h


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class SAGEModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        # self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return h  # self.pred(g, h)

