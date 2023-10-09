import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
from utils import to_etype_name


class GCMCGraphConv(nn.Module):
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
            nn.init.xavier_uniform_(self.weight)

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
    def __init__(
            self,
            rating_vals,
            gene_in_units,  # 4762
            cell_in_units,  # 847
            msg_units,
            out_units,
            dropout_rate=0.0,

            device=None,
    ):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals

        self.ufc = nn.Linear(msg_units, out_units)
        self.ifc = nn.Linear(msg_units, out_units)

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

            self.W_r = None
            subConv[rating] = GCMCGraphConv(
                gene_in_units,
                msg_units,
                weight=True,
                device=device,
                dropout_rate=dropout_rate,
            )
            subConv[rev_rating] = GCMCGraphConv(
                cell_in_units,
                msg_units,
                weight=True,
                device=device,
                dropout_rate=dropout_rate,
            )

        self.conv = dglnn.HeteroGraphConv(subConv, aggregate='stack')
        self.agg_act = nn.ReLU()
        self.out_act = lambda x: x
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
            Gene-cell rating graph. It should contain two node types: "gene"
            and "cell" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            Gene features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Cell features. If None, using an identity matrix.

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
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)

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


class Decoder(nn.Module):

    def __init__(self, dropout_rate=0.0):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):

        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes["cell"].data["h"] = ifeat
            graph.nodes["gene"].data["h"] = ufeat
            graph.apply_edges(fn.u_dot_v("h", "h", "sr"))

            return graph.edata['sr']


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


class LinearNet(nn.Module):
    def __init__(self, emb_dim=256):
        super(LinearNet, self).__init__()
        self.ac = nn.ReLU()
        self.gene_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim)
        )
        self.cell_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
        )
        self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.gene1_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.gene2_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            self.ac,
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        gene = []
        for i in range(0, 2):
            gene_emb = x[:, i, :]
            gene_emb = self.flatten(gene_emb)
            gene_emb_1 = self.gene_model(gene_emb)
            gene.append(gene_emb_1)

        cell = []
        for i in range(0, 2):
            cell_emb = x[:, i + 2, :]
            cell_emb = self.flatten(cell_emb)
            cell_emb_1 = self.cell_model(cell_emb)
            cell.append(cell_emb_1)

        gene1_cell = th.cat([gene[0], cell[0]], dim=1)
        gene2_cell = th.cat([gene[1], cell[1]], dim=1)

        gene1_cell_1 = self.gene1_fc(gene1_cell)
        gene2_cell_1 = self.gene2_fc(gene2_cell)

        gene_cell = th.cat([gene1_cell_1, gene2_cell_1], dim=1)

        x = self.fc(gene_cell)
        return x