import dgl
import math as m
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import torch as th
import collections

from utils import to_etype_name


class ScRNASeqData:
    def __init__(
            self,
            name,
            device,
            path='data_evaluation/bonemarrow/bone_marrow_cell.h5',
            istime=False,
            ish5=False,
            rank_all=0,
    ):
        self._name = name
        self._device = device
        self.rank_all = rank_all

        print("Starting processing {} ...".format(self._name))

        if not istime:
            if not ish5:
                data = pd.read_csv(path, header='infer', index_col=0)
            else:
                data = pd.read_hdf(path, key='/RPKMs').T

        else:
            time_h5 = []
            files = os.listdir(path)
            for i in range(len(files)):
                print('time_points', i)
                if name == "timeData/mesc1":
                    time_pd = pd.read_hdf(path + 'RPM_' + str(i) + '.h5', key='/RPKM')
                else:
                    time_pd = pd.read_hdf(path + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
                time_h5.append(time_pd)
            train_data = pd.concat(time_h5, axis=0, ignore_index=True)
            data = train_data.T

        assert np.unique(data.index).shape[0] == data.shape[0]
        assert np.unique(data.columns).shape[0] == data.shape[1]

        self.all_train_rating_info, self.rpkm_train_rating_info, num_gene, num_cell = self.build_bin_normalized_matrix(
            data)
        self.rpkm_train_rating_info, self.enc_gene_cell_g, self.dec_gene_cell_g = self.getScores(data)

        self.all_rating_info = self.all_train_rating_info

        print("......")

        self.train_rating_info = self.all_train_rating_info
        self.possible_rating_values = np.unique(
            self.train_rating_info["rating"].values
        )

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print(
            "\tAll train rating pairs : {}".format(
                self.all_train_rating_info.shape[0]
            )
        )
        print(
            "\t\tTrain rating pairs : {}".format(
                self.train_rating_info.shape[0]
            )
        )

        # Map gene/cell to the global id
        self.global_gene_id_map = {
            ele: i for i, ele in enumerate(range(num_gene))
        }
        self.global_cell_id_map = {
            ele: i for i, ele in enumerate(range(num_cell))
        }
        print(
            "Total gene number = {}, cell number = {}".format(
                len(self.global_gene_id_map), len(self.global_cell_id_map)
            )
        )

        self._num_gene = len(self.global_gene_id_map)
        self._num_cell = len(self.global_cell_id_map)

        ### Generate features
        self.gene_feature = None
        self.cell_feature = None

        self.gene_feature_shape = (self.num_gene, self.num_gene)
        self.cell_feature_shape = (self.num_cell, self.num_cell)
        info_line = "Feature dim: "
        info_line += "\ngene: {}".format(self.gene_feature_shape)
        info_line += "\ncell: {}".format(self.cell_feature_shape)
        print(info_line)

        train_rating_pairs, train_rating_values = self._generate_pair_value(
            self.train_rating_info
        )
        _, rpkm_train_rating_values = self._generate_pair_value(
            self.rpkm_train_rating_info
        )

        print(len(rpkm_train_rating_values))
        print(len(train_rating_values))

        def _make_labels(ratings):
            labels = th.LongTensor(
                np.searchsorted(self.possible_rating_values, ratings)
            ).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(
            train_rating_pairs, train_rating_values, add_support=True
        )
        print(
            self.train_enc_graph
        )

        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)

        self.train_labels = _make_labels(train_rating_values)

        self.train_truths = th.FloatTensor(rpkm_train_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.num_edges(str(r))
            return rst

        print(
            "Train enc graph: \t#gene:{}\t#cell:{}\t#pairs:{}".format(
                self.train_enc_graph.num_nodes("gene"),
                self.train_enc_graph.num_nodes("cell"),
                _npairs(self.train_enc_graph),
            )
        )
        print(
            "Train dec graph: \t#gene:{}\t#cell:{}\t#pairs:{}".format(
                self.train_dec_graph.num_nodes("gene"),
                self.train_dec_graph.num_nodes("cell"),
                self.train_dec_graph.num_edges(),
            )
        )

    def build_bin_normalized_matrix(self, data):

        print(data)
        class_num = self.rank_all

        num_gene = data.shape[0]
        num_cell = data.shape[1]
        data = data.T

        gene_name = list(data)
        new_data = pd.DataFrame()
        new_data.index = data.index
        new_data_rpkm = pd.DataFrame()
        new_data_rpkm.index = data.index
        new_data_dict = {}
        new_data_rpkm_dict = {}

        zero_index = np.where(data.values == 0)
        mask = np.ones_like(data.values)
        mask[zero_index] = 0
        for gene in gene_name:
            temp = data[gene]

            non_zero_element = np.log(temp[temp != 0.].values)

            if len(non_zero_element) == 0:
                new_data_dict[gene] = temp.apply(lambda x: 0)
                rpkm_temp = temp.apply(lambda x: 9999)
                new_data_rpkm_dict[gene] = rpkm_temp
                continue

            mean = np.mean(non_zero_element)
            tmin = np.min(non_zero_element)
            std = np.std(non_zero_element)
            tmax = np.max(non_zero_element)
            lower_bound = max(mean - 2 * std, tmin)
            upper_bound = min(mean + 2 * std, tmax)
            bucket_width = (upper_bound - lower_bound) / class_num
            mask_zero = np.ones_like(temp)
            mask_zero[temp == 0] = 0
            rpkm_temp = temp

            try:
                temp = temp.apply(lambda x: 0 if x == 0.0 else m.floor((m.log(x) - lower_bound) / bucket_width))
            except:
                temp = temp.apply(lambda x: 0 if x == 0.0 else 0)
            rpkm_temp = rpkm_temp.apply(lambda x: 9999 if x == 0 else (m.log(x) - lower_bound) / bucket_width)
            temp[temp >= class_num] = class_num - 1
            temp[(temp < 0)] = 0

            temp = temp + 1
            temp = temp * mask_zero
            new_data_dict[gene] = temp
            new_data_rpkm_dict[gene] = rpkm_temp

        new_data = pd.DataFrame(new_data_dict)
        print('gene expression level matrix:')
        print(new_data)
        new_data = th.tensor(new_data.T.values, dtype=th.float32)
        new_data_rpkm = pd.DataFrame(new_data_rpkm_dict)
        # print(new_data_rpkm)
        new_data_rpkm = th.tensor(new_data_rpkm.T.values, dtype=th.float32)

        from collections import Counter
        print(Counter(new_data.flatten().numpy()))

        # new_data to pairs (gene,cell,rating)
        gene, cell = np.where(new_data != 0)
        rating = new_data[gene, cell]
        pairs = np.stack([gene, cell, rating], axis=1)
        pd_pairs = pd.DataFrame(pairs, columns=['gene_id', 'cell_id', 'rating'])

        # new_rpkm_data to pairs
        gene, cell = np.where(new_data_rpkm != 9999)
        rating = new_data_rpkm[gene, cell]
        pairs = np.stack([gene, cell, rating], axis=1)
        pd_pairs_rpkm = pd.DataFrame(pairs, columns=['gene_id', 'cell_id', 'rating'])

        return pd_pairs, pd_pairs_rpkm, num_gene, num_cell

    def getScores(self, data):
        # rpkm = data = pkm = pd.read_csv(path, header='infer', index_col=0).T
        rpkm = data.T
        columns = rpkm.columns
        geneName_to_id = {}
        for i, geneName in enumerate(rpkm.columns):
            geneName_to_id[geneName] = i

        # 处理特征
        data_values = rpkm.values
        # # mask 0
        zero_index = np.where(data_values == 0)
        mask = np.ones_like(data_values)
        mask[zero_index] = 0
        mask = th.tensor(mask.T, dtype=th.float32)

        #
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            if sum(tmp != 0) == 0:
                means.append(0)
                stds.append(1)
            else:
                if sum(tmp != 0) == 1:
                    pass
                means.append(tmp[tmp != 0].mean())
                stds.append(tmp[tmp != 0].std())

        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 1
        stds[np.isinf(stds)] = 1
        means[np.isnan(stds)] = 0
        means[np.isinf(stds)] = 0
        stds[stds == 0] = 1
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -20)
        data_values = np.minimum(data_values, 20)
        input_feature = data_values.T

        rpkm_feature = th.tensor(input_feature, dtype=th.float32)
        rpkm_feature = rpkm_feature * mask
        print(rpkm_feature)
        # new_data to pairs (gene,cell,rating)
        gene, cell = np.where(mask != 0)  # np.where(rpkm_feature != 0)
        cell_idx = cell + rpkm_feature.shape[0]
        enc_gene_cell_g = dgl.graph((np.concatenate([gene, cell_idx]), np.concatenate([cell_idx, gene])))
        dec_gene_cell_g = dgl.graph((gene, cell_idx))

        rating = rpkm_feature[gene, cell]
        pairs = np.stack([gene, cell, rating], axis=1)
        pd_pairs = pd.DataFrame(pairs, columns=['gene_id', 'cell_id', 'rating'])

        return pd_pairs, enc_gene_cell_g, dec_gene_cell_g

    def _generate_pair_value(self, rating_info):
        rating_pairs = (
            np.array(
                [
                    self.global_gene_id_map[ele]
                    for ele in rating_info["gene_id"]
                ],
                dtype=np.int64,
            ),
            np.array(
                [
                    self.global_cell_id_map[ele]
                    for ele in rating_info["cell_id"]
                ],
                dtype=np.int64,
            ),
        )
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(
            self, rating_pairs, rating_values, add_support=False
    ):
        gene_cell_R = np.zeros(
            (self._num_gene, self._num_cell), dtype=np.float32
        )
        gene_cell_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {"gene": self._num_gene, "cell": self._num_cell}
        rating_row, rating_col = rating_pairs

        for rating in self.possible_rating_values:
            # if rating < len(self.possible_rating_values) - 1:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update(
                {
                    ("gene", str(rating), "cell"): (rrow, rcol),
                    ("cell", "rev-%s" % str(rating), "gene"): (rcol, rrow),
                }
            )

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert (
                len(rating_pairs[0])
                == sum([graph.num_edges(et) for et in graph.etypes]) // 2
        )

        if add_support:

            def _calc_norm(x):
                x = x.numpy().astype("float32")
                x[x == 0.0] = np.inf
                x = th.FloatTensor(1.0 / np.sqrt(x))
                return x.unsqueeze(1)

            gene_ci = []
            gene_cj = []
            cell_ci = []
            cell_cj = []

            for r in self.possible_rating_values:
                r = to_etype_name(r)
                gene_ci.append(graph["rev-%s" % r].in_degrees())
                cell_ci.append(graph[r].in_degrees())

                gene_cj.append(graph[r].out_degrees())
                cell_cj.append(graph["rev-%s" % r].out_degrees())

            gene_ci = _calc_norm(sum(gene_ci))
            cell_ci = _calc_norm(sum(cell_ci))

            gene_cj = _calc_norm(sum(gene_cj))
            cell_cj = _calc_norm(sum(cell_cj))

            graph.nodes["gene"].data.update({"ci": gene_ci, "cj": gene_cj})
            graph.nodes["cell"].data.update({"ci": cell_ci, "cj": cell_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        gene_cell_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_gene, self.num_cell),
            dtype=np.float32,
        )
        g = dgl.bipartite_from_scipy(
            gene_cell_ratings_coo, utype="_U", etype="_E", vtype="_V"
        )
        return dgl.heterograph(
            {("gene", "rate", "cell"): g.edges()},
            num_nodes_dict={"gene": self.num_gene, "cell": self.num_cell},
        )

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_gene(self):
        return self._num_gene

    @property
    def num_cell(self):
        return self._num_cell


class GeneData:
    def __init__(self, rpkm_path, label_path, divide_path, TF_num, gene_emb_path, cell_emb_path, istime, ish5=False,
                 gene_list_path=None, data_name=None, TF_random=False, save=False):
        self.gene_cell_src = None
        self.gene_cell_dst = None
        self.istime = istime
        self.data_name = data_name
        self.TF_random = TF_random
        self.save = save

        if not istime:
            if not ish5:
                self.df = pd.read_csv(rpkm_path, header='infer', index_col=0)
            else:
                self.df = pd.read_hdf(rpkm_path, key='/RPKMs').T

        else:
            time_h5 = []
            files = os.listdir(rpkm_path)
            for i in range(len(files)):
                if self.data_name.lower() == 'mesc1':
                    time_pd = pd.read_hdf(rpkm_path + 'RPM_' + str(i) + '.h5', key='/RPKM')
                else:
                    time_pd = pd.read_hdf(rpkm_path + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
                time_h5.append(time_pd)
            train_data = pd.concat(time_h5, axis=0, ignore_index=True)
            self.df = train_data.T
        self.origin_data = self.df.values

        self.df.columns = self.df.columns.astype(str)
        self.df.index = self.df.index.astype(str)
        self.df.columns = self.df.columns.str.upper()
        self.df.index = self.df.index.str.upper()
        self.cell_to_idx = dict(zip(self.df.columns.astype(str), range(len(self.df.columns))))
        self.gene_to_idx = dict(zip(self.df.index.astype(str), range(len(self.df.index))))

        self.gene_to_name = {}
        if gene_list_path:
            gene_list = pd.read_csv(gene_list_path, header=None, sep='\s+')
            gene_list[0] = gene_list[0].astype(str)
            gene_list[1] = gene_list[1].astype(str)
            gene_list[0] = gene_list[0].str.upper()
            gene_list[1] = gene_list[1].str.upper()
            self.gene_to_name = dict(zip(gene_list[0].astype(str), gene_list[1].astype(str)))

        self.start_index = []
        self.end_index = []
        self.gene_emb = np.load(gene_emb_path)
        self.cell_emb = np.load(cell_emb_path)
        self.all_emb = np.concatenate((self.gene_emb, self.cell_emb), axis=0)

        self.key_list = []
        self.gold_standard = {}
        self.datas = []
        self.gene_key_datas = []

        self.cell_datas = []
        self.labels = []
        self.idx = []

        self.geneHaveCell = collections.defaultdict(list)
        self.node_src = []
        self.node_dst = []

        self.getStartEndIndex(divide_path)
        self.getLabel(label_path)
        self.getGeneCell(self.df)

        self.getTrainTest(TF_num)

    def getStartEndIndex(self, divide_path):
        tmp = []
        with open(divide_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                tmp.append(int(line[0]))
        self.start_index = tmp[:-1]
        self.end_index = tmp[1:]

    def getLabel(self, label_path):
        s = open(label_path, 'r')
        for line in s:
            line = line.split()
            gene1 = line[0]
            gene2 = line[1]
            label = line[2]

            key = str(gene1) + "," + str(gene2)
            if key not in self.gold_standard.keys():
                self.gold_standard[key] = int(label)
                self.key_list.append(key)
            else:
                if label == 2:
                    # print(label)
                    pass
                else:
                    self.gold_standard[key] = int(label)
                self.key_list.append(key)
        s.close()
        print(len(self.key_list))
        print(len(self.gold_standard.keys()))
        # exit()

    def getTrainTest(self, TF_num):

        TF_order = list(range(0, len(self.start_index)))
        if self.TF_random:
            np.random.seed(42)
            np.random.shuffle(TF_order)
        print("TF_order", TF_order)
        index_start_list = np.asarray(self.start_index)
        index_end_list = np.asarray(self.end_index)
        index_start_list = index_start_list[TF_order]
        index_end_list = index_end_list[TF_order]

        print(index_start_list)
        print(index_end_list)
        # s = open(self.data_name + '_representation/gene_pairs.txt', 'w')
        # ss = open(self.data_name + '_representation/divide_pos.txt', 'w')
        pos_len = 0
        # ss.write(str(0) + '\n')
        for i in range(TF_num):
            name = self.data_name + '_representation/'
            # if os.path.exists(self.data_name + '_representation/' + str(i) + '_xdata.npy'):
            if self.save:
                x_data = np.load(name + str(i) + '_xdata.npy')
                h_data = np.load(name + str(i) + '_hdata.npy')
                y_data = np.load(name + str(i) + '_ydata.npy')
                gene_key_data = np.load(name + str(i) + '_gene_key_data.npy')

                self.datas.append(x_data)
                self.labels.append(y_data)
                self.gene_key_datas.append(gene_key_data)
                self.h_datas.append(h_data)
                continue

            start_idx = index_start_list[i]
            end_idx = index_end_list[i]

            print(i)
            print(start_idx, end_idx)

            this_datas = []
            this_key_datas = []
            this_labels = []

            for line in self.key_list[start_idx:end_idx]:

                label = self.gold_standard[line]
                gene1, gene2 = line.split(',')
                gene1 = gene1.upper()
                gene2 = gene2.upper()
                if int(label) != 2:
                    this_key_datas.append([gene1.lower(), gene2.lower(), label])
                    # s.write(gene1 + '\t' + gene2 + '\t' + str(label) + '\n')
                    if not self.gene_to_name:
                        gene1_idx = self.gene_to_idx[gene1]
                        gene2_idx = self.gene_to_idx[gene2]
                    else:
                        gene1_index = self.gene_to_name[gene1]
                        gene2_index = self.gene_to_name[gene2]
                        gene1_idx = self.gene_to_idx[gene1_index]
                        gene2_idx = self.gene_to_idx[gene2_index]

                    gene1_emb = self.gene_emb[gene1_idx]
                    gene2_emb = self.gene_emb[gene2_idx]

                    gene1_emb = np.expand_dims(gene1_emb, axis=0)
                    gene2_emb = np.expand_dims(gene2_emb, axis=0)

                    gene1_cells = self.geneHaveCell[gene1_idx]
                    if len(gene1_cells) == 0:
                        gene1_cells_emb = np.zeros(256)
                    else:
                        gene1_cells_emb = self.cell_emb[gene1_cells]
                        gene1_cells_emb = np.mean(gene1_cells_emb, axis=0)

                    gene2_cells = self.geneHaveCell[gene2_idx]
                    if len(gene2_cells) == 0:
                        gene2_cells_emb = np.zeros(256)
                    else:
                        gene2_cells_emb = self.cell_emb[gene2_cells]
                        gene2_cells_emb = np.mean(gene2_cells_emb, axis=0)

                    gene1_cells_emb = np.expand_dims(gene1_cells_emb, axis=0)
                    gene2_cells_emb = np.expand_dims(gene2_cells_emb, axis=0)

                    gene_emb = np.concatenate((gene1_emb, gene2_emb, gene1_cells_emb, gene2_cells_emb), axis=0)
                    this_datas.append(gene_emb)

                    this_labels.append(label)
            pos_len += len(this_datas)
            # ss.write(str(pos_len) + '\n')

            this_datas = np.asarray(this_datas)

            print(this_datas.shape)

            this_labels = np.asarray(this_labels)
            this_key_datas = np.asarray(this_key_datas)
            if self.save:
                if not os.path.exists(name):
                    os.mkdir(name)
                np.save(name + str(i) + '_xdata.npy', this_datas)
                np.save(name + str(i) + '_ydata.npy', this_labels)
                np.save(name + str(i) + '_gene_key_data.npy', this_key_datas)
                print(this_datas.shape, this_labels.shape)

            self.datas.append(this_datas)
            self.labels.append(this_labels)
            self.gene_key_datas.append(this_key_datas)
        # s.close()
        # ss.close()

    def getGeneCell(self, df):
        for i in range(df.shape[0]):
            j_nonzero = np.nonzero(df.iloc[i, :].values)[0]
            if len(j_nonzero) == 0:
                continue
            self.geneHaveCell[i].extend(j_nonzero)
