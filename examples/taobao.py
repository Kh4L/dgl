import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import pandas as pd
import dgl

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score

import dgl

import scipy.sparse


# TODO: move to separate file
class Taobao(dgl.data.DGLDataset):
    url = ('https://alicloud-dev.oss-cn-hangzhou.aliyuncs.com/'
           'UserBehavior.csv.zip')

    def __init__(
        self,
        root: str,
        add_item_category_rel: bool = True,
        add_item_item_rel: bool = False,
        add_edge_feat: bool = False,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.add_item_category_rel = add_item_category_rel
        self.add_item_item_rel = add_item_item_rel
        self.add_edge_feat = add_edge_feat
        super().__init__(name='taobao', raw_dir=root, save_dir=root,
                         force_reload=force_reload)
        self.transform = transform

    def download(self):
        path = dgl.data.utils.download(self.url, self.raw_dir)
        dgl.data.utils.extract_archive(path, self.raw_dir)
        os.remove(path)

    def process(self):
        raw_path = os.path.join(self.raw_dir, 'UserBehavior.csv')
        cols = ['userId', 'itemId', 'categoryId', 'behaviorType', 'timestamp']
        df = pd.read_csv(raw_path, names=cols)

        # Time representation (YYYY.MM.DD-HH:MM:SS -> Integer)
        # start: 1511539200 = 2017.11.25-00:00:00
        # end:   1512316799 = 2017.12.03-23:59:59
        start = 1511539200
        end = 1512316799
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        df = df.drop_duplicates()

        behavior_dict = {'pv': 0, 'cart': 1, 'buy': 2, 'fav': 3}
        df['behaviorType'] = df['behaviorType'].map(behavior_dict)

        num_entries = {}
        for name in ['userId', 'itemId', 'categoryId']:
            value, df[name] = np.unique(df[[name]].values, return_inverse=True)
            num_entries[name] = value.shape[0]


        # Create heterogeneous graph
        graph_data = {
            ('user', 'u_i', 'item'): (torch.from_numpy(df['userId'].values),
                                      torch.from_numpy(df['itemId'].values)),
            ('item', 'i_u', 'user'): (torch.from_numpy(df['itemId'].values),
                                      torch.from_numpy(df['userId'].values)),
        }

        if self.add_item_item_rel:
            print("Processing item<>item rel with scipy...")
            tic = time.time()
            u_i_row = torch.from_numpy(df['userId'].values)
            u_i_col = torch.from_numpy(df['itemId'].values)
            edge_attr = torch.ones(u_i_row.size(0), device="cpu")
            sci_mat = scipy.sparse.coo_matrix((edge_attr.numpy(), (u_i_row.numpy(), u_i_col.numpy())), (num_entries['userId'], num_entries['itemId']))
            comat = sci_mat.T @ sci_mat
            comat.setdiag(0)
            comat = comat >= 3.
            comat = comat.tocoo()
            i_i_row = torch.from_numpy(comat.row).to(torch.long)
            i_i_col = torch.from_numpy(comat.col).to(torch.long)
            graph_data[('item', 'i_i', 'item')] = (i_i_row, i_i_col)
            print(f"Processed item<>item rel in {time.time()-tic}s!")

        num_nodes_dict = {
            'user': num_entries['userId'],
            'item': num_entries['itemId'],
        }

        if self.add_item_category_rel:
            graph_data[('item', 'i_c', 'category')] = (
                 torch.from_numpy(df[['itemId', 'categoryId']].drop_duplicates()['itemId'].values),
                 torch.from_numpy(df[['itemId', 'categoryId']].drop_duplicates()['categoryId'].values))
            num_nodes_dict['category'] = num_entries['categoryId']

        self.graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

        if self.add_edge_feat:
            self.graph.edges[('user', 'u_i', 'item')].data['time'] = torch.from_numpy(df['timestamp'].values)
            self.graph.edges[('user', 'u_i', 'item')].data['behavior'] = torch.from_numpy(df['behaviorType'].values)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.graph)
        return self.graph

    def __len__(self):
        return 1

    def has_cache(self):
        graph_path = os.path.join(self.save_dir, 'taobao_dgl.bin')
        return os.path.exists(graph_path)

    def save(self):
        dgl.save_graphs(os.path.join(self.save_dir, 'taobao_dgl.bin'), [self.graph])

    def load(self):
        graphs, _ = dgl.load_graphs(os.path.join(self.save_dir, 'taobao_dgl.bin'))
        self.graph = graphs[0]


# starting taobao example


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, in_feats, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, hidden_channels, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_channels, out_channels, aggregator_type='mean')

    def forward(self, blocks, x):
        x = self.conv1(blocks[0]['i_i'], x).relu()
        x = self.conv2(blocks[1]['i_i'], x).relu()
        return x

class UserGNNEncoder(torch.nn.Module):
    def __init__(self, in_feats, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, hidden_channels, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv((hidden_channels, hidden_channels), hidden_channels, aggregator_type='mean')
        self.conv3 = dgl.nn.SAGEConv((hidden_channels, hidden_channels), hidden_channels, aggregator_type='mean')
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, blocks, x_dict):
        item_x = self.conv1(blocks[0]['i_i'],  x_dict['item']).relu()
        user_x = self.conv2(blocks[0]['i_u'], (x_dict['item'], x_dict['user'])).relu()
        # we care only about the dst_nodes in blocks[1] 
        user_x = user_x[:blocks[1].num_dst_nodes('user')]
        user_x = self.conv3(blocks[1]['i_u'], (item_x, user_x)).relu()
        return self.lin(user_x)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, pos_graph, neg_graph):
        pos_src, pos_dst = pos_graph.edges(etype='u_i')
        neg_src, neg_dst = neg_graph.edges(etype='u_i')

        z_src_all = torch.cat((z_src[pos_src], z_src[neg_src]))
        z_dst_all = torch.cat((z_dst[pos_dst], z_dst[neg_dst]))
        z = torch.cat((z_src_all, z_dst_all), dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
        self.item_emb = torch.nn.Embedding(num_items, hidden_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, blocks, pos_graph, neg_graph):
        x_dict = {
            'item': self.item_emb(blocks[0].srcdata['x']['item']),
            'user': self.user_emb(blocks[0].dstdata['x']['user']),
        }
        z_dict = {}
        z_dict['item'] = self.item_encoder(blocks, x_dict['item'])
        z_dict['user'] = self.user_encoder(blocks, x_dict)
        return self.decoder(z_dict['user'], z_dict['item'], pos_graph, neg_graph)

def run(rank, world_size):
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    # Load data
    dataset = Taobao(root='Taobao', add_item_item_rel=True, add_item_category_rel=False)
    graph = dataset[0]

    graph.nodes['user'].data["x"] = torch.arange(0, graph.num_nodes('user'))
    graph.nodes['item'].data["x"] = torch.arange(0, graph.num_nodes('item'))

    # TODO split graph into train val test
    #train_eid, val_eid, test_eid = dgl.data.utils.split_dataset(graph)
    
    # TODO check if we should replace negative_sampler.Uniform( by custom one
    class NegativeSampler(object):
        def __init__(self, g, k):
            # caches the probability distribution
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes}
            self.k = k

        def __call__(self, g, eids_dict):
            result_dict = {}
            for etype, eids in eids_dict.items():
                src, _ = g.find_edges(eids, etype=etype)
                src = src.repeat_interleave(self.k)
                dst = self.weights[etype].multinomial(len(src), replacement=True)
                result_dict[etype] = (src, dst)
            return result_dict


    sampler = dgl.dataloading.NeighborSampler([8, 4])
    edge_pred_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(1))
    
    # TODO figure out if we need this in `edge_pred_sampler`:
    # exclude='reverse_types',
    # reverse_etypes={'u_i': 'i_u', 'i_u': 'u_i'})
    
    train_eid_dict = {
        etype: graph.edges(etype=etype, form='eid')
        for etype in graph.canonical_etypes}

    # Create dataloaders
    if world_size > 1:
        # TODO implem
        train_dataloader = dgl.dataloading.DistDataLoader(
            dataset=train_eid_dict,
            graph=graph,
            batch_size=2048,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            sampler=sampler
        )
    else:
        train_dataloader = dgl.dataloading.DataLoader(
            graph, train_eid_dict, edge_pred_sampler,
            batch_size=2048,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )


    val_dataloader =dgl.dataloading.DataLoader(
        graph, train_eid_dict, edge_pred_sampler,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    test_dataloader = dgl.dataloading.DataLoader(
        graph, train_eid_dict, edge_pred_sampler,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    # Create model
    model = Model(
        num_users=graph.num_nodes('user'),
        num_items=graph.num_nodes('item'),
        hidden_channels=64,
        out_channels=64
    ).to(device)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1, 21):
        loss = train(model, optimizer, train_dataloader, device)
        if rank == 0 or world_size == 1:
            val_auc = test(model, val_dataloader, device)
            test_auc = test(model, test_dataloader, device)
            print(f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val: {val_auc:.4f}, '
                  f'Test: {test_auc:.4f}')

def train(model, optimizer, dataloader, device):
    model.train()
    total_loss = total_examples = 0
    for _, pos_graph, neg_graph, blocks in tqdm.tqdm(dataloader):
        blocks = [b.to(device) for b in blocks]
        
        optimizer.zero_grad()
        logits = model(blocks, pos_graph, neg_graph)
        # 1/2 pos 1/2 neg logits
        labels = torch.zeros(logits.size(0), device=logits.device)
        labels[:logits.size(0) // 2] = 1

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) 
        total_examples += len(logits)
    return total_loss / total_examples

@torch.no_grad()
def test(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for _, pos_graph, neg_graph, blocks in tqdm.tqdm(dataloader):
        blocks = [b.to(device) for b in blocks]
        logits = model(blocks, pos_graph, neg_graph)
        labels = torch.zeros(logits.size(0))
        labels[:logits.size(0) // 2] = 1
        y_true.append(labels)
        y_pred.append(logits.sigmoid().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return roc_auc_score(y_true, y_pred)

if __name__ == '__main__':
    world_size = 1
    # TODO(spanev): uncomment
    # world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f'Let\'s use {world_size} GPUs!')
        mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    else:
        print('Running on a single GPU or CPU.')
        run(0, 1)
