import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, epochs, lr=1e-4, graph_loss_weight=1, device='cuda'):
    print(f'start training: lr: {lr}, graph_loss_weight: {graph_loss_weight}, device: {device}')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.cuda()
            
            recon_x, graph_feat = model(data)
            gt_graph_feat = model.gnn(data)
            
            #print('shape: ', data.shape, recon_x.shape, data.shape, graph_feat.shape, gt_graph_feat.shape)
            #print(recon_x, data)
            recon_loss = criterion(recon_x, data)
            graph_loss = criterion(graph_feat, gt_graph_feat)
            agg_loss = recon_loss + graph_loss_weight * graph_loss

            optimizer.zero_grad()
            agg_loss.backward()
            optimizer.step()

            total_loss += recon_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')