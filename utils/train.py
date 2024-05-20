import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, epochs, lr=1e-4, graph_loss_weight=1, device='cuda'):
    print(f'start training: lr: {lr}, graph_loss_weight: {graph_loss_weight}, device: {device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            
            recon_x, graph_feat = model(data)
            gt_graph_feat = model.gnn(data)
            
            recon_loss = nn.BCELoss(recon_x, data)
            graph_loss = nn.BCELoss(graph_feat, gt_graph_feat)
            agg_loss = recon_loss + graph_loss_weight * graph_loss

            optimizer.zero_grad()
            agg_loss.backward()
            optimizer.step

            total_loss += recon_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')