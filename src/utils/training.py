import torch
from torch.optim.lr_scheduler import StepLR

def compute_pde_residual(model, x, y, z, t, phi, k, mu, ct, q0, well_pos, p_scale, p_base):
    inputs = torch.stack([x, y, z, t], dim=1).requires_grad_(True).to(x.device)
    p = model(inputs, p_scale, p_base)
    p_grad = torch.autograd.grad(p, inputs, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x, p_y, p_z, p_t = p_grad[:, 0], p_grad[:, 1], p_grad[:, 2], p_grad[:, 3]
    p_xx = torch.autograd.grad(p_x, inputs, grad_outputs=torch.ones_like(p_x), create_graph=True)[0][:, 0]
    p_yy = torch.autograd.grad(p_y, inputs, grad_outputs=torch.ones_like(p_y), create_graph=True)[0][:, 1]
    p_zz = torch.autograd.grad(p_z, inputs, grad_outputs=torch.ones_like(p_z), create_graph=True)[0][:, 2]
    sigma = 0.05
    q = q0 * torch.exp(-((x - well_pos[0])**2 + (y - well_pos[1])**2 + (z - well_pos[2])**2) / (2 * sigma**2))
    residual = phi * ct * p_t - (k / mu) * (p_xx + p_yy + p_zz) - q
    return residual

def train_model(model, x_r, y_r, z_r, t_r, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, x_d, y_d, z_d, t_d, p_d, epochs, phi, k, mu, ct, q0, well_pos, p_scale, p_base):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
    losses = {'pde': [], 'bc': [], 'ic': [], 'data': [], 'total': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        residual = compute_pde_residual(model, x_r, y_r, z_r, t_r, phi, k, mu, ct, q0, well_pos, p_scale, p_base)
        loss_pde = torch.mean(residual**2)
        inputs_b = torch.stack([x_b, y_b, z_b, t_b], dim=1).requires_grad_(True).to(x_r.device)
        p_b = model(inputs_b, p_scale, p_base)
        p_grad_b = torch.autograd.grad(p_b, inputs_b, grad_outputs=torch.ones_like(p_b), create_graph=True)[0]
        p_x_b, p_y_b, p_z_b = p_grad_b[:, 0], p_grad_b[:, 1], p_grad_b[:, 2]
        mask_x0 = (x_b == 0)
        mask_x1 = (x_b == 1)
        loss_bc_x = torch.mean(p_x_b[mask_x0]**2) + torch.mean(p_x_b[mask_x1]**2)
        mask_y0 = (y_b == 0)
        mask_y1 = (y_b == 1)
        loss_bc_y = torch.mean(p_y_b[mask_y0]**2) + torch.mean(p_y_b[mask_y1]**2)
        mask_z0 = (z_b == 0)
        mask_z1 = (z_b == 1)
        loss_bc_z = torch.mean(p_z_b[mask_z0]**2) + torch.mean(p_z_b[mask_z1]**2)
        loss_bc = loss_bc_x + loss_bc_y + loss_bc_z
        inputs_i = torch.stack([x_i, y_i, z_i, t_i], dim=1).to(x_r.device)
        p_i = model(inputs_i, p_scale, p_base)
        loss_ic = torch.mean(((p_i - p_base) / p_scale)**2)
        inputs_d = torch.stack([x_d, y_d, z_d, t_d], dim=1).to(x_r.device)
        p_d_pred = model(inputs_d, p_scale, p_base)
        loss_data = torch.mean(((p_d_pred - p_d) / p_scale)**2)
        loss = loss_pde + 100 * loss_bc + 10000 * loss_ic + 1000000 * loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses['pde'].append(loss_pde.item())
        losses['bc'].append(loss_bc.item())
        losses['ic'].append(loss_ic.item())
        losses['data'].append(loss_data.item())
        losses['total'].append(loss.item())
    return losses