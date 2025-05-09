# 16. Draw PINN Architecture
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8)
ax.axis('off')

layers = [
    (1, 4),    # inputs x,y,z,t
    (3, 200),  # hidden 1
    (5, 200),  # hidden 2
    (7, 200),  # hidden 3
    (9, 200),  # hidden 4
    (11, 200), # hidden 5
    (13, 1)    # output p
]

# draw neurons + sparse connections
for i, (x, n) in enumerate(layers):
    if i == 0:
        y_pos = np.linspace(6, 2, n)
    elif i == len(layers)-1:
        y_pos = [4]
    else:
        m = min(n, 5)
        y_pos = np.linspace(6, 2, m)
    for j, y in enumerate(y_pos):
        circ = plt.Circle((x, y), 0.2, color='skyblue', ec='black')
        ax.add_patch(circ)
        if i == 0:
            labels = ['x','y','z','t']
            ax.text(x-0.3, y, labels[j], ha='right', va='center', fontsize=12)
        elif i == len(layers)-1:
            ax.text(x+0.3, y, '', ha='left', va='center', fontsize=12)
        elif len(y_pos)<n and j==len(y_pos)-1:
            ax.text(x, y-0.6, '...', ha='center', va='center', fontsize=16)
    # connections
    if i < len(layers)-1:
        nx, nn = layers[i+1]
        if i+1==len(layers)-1:
            ny_pos=[4]
        else:
            mm = min(nn,5)
            ny_pos = np.linspace(6, 2, mm)
        for yy in y_pos:
            for y2 in ny_pos:
                ax.plot([x+0.2, nx-0.2], [yy, y2], 'k-', alpha=0.1)

# layer labels
for i, (x, _) in enumerate(layers):
    if i==0:
        lbl='Input\n(x,y,z,t)'
    elif i==len(layers)-1:
        lbl='Output\n(p)'
    else:
        lbl='Hidden\n(200 neurons,\ntanh)'
    ax.text(x, 7.3, lbl, ha='center', va='bottom', fontsize=10)

# loss boxes & arrows
losses = [
    (15, 5, 'PDE Loss'),
    (15, 4, 'Boundary\nCondition Loss'),
    (15, 3, 'Initial\nCondition Loss'),
    (15, 2, 'Data Loss')
]
for lx, ly, txt in losses:
    rect = plt.Rectangle((lx-0.9, ly-0.3), 1.8, 0.6, fill=False, ec='red', lw=1.5)
    ax.add_patch(rect)
    ax.text(lx, ly, txt, ha='center', va='center', fontsize=10)
    ax.annotate('', xy=(lx-0.9, ly), xytext=(13.2, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax.text(12, 4, 'p(x,y,z,t)', ha='center', va='center', fontsize=10)

plt.title('PINN Architecture with Loss Components', fontsize=14, pad=20)
plt.savefig('pinn_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

