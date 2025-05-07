# -----------------------------
# 1. Importaciones
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 2. Dataset simulado
def generar_dataset_simulado(num_secuencias = 100, min_longitud = 512, max_longitud = 16384, dimension_embedding = 128):
    dataset = []
    for _ in range(num_secuencias):
        longitud = torch.randint(low = min_longitud, high = max_longitud, size = (1,)).item()
        secuencia = torch.randn(longitud, dimension_embedding) 
        dataset.append(secuencia)
    return dataset

dataset_simulado = generar_dataset_simulado()
print(dataset_simulado[0].shape)

# -----------------------------
# 3. Implementación del Linformer
class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, k=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.k = k
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "La dimensión debe ser divisible por el número de heads"

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.E = nn.Parameter(torch.randn(self.k, self.head_dim))
        self.F = nn.Parameter(torch.randn(self.k, self.head_dim))

        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.to_q(x).view(B, N, H, d).transpose(1, 2)
        k = self.to_k(x).view(B, N, H, d).transpose(1, 2)
        v = self.to_v(x).view(B, N, H, d).transpose(1, 2)

        k_proj = torch.einsum('b h n d, k d -> b h k d', k, self.E)
        v_proj = torch.einsum('b h n d, k d -> b h k d', v, self.F)

        attn = torch.einsum('b h n d, b h k d -> b h n k', q, k_proj) / (d ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h n k, b h k d -> b h n d', attn, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)

# -----------------------------
# 4. Testeo del modelo
# Tomamos una secuencia del dataset
secuencia = dataset_simulado[0]
secuencia = secuencia.unsqueeze(0)  # Agregamos dimensión batch

modelo_linformer = LinformerSelfAttention(dim=128, num_heads=8, k=256)
salida = modelo_linformer(secuencia)

print("Salida:", salida.shape)


# Volver a repetir bloque de código para entenderlo mejor. Ejercicio terminado (falta optimnización para casos reales, pero como un esquemita esta).