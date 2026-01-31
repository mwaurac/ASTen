import ASTen
import numpy as np

# Test view operation
a = np.arange(6, dtype=np.float32)
t = ASTen.Tensor(a)

print("Original tensor shape:", t.shape)
print("Original tensor: ", t)

# Reshape to (2, 3)
t_view = t.view([2, 3])
print("Viewed tensor shape:", t_view.shape)
print("Viewed tensor: ", t_view.numpy())

# Reshape to (3, 2)
t_view2 = t_view.reshape([1, 1, 1, 6])
print("Viewed tensor shape:", t_view2.shape)
print("Reshaped tensor:", t_view2.numpy())

# test permute
per_tensor = ASTen.Tensor([[1, 2], [3, 4], [5, 6]])
print(per_tensor.shape)
t_perm = per_tensor.permute(1, 0)
print(t_perm.numpy())
print(t_perm.shape)