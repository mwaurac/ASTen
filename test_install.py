import ASTen
import numpy as np

print("Successfully imported ASTen")

# Test tensor creation
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
t = ASTen.Tensor(a)

print("Tensor created successfully")
print("Tensor ndim:", t.ndim)
print("Tensor shape:", t.shape)
print("Tensor dtype:", t.dtype)
print("Tensor requires grad:", t.requires_grad)
print("")

# Test numpy conversion
b = t.numpy()
print("Tensor converted back to numpy successfully")
print(b)
