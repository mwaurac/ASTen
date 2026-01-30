# ASTen (A Small Tensor Library)

ASTen is a small, educational tensor library inspired by PyTorch. It is designed to help users understand the internal workings of the Pytorch framework by providing a simplified implementation of core components.

## Inspiration
- After reading the Pytorch internals by ezyang, I was motivated to build a similar library. I wanted something simple but a reflection of Pytorch itself. I have been using AI to navigate through the pytorch codebase and also to debug my C/C++ code cause I am new to C and C++

### Current Features
- **Tensor:** A multi-dimensional array object.
- **View:** Create a new tensor that is a view of an existing tensor.
- **Reshape** View that works on non-contiguous tensors.

### Planned Features
- [ ] Mathematical operations
- [ ] Autograd engine
- [ ] CUDA Support

## Project Structure

The project is organized into the following directories, mimicking a simplified PyTorch structure:

```
.
├── ASTen/
├── aten/
│   └── native/
├── c10/
│   └── core/
├── setup.py
└── README.md
```

## Installation

You can install ASTen from the source:

```bash
pip install -e .
```

## Usage

Here is a simple example of how to create a tensor and use the `view` operation:

```python
import ASTen
import numpy as np

# Create a tensor from a numpy array
data = np.array([1, 2, 3, 4, 5, 6])
x = ASTen.tensor(data)

print(f"Original tensor shape: {x.shape}")

# Create a view of the tensor
y = x.view((2, 3))
print(f"Viewed tensor shape: {y.shape}")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.