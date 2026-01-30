#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

extern "C" {
    #include "../c10/core/tensor.h"
    #include "../c10/core/dtype.h"
    #include "../c10/core/device.h"
    #include "../aten/native/view_ops.h"
}

namespace py = pybind11;

class TensorWrapper {
public:
    Tensor* ptr;
    
    TensorWrapper(Tensor* t) : ptr(t) {
        if (ptr) tensor_retain(ptr);
    }
    
    ~TensorWrapper() {
        if (ptr) tensor_release(ptr);
    }
    
    TensorWrapper(const TensorWrapper& other) : ptr(other.ptr) {
        if (ptr) tensor_retain(ptr);
    }
    
    TensorWrapper& operator=(const TensorWrapper& other) {
        if (this != &other) {
            if (ptr) tensor_release(ptr);
            ptr = other.ptr;
            if (ptr) tensor_retain(ptr);
        }
        return *this;
    }
};

TensorWrapper* numpy_to_tensor(py::array arr, const std::string& device_str) {
    auto buf = arr.request();
    
    DType dtype;
    if (buf.format == py::format_descriptor<float>::format()) {
        dtype = DTYPE_FLOAT32;
    } else if (buf.format == py::format_descriptor<double>::format()) {
        dtype = DTYPE_FLOAT64;
    } else if (buf.format == py::format_descriptor<int32_t>::format()) {
        dtype = DTYPE_INT32;
    } else if (buf.format == py::format_descriptor<int64_t>::format()) {
        dtype = DTYPE_INT64;
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
    
    Device device = device_str == "cuda" ? device_cuda(0) : device_cpu();
    
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    Tensor* t = tensor_empty(shape.data(), shape.size(), dtype, device);
    
    // Copy data
    if (device.type == DEVICE_CPU) {
        void* src = buf.ptr;
        void* dst = storage_data(t->storage);
        size_t size = storage_size_bytes(t->storage);
        memcpy(dst, src, size);
    }
    
    return new TensorWrapper(t);
}

py::array tensor_to_numpy(const TensorWrapper& tw) {
    Tensor* t = tw.ptr;
    if (!t || t->storage->device.type != DEVICE_CPU) {
        throw std::runtime_error("Can only convert CPU tensors to numpy");
    }
    
    std::vector<ssize_t> shape(t->ndim);
    std::vector<ssize_t> strides(t->ndim);
    
    for (size_t i = 0; i < t->ndim; i++) {
        shape[i] = t->shape[i];
        strides[i] = t->strides[i] * dtype_size(t->storage->dtype);
    }
    
    void* data = storage_data(t->storage);
    
    switch (t->storage->dtype) {
        case DTYPE_FLOAT32:
            return py::array_t<float>(shape, strides, (float*)data);
        case DTYPE_FLOAT64:
            return py::array_t<double>(shape, strides, (double*)data);
        case DTYPE_INT32:
            return py::array_t<int32_t>(shape, strides, (int32_t*)data);
        case DTYPE_INT64:
            return py::array_t<int64_t>(shape, strides, (int64_t*)data);
        default:
            throw std::runtime_error("Unsupported dtype for numpy conversion");
    }
}


PYBIND11_MODULE(_C, m) {
    m.doc() = "ASTen C++ backend";
    
    // Tensor class
    py::class_<TensorWrapper>(m, "Tensor")
        .def(py::init([](py::array arr, const std::string& device_str) {
            return numpy_to_tensor(arr, device_str);
        }), py::arg("data"), py::arg("device") = "cpu")
        
        .def("numpy", &tensor_to_numpy)

        .def("reshape", [](const TensorWrapper& tw, std::vector<size_t> shape) {
            return new TensorWrapper(tensor_reshape(tw.ptr, shape.data(), shape.size()));
        })
        .def("view", [](const TensorWrapper& tw, std::vector<size_t> shape) {
            return new TensorWrapper(tensor_view(tw.ptr, shape.data(), shape.size()));
        })
        .def_property_readonly("shape", [](const TensorWrapper& tw) {
            std::vector<size_t> shape(tw.ptr->shape, tw.ptr->shape + tw.ptr->ndim);
            return shape;
        })
        
        .def_property_readonly("ndim", [](const TensorWrapper& tw) {
            return tw.ptr->ndim;
        })
        
        .def_property_readonly("dtype", [](const TensorWrapper& tw) {
            return std::string(dtype_name(tw.ptr->storage->dtype));
        })
        
        .def_property("requires_grad", 
            [](const TensorWrapper& tw) { return tw.ptr->requires_grad; },
            [](TensorWrapper& tw, bool req_grad) {
                tensor_set_requires_grad(tw.ptr, req_grad);
        });
}