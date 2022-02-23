#ifndef pooling_layers_h
#define pooling_layers_h

template <typename IN_T=float, typename T=size_t>
void _align_memory_before_conv(Matrix<IN_T> &x, std::vector<T> kernel_shape) {
    std::vector<T> x_shape = x.shape();
    printf("b4 resize: ");print_vec(x_shape);
    std::vector<T> x_strides = x.strides();
    x.resize({x_shape[0], x_shape[1], x_shape[2], kernel_shape[3]});
    std::vector<T> new_x_shape = x.shape();
    printf("after resize: ");print_vec(x_shape);
    for (int i=(x_shape[0]-1); i>-1; i--) {
        for (int j=(x_shape[1]-1); j>-1; j--) {
            for (int k=(x_shape[2]-1); k>-1; k--) {
                for (int l=(new_x_shape[3]-1); l>-1; l--) {
                    if (l>=x_shape[3]) {
                        x.set(0, i, j, k, l);
                    } else {
                        x.set(x.m_data[i*x_strides[0]+j*x_strides[1]+k*x_strides[2]+l], i, j, k, l);
                    }
                }
            }
        }
    }
}


#endif