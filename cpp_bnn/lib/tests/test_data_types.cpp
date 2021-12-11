#include "../utils/data_type.h"

int main() {/*
    {
        printf("2d array: \n");
        Matrix<float> x({12, 1}, 1);
        x.m_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        printf("original: \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        printf("reshape (2, 6): \n");
        x.reshape(2, 6);
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(1, 0);
        printf("transpose (1,0): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(12, 1);
        printf("reshape back to original: \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
    }
    {
        printf("2d array: \n");
        Matrix2D<float> x({12, 1}, 1);
        //x.m_data = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}};
        for (int i=0; i<12; ++i) {
            x.m_data[i] = {i+1};
        }
        printf("original: \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        printf("reshape (2, 6): \n");
        x.reshape(2, 6);
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(1, 0);
        printf("transpose (1,0): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(12, 1);
        printf("reshape back to original: \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
    }
    {
        printf("4d array: \n");
        Matrix<float> x({1, 3, 3, 2}, 1);
        x.m_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(1,18);
        printf("reshape (1, 18): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(1, 3, 3, 2);
        printf("reshape (1, 3, 3, 2): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(1,2,0,3);
        printf("transpose (1,2,0,3): \n");
        printf("ndim: %d\n", x.ndim());
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(0,1, 2,3);
        printf("transpose (0,1,2,3): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
    }*/
    {
        printf("4d array: \n");
        Matrix2D<float> x({1, 3, 3, 2}, 1);
        x.m_data[0] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        for (int i=0; i<18; ++i) {
            std::cout <<x.m_data[0][i];
        }
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(1,18);
        printf("reshape (1, 18): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.reshape(1, 3, 3, 2);
        printf("reshape (1, 3, 3, 2): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(1,2,0,3);
        printf("transpose (1,2,0,3): \n");
        printf("ndim: %d\n", x.ndim());
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
        
        x.transpose(0,1, 2,3);
        printf("transpose (0,1,2,3): \n");
        printf("strides: ");print_vec(x.strides());
        print_mat(x);
    }
    {
        PO2_5bits_t a(0);
        a.sign = 1;
        a.value = -3;
        
        printf("PO2_5bits_t --------------------------\n");
        printf("sign: %d <->", int(a.sign));
        printf("value: %d \n", a.value);
        a.value = 4;
        printf("sign: %d <->", int(a.sign));
        printf("value: %d \n", a.value);
        a.sign = -1;
        printf("sign: %d <->", int(a.sign));
        printf("value: %d \n", a.value);
        a.sign = 0;
        printf("sign: %d <->", int(a.sign));
        printf("value: %d \n", a.value);
        
    }
    
    
    return 0;
}