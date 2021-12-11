#ifndef data_type_h
#define data_type_h

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <arm_neon.h>

//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//#include <pybind11/numpy.h>

// https://en.cppreference.com/w/cpp/language/bit_field
typedef unsigned short int usint;

struct PO2_5bits_t {
    usint sign : 1;
    int8_t value : 6;
    
    void get_sign(int val) {
        if (val<0) {
            sign=1;
        } else {
            sign=0;
        }
    }
    //PO2_5bits_t(int val) : sign(0), value(val) {};
    PO2_5bits_t(const int val) : sign(0), value(val) {};
    PO2_5bits_t(int sign, int val) : value(val) {
        get_sign(sign);
    };
    void set(int sign, int val) {
        value = val;
        get_sign(sign);
    };
    /*PO2_5bits_t &operator=(int val) {
        sign = 1;
        value = val;
        return *this;
    }*/
    float operator*(float x) {
        unsigned int sign_bit = 0b10000000000000000000000000000000;
        unsigned int exponent_bit = 0b01111111100000000000000000000000;
        unsigned int fraction_bit = 0b00000000011111111111111111111111;
        
        int int_x = *(int*)&x;
        
        unsigned int _sign = ((int_x & sign_bit) != (sign<<31)) ? (1<<31):0;
        unsigned int _exponent = int_x & exponent_bit;
        unsigned int _fraction = int_x & fraction_bit;
        
        _exponent = ((_exponent>>23)+value) <<23;

        int_x = _sign|_exponent|_fraction;
        float float_x = *(float*)&int_x;
        
        return float_x;
    }
    float operator*(float16_t x) {
        unsigned short int sign_bit = 0b1000000000000000;
        unsigned short int exponent_bit = 0b0111110000000000;
        unsigned short int fraction_bit = 0b0000001111111111;
        
        //unsigned short int  _sign = 0;
        unsigned short int int_x = *(unsigned short int*)&x;
        
        unsigned short int  _sign = ((int_x & sign_bit) != (sign<<15)) ? (1<<15):0;
        unsigned short int  _exponent = int_x & exponent_bit;
        unsigned short int  _fraction = int_x & fraction_bit;
        
        _exponent = ((_exponent>>10)+value) <<10;

        int_x = _sign|_exponent|_fraction;
        float float_x = *(float16_t*)&int_x;
        
        return float_x;
    }
    float operator*(bool x_bool) {
        float16_t x = -1;
        if (x_bool) {
            x = 1;
        };
        
        unsigned short int sign_bit = 0b1000000000000000;
        unsigned short int exponent_bit = 0b0111110000000000;
        unsigned short int fraction_bit = 0b0000001111111111;
        
        unsigned short int int_x = *(unsigned short int*)&x;
        
        //unsigned short int  _sign = 0;
        //if ((int_x & sign_bit) != (sign<<15)) {
        //    _sign = (1<<15);
        //}
        unsigned short int  _sign = ((int_x & sign_bit) != (sign<<15)) ? (1<<15):0;
        //printf("after sign: %d, sign: %d, x: %d \n", _sign, (sign<<15), (int_x & sign_bit));
        unsigned short int  _exponent = int_x & exponent_bit;
        unsigned short int  _fraction = int_x & fraction_bit;
        
        _exponent = ((_exponent>>10)+value) <<10;

        int_x = _sign|_exponent|_fraction;
        float float_x = *(float16_t*)&int_x;
        
        return float_x;
    }
};

template <class T_print>
void print_PO2(const T_print &to_print) {
    std::vector<size_t> shape = to_print.shape();
    
    print_shape(to_print);
    print_stride(to_print);
    
    std::cout << "\n";
    /*if (to_print.ndim() == 4) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        std::cout << to_print(i, j, k, l) << " , ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (to_print.ndim() == 3) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    std::cout << to_print(i, j, k, 0) << " , ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else*/ if (to_print.ndim() == 2) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                std::cout << to_print(i, j, 0, 0).value << " value <-> sign";
                std::cout << to_print(i, j, 0, 0).sign << " , ";
            }
            std::cout << "\n";
        }
    } /*else if (to_print.ndim() == 1) {
        for (int i = 0; i < shape[0]; i++) {
            std::cout << to_print(i, 0, 0, 0) << " , ";
        }
        std::cout << "\n";
    }*/
};

template <class T_print>
void print_vec(const T_print &to_print) {
    size_t size = to_print.size();
    
    for (int i = 0; i<size; i++) {
        std::cout << to_print[i] << "," ;
    }
    std::cout << "\n";
};

template <class T_print>
void print_shape(const T_print &to_print) {
    std::vector<size_t> shape = to_print.shape();
    
    printf("shape: "); print_vec(shape);
};

template <class T_print>
void print_stride(const T_print &to_print) {
    std::vector<size_t> stride = to_print.strides();
    
    printf("stride: "); print_vec(stride);
};


template <class T_print>
void print_mat_bool(T_print &to_print) {
    std::vector<size_t> shape = to_print.shape();
    std::vector<size_t> strides = to_print.strides();
    
    //if (to_print.data()) {
    //    printf("Null data. \n");
    //    return;
    //}
    print_shape(to_print);
    print_stride(to_print);
    
    if (to_print.ndim() == 4) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        if (to_print(i, j, k, l)) {
                            std::cout << 1 << " , ";
                        } else {
                            std::cout << 0 << " , ";
                        }

                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (to_print.ndim() == 2) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                if (to_print(i, j, 0, 0)) {
                    std::cout << 1 << " , ";
                } else {
                    std::cout << 0 << " , ";
                }
            }
            std::cout << "\n";
        }
    }
}

template <class T_print>
void print_mat(const T_print &to_print) {
    std::vector<size_t> shape = to_print.shape();
    
    print_shape(to_print);
    print_stride(to_print);
    
    std::cout << "\n";
    if (to_print.ndim() == 4) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        std::cout << to_print(i, j, k, l) << " , ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (to_print.ndim() == 3) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    std::cout << to_print(i, j, k, 0) << " , ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (to_print.ndim() == 2) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                std::cout << to_print(i, j, 0, 0) << " , ";
            }
            std::cout << "\n";
        }
    } else if (to_print.ndim() == 1) {
        for (int i = 0; i < shape[0]; i++) {
            std::cout << to_print(i, 0, 0, 0) << " , ";
        }
        std::cout << "\n";
    }
};


template <class T>
class Matrix {
    private:
        std::vector<size_t> ori_shape;
    
    public:
        std::vector<T>      m_data;     // data array
        std::vector<size_t> m_shape;    // number of entries in each dimensions
        std::vector<size_t> m_strides;  // stride length for each index
        std::vector<size_t> axis_order = {0, 1, 2, 3};
        std::vector<bool> reverse_flag = {false, false, false, false};
    
        ~Matrix<T>()=default;
        // default constructor
        // -------------------
        Matrix<T>()=default;
    
        // copy constructor
        //https://www.fluentcpp.com/2018/07/17/how-to-construct-c-objects-without-making-copies/
        // ----------------
        Matrix (Matrix<T> &&)=default;
        Matrix (Matrix<T> &)=default;
        Matrix (const Matrix<T> &) = default;
        /*Matrix (Matrix<T> & mat_in): 
            ori_shape(mat_in.m_shape), 
            m_shape(mat_in.m_shape), 
            m_strides(mat_in.m_strides),
            axis_order(mat_in.axis_order), reverse_flag(mat_in.reverse_flag) {
                if (m_data.data() == mat_in.m_data.data()) {
                    printf("same location");
                }
                if (std::is_lvalue_reference<Matrix>::value) {
                    printf("lvalue");
                    //std::remove_reference_t<U>&, 
                    std::vector<T> tmp(std::move(m_data));
                    m_data = std::move(mat_in.m_data);
                    mat_in.m_data = std::move(tmp);
                    //m_data = mat_in.m_data;        
                } else {
                    printf("rvalue");
                    //std::remove_reference_t<U>&&>;
                    ;
                    printf("before size: %d \n", m_data.size());
                    //m_data = std::move(std::remove_reference_t<Matrix<T>&>mat_in.m_data);
                    m_data = mat_in.m_data;
                    //std::vector<T> tmp(std::move(m_data));
                    //m_data = std::move(mat_in.m_data);
                    //mat_in.m_data = std::move(tmp);
                    printf("after size: %d \n", m_data.size());
                    //mat_in.m_data = NULL;
                    //m_data = mat_in.m_data;        
                }
                //std::forward<Matrix<T>>(mat_in))
                //std::forward<std::vector<T>>(
            };*/
        Matrix<T>& operator= (Matrix<T> &&)=default;
        Matrix<T>& operator= (Matrix<T> &)=default;
            
        // helpers
        size_t set_size() {
            if (ndim() == 0) {
                return 0;
            }
            int size = 1;
            for (int i = 0; i<ndim(); i++) {
                if (m_shape[i] != 0) {
                    size *= m_shape[i];
                }
            }
            return size;
        }
    
        
        void set_stride() {
            for (int i = 4; i>0; i--) {
                if (i > ndim()) {
                    m_strides[i-1] = 0;
                } else if (i == ndim()) {
                    m_strides[i-1] = 1;
                } else {
                    m_strides[i-1] = m_strides[i]*m_shape[i];
                }
                //printf("iter [%d]: %d\n", i, m_strides[i-1]);
            }
        }
        /*
        void set_stride(size_t size) {
            for (int i = 0; i<ndim(); i++) {
                size /= m_shape[i];
                m_strides[i] = size;
            }
            return size;
        }*/
    
        // constructor
        // -----------

        Matrix<T>(const std::vector<size_t> &shape, const T value=0, const T *data=NULL) {
            if ( shape.size()<1 || shape.size()>4 ) {
                throw std::runtime_error("Input should be 1-D, 2-D, 3-D, or 4-D");
            }

            // store 'm_strides' and 'm_shape' always in 3-D,
            // use unit-length for "extra" dimensions (> 'shape.size()')
            while ( m_shape  .size()<4 ) { m_shape  .push_back(0); }
            while ( m_strides.size()<4 ) { m_strides.push_back(0); }
            
            

            for ( int i=0 ; i<shape.size() ; i++ ) {m_shape[i] = shape[i];}
            ori_shape = m_shape;
            
            int _size = set_size();
            set_stride();
            
            
            m_data.resize(_size, value);
            if ( data!=NULL )
                for ( int i=0 ; i<_size ; i++ )
                    m_data[i] = data[i];
        };

        void resize(const std::vector<size_t> &shape, int init_value=0, const T *data=NULL) {
            if ( shape.size()<1 || shape.size()>4 ) {
                throw std::runtime_error("Input should be 1-D, 2-D, 3-D, or 4-D");
            }

            // store 'm_strides' and 'm_shape' always in 3-D,
            // use unit-length for "extra" dimensions (> 'shape.size()')
            while ( m_shape  .size()<4 ) { m_shape  .push_back(0); }
            while ( m_strides.size()<4 ) { m_strides.push_back(0); }

            for ( int i=0 ; i<shape.size() ; i++ ) {m_shape[i] = shape[i];}
            ori_shape = m_shape;
            
            int _size = set_size();
            set_stride();
            
            m_data.resize(_size, T(init_value));
            if ( data!=NULL )
                for ( int i=0 ; i<_size ; i++ )
                    m_data[i] = data[i];
        };

        // index operators
        // ---------------

        typename std::vector<T>::reference operator[](size_t i) { return m_data[i]; };
        //constexpr typename std::vector<T>::reference operator[](size_t i) const { return m_data[i]; };
        const T& operator[](size_t i) const { return m_data[i]; };
    
        T read_data(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix*>(this)->read_data(i, j, k, l);
        };
    
        const T read_data(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            return m_data[i*m_strides[axis_order[0]] + j*m_strides[axis_order[1]] + k*m_strides[axis_order[2]] + l*m_strides[axis_order[3]]];
        };
    
    
        T operator()(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix*>(this)->operator()(i, j, k, l);
        };

        const T operator()(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            if (reverse_flag[0] || reverse_flag[1] || reverse_flag[2] || reverse_flag[3]) {
                return reverse_index(i, j, k, l);
            } else {
                return read_data(i, j, k, l);
            }
        };
        
        T reverse_index(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix*>(this)->reverse_index(i, j, k, l);
        }
    
        const T reverse_index(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            size_t i_r, j_r, k_r, l_r;
            if (reverse_flag[0] != 0) {
                i_r = m_shape[0]-1-i;
            } else {
                i_r = i;
            }
            if (reverse_flag[1] != 0) {
                j_r = m_shape[1]-1-j;
            } else {
                j_r = j;
            }
            if (reverse_flag[2] != 0) {
                k_r = m_shape[2]-1-k;
            } else {
                k_r = k;
            }
            if (reverse_flag[3] != 0) {
                l_r = m_shape[3]-1-l;
            } else {
                l_r = l;
            }
            return read_data(i_r, j_r, k_r, l_r);
        }

        void set(T value, size_t i, size_t j=0, size_t k=0, size_t l=0) {  
            m_data[i*m_strides[axis_order[0]] + j*m_strides[axis_order[1]] + k*m_strides[axis_order[2]] + l*m_strides[axis_order[3]]] = value; 
        };
    
        void set_vector(T value, size_t i) {  
            m_data[i] = value; 
        };
    
        T get_vector(size_t i) {  
            return m_data[i]; 
        };
    
        // iterators
        // ---------
        auto begin()       { return m_data.begin(); }
        auto begin() const { return m_data.begin(); }
        auto end()         { return m_data.end();   }
        auto end() const   { return m_data.end();   }

        // return pointer to data
        // ----------------------
        T* data(void) { return m_data.data(); };
        const T* data(void) const { return m_data.data(); };

        // return shape array [ndim]
        // -------------------------
        std::vector<size_t> shape(size_t ndim=0) const {
            if ( ndim == 0 )
                ndim = this->ndim();

            std::vector<size_t> ret(ndim);

            for ( size_t i = 0 ; i < ndim ; ++i )
                ret[i] = m_shape[i];

            return ret;
        };

        // return strides array [ndim]
        // ---------------------------
        std::vector<size_t> strides(bool bytes=false) const {
            size_t ndim = this->ndim();
            std::vector<size_t> ret(ndim);
            
            for ( size_t i = 0 ; i < ndim ; ++i )
                ret[i] = m_strides[i];
            
            if ( bytes )
                for ( size_t i = 0 ; i < ndim ; ++i )
                    ret[i] *= sizeof(T);
            
            return ret;
        };
    
        // return size
        // -----------
        size_t size ( void ) const { return m_data.size(); };

        // return number of dimensions
        // ---------------------------
        size_t ndim ( void ) const {
            size_t i;
            for ( i = 4 ; i > 0 ; i-- )
                if ( m_shape[i-1] != 0 )
                  break;
            return i;    
        };
    
        void transpose(size_t i_t=0, size_t j_t=0, size_t k_t=0, size_t l_t=0) {
            axis_order[0] = i_t;
            axis_order[1] = j_t;
            axis_order[2] = k_t;
            axis_order[3] = l_t;
            
            std::vector<size_t> tmp_shape;
            if (ndim() == 1) {
                tmp_shape = {ori_shape[i_t], 0, 0, 0};
            } else if (ndim() == 2) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], 0, 0};
            } else if (ndim() == 3) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], 0};
            } else if (ndim() == 4) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], ori_shape[l_t]};
            }
            
            m_shape = tmp_shape;
        };
        /*
        void permanent_transpose(size_t i_t=0, size_t j_t=0, size_t k_t=0, size_t l_t=0) {
            std::vector<size_t> tmp_strides(4,0);
            
            tmp_strides[0] = m_strides[i_t];
            tmp_strides[1] = m_strides[j_t];
            tmp_strides[2] = m_strides[k_t];
            tmp_strides[3] = m_strides[l_t];
            m_strides = tmp_strides;
            
            std::vector<size_t> tmp_shape;
            if (ndim() == 1) {
                tmp_shape = {ori_shape[i_t], 0, 0, 0};
            } else if (ndim() == 2) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], 0, 0};
            } else if (ndim() == 3) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], 0};
            } else if (ndim() == 4) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], ori_shape[l_t]};
            }
            
            m_shape = tmp_shape;
            ori_shape = tmp_shape;
        };
        */
    
        void reshape(size_t i_t, size_t j_t=0, size_t k_t=0, size_t l_t=0) {
            size_t _size = size();
            size_t new_size = i_t;
            if (i_t == 0) {
                throw std::runtime_error("1st value 0 error");
            }
            if (j_t != 0) {
                new_size *= j_t;
            }
            if (k_t != 0) {
                new_size *= k_t;
            }
            if (l_t != 0) {
                new_size *= l_t;
            }
            if (new_size != _size) {
                throw std::runtime_error("Reshape value error");
            }
            m_shape = {i_t, j_t, k_t, l_t};
            ori_shape = m_shape;
            set_stride();
        };
    
        void reverse(bool i_t=false, bool j_t=false, bool k_t=false, bool l_t=false) {
            reverse_flag[0] = i_t;
            reverse_flag[1] = j_t;
            reverse_flag[2] = k_t;
            reverse_flag[3] = l_t;
        };
    

}; // class Matrix



template <class T>
class Matrix2D : public Matrix<T>{
    private:
        std::vector<size_t> ori_shape;
        size_t read_shape;
    
    public:
        std::vector<std::vector<T> > m_data;     // data array
        std::vector<size_t> m_shape;    // number of entries in each dimensions
        std::vector<size_t> m_strides;  // stride length for each index
        std::vector<size_t> axis_order = {0, 1, 2, 3};
        std::vector<bool> reverse_flag = {false, false, false, false};
    
        ~Matrix2D<T>() {};
        // default constructor
        // -------------------
        Matrix2D<T>(){};
    
        // helpers
        /*
        size_t set_size() {
            if (ndim() == 0) {
                return 0;
            }
            int size = 1;
            for (int i = 0; i<ndim(); i++) {
                if (m_shape[i] != 0) {
                    size *= m_shape[i];
                }
            }
            return size/m_shape[0];
        }
        */
        
        void set_stride() {
            for (int i = 4; i>0; i--) {
                if (i > ndim()) {
                    m_strides[i-1] = 0;
                } else if (i == ndim()) {
                    m_strides[i-1] = 1;
                } else {
                    m_strides[i-1] = m_strides[i]*m_shape[i];
                }
                //printf("iter [%d]: %d\n", i, m_strides[i-1]);
            }
        }
        // constructor
        // -----------
        Matrix2D<T>(const std::vector<size_t> &shape, const T value=0, const T *data=NULL) {
            if ( shape.size()<1 || shape.size()>4 ) {
                throw std::runtime_error("Input should be 1-D, 2-D, 3-D, or 4-D");
            }

            // store 'm_strides' and 'm_shape' always in 3-D,
            // use unit-length for "extra" dimensions (> 'shape.size()')
            while ( m_shape  .size()<4 ) { m_shape  .push_back(0); }
            while ( m_strides.size()<4 ) { m_strides.push_back(0); }
            read_shape = 1;
            

            for ( int i=0 ; i<shape.size() ; i++ ) {m_shape[i] = shape[i];}
            for ( int i=1 ; i<shape.size() ; i++ ) {read_shape *= (shape[i]!=0)? shape[i]:1;}
            ori_shape = m_shape;
            
            set_stride();
            
            m_data.resize(m_shape[0]);
            for (int i=0; i<m_shape[0]; ++i) {
                m_data[i].resize(read_shape, value);
            }
            if ( data!=NULL )
                for ( int i=0 ; i<size() ; i++ )
                    m_data[i/m_shape[0]][i%m_shape[0]] = data[i];
        };

        void resize(const std::vector<size_t> &shape, T init_value=0, const T *data=NULL) {
            if ( shape.size()<1 || shape.size()>4 ) {
                throw std::runtime_error("Input should be 1-D, 2-D, 3-D, or 4-D");
            }

            // store 'm_strides' and 'm_shape' always in 3-D,
            // use unit-length for "extra" dimensions (> 'shape.size()')
            while ( m_shape  .size()<4 ) { m_shape  .push_back(0); }
            while ( m_strides.size()<4 ) { m_strides.push_back(0); }
            read_shape = 1;

            for ( int i=0 ; i<shape.size() ; i++ ) {m_shape[i] = shape[i];}
            for ( int i=1 ; i<shape.size() ; i++ ) {read_shape *= (shape[i]!=0)? shape[i]:1;}
            ori_shape = m_shape;
            
            // int _size = set_size();
            set_stride();
            
            m_data.resize(m_shape[0]);
            for (int i=0; i<m_shape[0]; ++i) {
                m_data[i].resize(read_shape, init_value);
            }
        };
        // copy constructor
        // ----------------

        Matrix2D               (const Matrix2D<T> &) = default;
        Matrix2D               (Matrix2D<T> &&) = default;
        Matrix2D<T>& operator= (const Matrix2D<T> &) = default;

        // index operators
        // ---------------

        typename std::vector<T>::reference operator[](size_t i) {
            //std::cout << "row idx: " << i/read_shape[1] << "ele idx: " << i%read_shape[1] << "\n";
            return m_data[i/read_shape][i%read_shape]; };
        const T& operator[](size_t i) const { return m_data[i/read_shape][i%read_shape]; };
    
        T read_data(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix2D*>(this)->read_data(i, j, k, l);
        };
    
        const T read_data(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            size_t idx = i*m_strides[axis_order[0]] + j*m_strides[axis_order[1]] + k*m_strides[axis_order[2]] + l*m_strides[axis_order[3]];
            return m_data[idx/read_shape][idx%read_shape];
            //std::vector<size_t> indices = {i, j, k, l};
            //return m_data[indices[axis_order[0]]][indices[axis_order[1]]*m_strides[1] + indices[axis_order[2]]*m_strides[2] + indices[axis_order[3]]*m_strides[3]];
        };
    
    
        T operator()(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix2D*>(this)->operator()(i, j, k, l);
        };

        const T operator()(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            if (reverse_flag[0] || reverse_flag[1] || reverse_flag[2] || reverse_flag[3]) {
                return reverse_index(i, j, k, l);
            } else {
                return read_data(i, j, k, l);
            }
        };
        
        T reverse_index(size_t i, size_t j=0, size_t k=0, size_t l=0) {
            return const_cast<const Matrix2D*>(this)->reverse_index(i, j, k, l);
        }
    
        const T reverse_index(size_t i, size_t j=0, size_t k=0, size_t l=0) const {
            size_t i_r, j_r, k_r, l_r;
            if (reverse_flag[0] != 0) {
                i_r = m_shape[0]-1-i;
            } else {
                i_r = i;
            }
            if (reverse_flag[1] != 0) {
                j_r = m_shape[1]-1-j;
            } else {
                j_r = j;
            }
            if (reverse_flag[2] != 0) {
                k_r = m_shape[2]-1-k;
            } else {
                k_r = k;
            }
            if (reverse_flag[3] != 0) {
                l_r = m_shape[3]-1-l;
            } else {
                l_r = l;
            }
            return read_data(i_r, j_r, k_r, l_r);
        }

        void set(T value, size_t i, size_t j=0, size_t k=0, size_t l=0) {
            // std::vector<size_t> indices = {i, j, k, l};
            // m_data[indices[axis_order[0]]][indices[axis_order[1]]*m_strides[1] + indices[axis_order[2]]*m_strides[2] + indices[axis_order[3]]*m_strides[3]] = value;
            size_t idx = i*m_strides[axis_order[0]] + j*m_strides[axis_order[1]] + k*m_strides[axis_order[2]] + l*m_strides[axis_order[3]];
            m_data[idx/read_shape][idx%read_shape] = value;
        };
    
        // iterators
        // ---------
        auto begin()       { return m_data.begin(); }
        auto begin() const { return m_data.begin(); }
        auto end()         { return m_data.end();   }
        auto end() const   { return m_data.end();   }

        // return pointer to data
        // ----------------------
        T* data(void) { return m_data.data(); };
        const T* data(void) const { return m_data.data(); };

        // return shape array [ndim]
        // -------------------------
        std::vector<size_t> shape(size_t ndim=0) const {
            if ( ndim == 0 )
                ndim = this->ndim();

            std::vector<size_t> ret(ndim);

            for ( size_t i = 0 ; i < ndim ; ++i )
                ret[i] = m_shape[i];

            return ret;
        };

        // return strides array [ndim]
        // ---------------------------
        std::vector<size_t> strides(bool bytes=false) const {
            size_t ndim = this->ndim();
            std::vector<size_t> ret(ndim);
            
            for ( size_t i = 0 ; i < ndim ; ++i )
                ret[i] = m_strides[i];
            
            if ( bytes )
                for ( size_t i = 0 ; i < ndim ; ++i )
                    ret[i] *= sizeof(T);
            
            return ret;
        };
    
        // return size
        // -----------
        size_t size ( void ) const { return m_data.size()*m_data[0].size(); };

        // return number of dimensions
        // ---------------------------
        size_t ndim ( void ) const {
            size_t i;
            for ( i = 4 ; i > 0 ; i-- )
                if ( m_shape[i-1] != 0 )
                  break;
            return i;    
        };
    
        void transpose(size_t i_t=0, size_t j_t=0, size_t k_t=0, size_t l_t=0) {
            axis_order[0] = i_t;
            axis_order[1] = j_t;
            axis_order[2] = k_t;
            axis_order[3] = l_t;
            
            std::vector<size_t> tmp_shape;
            if (ndim() == 1) {
                tmp_shape = {ori_shape[i_t], 0, 0, 0};
            } else if (ndim() == 2) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], 0, 0};
            } else if (ndim() == 3) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], 0};
            } else if (ndim() == 4) {
                tmp_shape = {ori_shape[i_t], ori_shape[j_t], ori_shape[k_t], ori_shape[l_t]};
            }
            
            m_shape = tmp_shape;
        };
    
        void reshape(size_t i_t, size_t j_t=0, size_t k_t=0, size_t l_t=0) {
            size_t _size = size();
            size_t new_size = i_t;
            if (i_t == 0) {
                throw std::runtime_error("1st value 0 error");
            }
            if (j_t != 0) {
                new_size *= j_t;
            }
            if (k_t != 0) {
                new_size *= k_t;
            }
            if (l_t != 0) {
                new_size *= l_t;
            }
            if (new_size != _size) {
                throw std::runtime_error("Reshape value error");
            }
            m_shape = {i_t, j_t, k_t, l_t};
            ori_shape = m_shape;
            set_stride();
        };
    
        void reverse(bool i_t=false, bool j_t=false, bool k_t=false, bool l_t=false) {
            reverse_flag[0] = i_t;
            reverse_flag[1] = j_t;
            reverse_flag[2] = k_t;
            reverse_flag[3] = l_t;
        };
    

}; // class Matrix 2d
/*
py::class_<Matrix<T>>(m, "Matrix", py::buffer_protocol())
    .def(py::init<py::ssize_t, py::ssize_t>())
    /// Construct from a buffer
    .def(py::init([](const py::buffer &b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<float>::format() || info.ndim != 2)
            throw std::runtime_error("Incompatible buffer format!");

        auto v = new Matrix(info.shape[0], info.shape[1]);
        memcpy(v->data(), info.ptr, sizeof(float) * (size_t) (v->rows() * v->cols()));
        return v;
    }))

    .def("rows", &Matrix::rows)
    .def("cols", &Matrix::cols)

    /// Bare bones interface
    .def("__getitem__",
         [](const Matrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
             if (i.first >= m.rows() || i.second >= m.cols())
                 throw py::index_error();
             return m(i.first, i.second);
         })
    .def("__setitem__",
         [](Matrix &m, std::pair<py::ssize_t, py::ssize_t> i, float v) {
             if (i.first >= m.rows() || i.second >= m.cols())
                 throw py::index_error();
             m(i.first, i.second) = v;
         })
    /// Provide buffer access
    .def_buffer([](Matrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               // Pointer to buffer
            { m.rows(), m.cols() },                 // Buffer dimensions
            { sizeof(float) * size_t(m.cols()),     // Strides (in bytes) for each index
              sizeof(float) }
        );
    });
*/
#endif