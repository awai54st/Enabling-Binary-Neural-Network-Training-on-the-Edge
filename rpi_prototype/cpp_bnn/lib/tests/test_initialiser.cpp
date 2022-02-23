#include "../utils/data_type.h"
#include "../utils/initialiser.h"

int main() {
    Matrix<float> test_input({3, 4});
    glorot_normal_initializer(test_input);
    print_mat(test_input);
}