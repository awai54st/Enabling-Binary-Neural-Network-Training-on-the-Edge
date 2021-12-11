#include "optimizers.h"
#include "../utils/initialiser.h"
#include "../naive_layers/vanilla_layers/dense/vanilla_dense.h"
#include "../naive_layers/common_layers/activation_layer/activation_layer.h"

class TestClass : public BaseLayer<Matrix> {
    public:
        Matrix<float> w;
        TestClass(Matrix<float> w): w(w) {};
        ~TestClass() {};
        Matrix<float> &get_weight() {return w;};
        float get_gradient(size_t index) {return w[index];};
};

void test_Adam() {
    Matrix<float> test_input({1}, 10);
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Adam: \n");
    print_mat(test_input);
    
    std::vector<std::unique_ptr<BaseLayer<Matrix>>> layer_seq(1);
    layer_seq[0] = std::make_unique<TestClass>(test_input);
    
    Adam test_Adam;
    for (int _step=0; _step<7; _step++) {
        printf("Curr step: %d ---------------------------------------\n", _step);
        test_Adam.update(layer_seq);

        printf("Adam step counter: %d\n", test_Adam.curr_step);

        printf("Updated weights : \n");
        print_mat(layer_seq[0]->get_weight());

        printf("m adam array: "); 
        print_mat(test_Adam.m_adam_arrs[0]);

        printf("v adam array: "); 
        print_mat(test_Adam.v_adam_arrs[0]);
    }
}

void test_Adam_2() {
    Matrix<float> test_input({2, 4}, 0);
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<4; j++) {
            test_input.set((i+1)*(j+1), i, j);
        }
    }
    
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Adam: \n");
    print_mat(test_input);
    
    std::vector<std::unique_ptr<BaseLayer<Matrix>>> layer_seq(1);
    layer_seq[0] = std::make_unique<TestClass>(test_input);
    
    Adam test_Adam(1e-1);
    for (int _step=0; _step<7; _step++) {
        printf("Curr step: %d ---------------------------------------\n", _step);
        test_Adam.update(layer_seq);

        printf("Adam step counter: %d\n", test_Adam.curr_step);

        printf("Updated weights : \n");
        print_mat(layer_seq[0]->get_weight());

        printf("m adam array: "); 
        print_mat(test_Adam.m_adam_arrs[0]);

        printf("v adam array: "); 
        print_mat(test_Adam.v_adam_arrs[0]);
    }
}

void test_Adam_work() {
    Matrix<float> test_input({4, 5});
    glorot_normal_initializer<float>(test_input);
    printf("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    printf("Test Adam\n");
    print_mat(test_input);
    
    
    std::vector<std::unique_ptr<BaseLayer<Matrix>>> layer_seq(3);
    layer_seq[0] = std::make_unique<Vanilla_BNNDense>(4, false);
    layer_seq[1] = std::make_unique<Vanilla_BNNDense>(4, false);
    layer_seq[2] = std::make_unique<BinaryActivation<float>>();

    Matrix<float> test_output = layer_seq[0]->forward(test_input, true);
    test_output = layer_seq[1]->forward(test_output, true);
    test_output = layer_seq[2]->forward(test_output, true);
    test_output = layer_seq[2]->backprop(test_output);
    test_output = layer_seq[1]->backprop(test_output);
    test_output = layer_seq[0]->backprop(test_output);
    
    
    printf("gradient before: "); print_mat(layer_seq[0]->get_weight());
    Adam test_Adam;
    printf("Adam\n");
    test_Adam.update(layer_seq);
    printf("m_data array: %d\n", test_Adam.m_adam_arrs.size());
    print_mat(test_Adam.m_adam_arrs[0]);
    printf("gradient after: "); print_mat(layer_seq[0]->get_weight());
}

int main(int argc, char * argv[]) {
    const int CHOISE = atoi(argv[1]);
    const int SIZE = atoi(argv[2]);
    
    test_Adam_2();
    
    return 0;
}