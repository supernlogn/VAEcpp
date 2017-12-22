#include "mnist_reader/mnist_reader.hpp"
#include "trainer.hpp"
#include <iostream>


using namespace std;

#define MNIST_DATA_LOCATION "/home/sniper/projects/AI/VAEcpp/mnist_dataset"

int main(int argc, char * argv[])
{

    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    // std::cout << dataset.test_images[10000].size() << std::endl;
    Trainer<double, 28, 28> * mnist_vae_trainer = new Trainer<double, 28, 28>(); 
    mnist_vae_trainer->train_vae(1000, dataset.training_images);
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    return 0;
}