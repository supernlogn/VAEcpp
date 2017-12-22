#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "VAE/vae.hpp"
#include <vector>

template <typename T, std::size_t N1, std::size_t N2>
class Trainer
{
    public:
        Trainer();
        ~Trainer();
        void train_vae(const int steps, std::vector<std::vector<uint8_t> > & dataset_input);
        Vae<T, N1, N2> * get_vae();
        std::vector<T> * get_loss_vector();
    private:
        std::vector<T> * loss_vector;
        Vae<T, N1, N2> * vae;
};

#endif