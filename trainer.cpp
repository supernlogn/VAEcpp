#include "trainer.hpp"
#include <algorithm>

template <typename T, std::size_t N1, std::size_t N2>
Trainer<T, N1, N2>::Trainer()
{
    this->loss_vector = NULL;
    this->vae = new Vae<T, N1, N2>();
}
template <typename T, std::size_t N1, std::size_t N2>
Trainer<T, N1, N2>::~Trainer()
{
    delete vae;
    if(this->loss_vector != NULL)
    {
        delete this->loss_vector;
    }
}
template <typename T, std::size_t N1, std::size_t N2>
void Trainer<T, N1, N2>::train_vae(const int steps, std::vector<std::vector<uint8_t> > & dataset_input)
{
    std::vector<T> input = std::vector<T>(dataset_input[0].size());
    int i, j;
    input = std::vector<T> (dataset_input[0].size());
    T div = (T) 1 / (T) UINT8_MAX;
    for(i = 0; i < std::min((int32_t) dataset_input.size(), steps); ++i)
    {
        /* handle input */
        if(dataset_input[i].size() == 0)
        {
            continue;
        }
        T mean = 0;
        for(j = 0; j < dataset_input[i].size(); ++j)
        {
            mean += dataset_input[i][j];
            input[j] = dataset_input[i][j];
        }
        for(j = 0; j < input.size(); ++j)
        {
            input[j] -= mean;
            input[j] *=  div;            
        }
        /* update weights */ 
        this->vae->encode(input);
        this->vae->decode();
        this->vae->update(input);
    }
}
template <typename T, std::size_t N1, std::size_t N2>
Vae<T, N1, N2> * Trainer<T, N1, N2>::get_vae()
{
    return this->vae;
}
template <typename T, std::size_t N1, std::size_t N2>
std::vector<T> * Trainer<T, N1, N2>::get_loss_vector()
{
    return this->loss_vector;
}


template class Trainer<double, 28, 28>;