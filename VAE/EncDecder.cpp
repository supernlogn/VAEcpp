#include "EncDecder.hpp"

#include <iterator>
#include <random>
#include <algorithm>


/** ================== Encoder ================== **/
template <typename T, std::size_t N>
Encoder<T, N>::Encoder(std::vector<int> & kernel, std::vector<int> & strides)
{
    this->weights = new std::array<T, N>();
    this->strides = new std::vector<int> (1, 1);
    this->dims = new std::vector<int>(kernel);
    this->learning_rate = (T) 0.00001f;
    init_random();
}
template <typename T, std::size_t N>
Encoder<T, N>::~Encoder()
{
    delete this->weights;
    delete this->strides;
    delete this->dims;
}

template <typename T, std::size_t N>
void Encoder<T, N>::update(std::vector<T> & output, std::vector<T> & input)
{
    /* loss is KL-divergence between a gaussian distribution and the output */
    const int I = std::min(output.size(), input.size());
    T unit_gaussian;
    T delta = 0;
    T KL_div = 0;
    const T m = this->learning_rate;
    std::default_random_engine generator;
    std::normal_distribution<T> n_distribution(0, (T)1);
    /* update weights */
    for(int i = 0; i < I; ++i)
    {
        unit_gaussian = n_distribution(generator);

        KL_div += unit_gaussian * std::log2(output[i]) - unit_gaussian * std::log2(output[i]);

        delta = (unit_gaussian / output[i]) * input[i];
        if(output[i] < 0)
        {
            delta *= 0.001;
        }
        (*(this->weights))[i] += m * delta;
    }

    this->loss = KL_div;
}

template <typename T, std::size_t N>
void Encoder<T, N>::get_output(std::vector<T> & output, std::vector<T> & input)
{
    int i, j, k;
    const int I = input.size() / (*(this->dims))[0];
    const int J = (*(this->dims))[1];
    const int K = (*(this->dims))[0];

    for(i = 0; i < output.size(); ++i)
    {
        output[i] = 0;
    }

    /* matrix multiplication */
    for(i = 0; i < I; ++i)
    {
        for(k = 0; k < K; ++k)
        {
            for(j = 0; j < J; ++j)
            {
                output[i*I+j] += (*(this->weights))[i*K+k] * input[i*k+j];
            }
        }
    }
    /* leaky relu activation */
    for(i = 0; i < I*J; ++i)
    {
        if(output[i] < 0)
        {
            output[i] = 0.001;
        }
    }
}
template <typename T, std::size_t N>
void Encoder<T, N>::init_random()
{
    const int I = this->weights->size();
    std::default_random_engine generator;
    std::normal_distribution<T> n_distribution(0, (T)1);
    for(int i = 0; i < I; ++i)
    {
        (*(this->weights))[i] = n_distribution(generator);
    }
}
template <typename T, std::size_t N>
T Encoder<T, N>::get_loss()
{
    return loss;
}
/** ================== Decoder ================== **/
template <typename T, std::size_t N>
Decoder<T, N>::Decoder(std::vector<int> & kernel, std::vector<int> & strides)
{
    this->weights = new std::array<T, N>();
    this->strides = new std::vector<int> (1, 1);
    this->dims = new std::vector<int>(kernel.begin(), kernel.end());
    this->learning_rate = (T) 0.00001f;
    init_random();
}
template <typename T, std::size_t N>
Decoder<T, N>::~Decoder()
{
    delete this->weights;
    delete this->strides;
    delete this->dims;
}

template <typename T, std::size_t N>
void Decoder<T, N>::update(std::vector<T> & output, std::vector<T> & input)
{
    /* Loss is the mean square between the real image and the generated image */
    const int I = input.size();
    T loss = 0;
    T delta = 0;
    T loss_div = 2.0 / I;
    const T m = this->learning_rate;
    for(int i = 0; i < I; ++i)
    {
        delta = loss_div * (output[i] - input[i]) * input[i];
        loss += loss_div * std::pow(output[i] - input[i], 2);
        if(output[i] < 0)
        {
            delta *= 0.001;
        }
        (*(this->weights))[i] += m * delta;
    }
    this->loss = loss;
}

template <typename T, std::size_t N>
void Decoder<T, N>::get_output(std::vector<T> & output, std::vector<T> & input)
{
    int i, j, k;   
    const int I = input.size() / (*(this->dims))[0];
    const int J = (*(this->dims))[1];
    const int K = (*(this->dims))[0];

    for(i = 0; i < output.size(); ++i)
    {
        output[i] = 0;
    }
    /* matrix multiplication */
    for(i = 0; i < I; ++i)
    {
        for(k = 0; k < K; ++k)
        {
            for(j = 0; j < J; ++j)
            {
                output[i*I+j] += (*(this->weights))[i*K+k] * input[i*k+j];
            }
        }
    }
    /* leaky relu activation */
    for(i = 0; i < I*J; ++i)
    {
        if(output[i] < 0)
        {
            output[i] = 0.001;
        }
    }    
}
template <typename T, std::size_t N>
void Decoder<T, N>::init_random()
{
    const int I = this->weights->size();
    std::default_random_engine generator;
    std::normal_distribution<T> n_distribution(0, (T)1);
    for(int i = 0; i < I; ++i)
    {
        (*(this->weights))[i] = n_distribution(generator);
    }
}
template <typename T, std::size_t N>
T Decoder<T, N>::get_loss()
{
    return loss;
}

template class Encoder<double, 2800>;
template class Decoder<double, 2800>;