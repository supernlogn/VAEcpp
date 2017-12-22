#include "vae.hpp"
#include <cstddef>
#include <iostream>

template  <typename T, std::size_t N1, std::size_t N2>
void Vae<T, N1, N2>::set_encoder(const int s1, const int s2)
{
    std::vector<int> kernel({s1, s2});
    std::vector<int> strides(1, 1);
    this->encoder = new Encoder<T, NUMBER_WEIGHTS>(kernel, strides);
}

template  <typename T, std::size_t N1, std::size_t N2>
Encoder<T, NUMBER_WEIGHTS> * Vae<T, N1, N2>::get_encoder()
{
    return this->encoder;
}

template <typename T, std::size_t N1, std::size_t N2>
void Vae<T, N1, N2>::set_decoder(const int s1, const int s2)
{
    std::vector<int> kernel({s1, s2});
    std::vector<int> strides(1, 1);
    this->decoder = new Decoder<T, NUMBER_WEIGHTS>(kernel, strides);
}

template <typename T, std::size_t N1, std::size_t N2>
Decoder<T, NUMBER_WEIGHTS> * Vae<T, N1, N2>::get_decoder()
{
    return this->decoder;
}

template <typename T, std::size_t N1, std::size_t N2>
T Vae<T, N1, N2>::get_loss()
{
    this->loss = this->encoder->get_loss() + this->decoder->get_loss();
    return this->loss;
}

template <typename T, std::size_t N1, std::size_t N2>
void Vae<T, N1, N2>::update(std::vector<T> & input)
{
    this->decoder->update( *(this->generated_image), input);
    this->encoder->update(*(this->latent_variables), input);
}

template <typename T, std::size_t N1, std::size_t N2>
void Vae<T, N1, N2>::encode(std::vector<T> & input)
{
    this->encoder->get_output(*(this->latent_variables), input);
}

template <typename T, std::size_t N1, std::size_t N2>
std::vector<T> * Vae<T, N1, N2>::get_latent_variables()
{
    return this->latent_variables;
}

template <typename T, std::size_t N1, std::size_t N2>
void Vae<T, N1, N2>::decode()
{
    std::cout << (*(this->decoder->weights))[20] << std::endl;
    this->decoder->get_output(*(this->generated_image), *(this->latent_variables));
}

template <typename T, std::size_t N1, std::size_t N2>
std::vector<T> * Vae<T, N1, N2>::get_generated_image()
{
    return this->generated_image;
}

template class Vae<double, 28, 28>;