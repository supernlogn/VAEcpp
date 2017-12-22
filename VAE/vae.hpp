#ifndef VAE_HPP
#define VAE_HPP

#include "EncDecder.hpp"
#include <cstddef>
#include <vector>

#define W1 N2 // equals N2 of input image
#define W2 100
#define NUMBER_WEIGHTS (N2 * W2)

template<typename T, std::size_t N1, std::size_t N2>
class Vae
{
  public:
    Vae()
    {
        this->generated_image = new std::vector<T>(N1 * N2);
        this->latent_variables = new std::vector<T>(N1 * W2);
        this->set_encoder(N2, W2);
        this->set_decoder(W2, N2);
    }    
    ~Vae() = default;
    void set_encoder(const int s1, const int s2);
    Encoder<T, NUMBER_WEIGHTS> * get_encoder();
    void set_decoder(const int s1, const int s2);
    Decoder<T, NUMBER_WEIGHTS> * get_decoder();

    T get_loss();
    void update(std::vector<T> & input);
    void encode(std::vector<T> & input);
    std::vector<T> * get_latent_variables();
    void decode();
    std::vector<T> * get_generated_image();
  private:
    std::vector<T> * latent_variables;
    std::vector<T> * generated_image;
    Encoder<T, NUMBER_WEIGHTS> * encoder;
    Decoder<T, NUMBER_WEIGHTS> * decoder;
    T loss;
};


#endif