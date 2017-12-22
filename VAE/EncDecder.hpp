#ifndef ENCODER_DECODER_HPP
#define ENCODER_DECODER_HPP

#include <vector>
#include <array>

template <typename T, std::size_t N>
class Encoder
{
  public:
    Encoder(std::vector<int> & kernel, std::vector<int> & strides);
    ~Encoder();
    void update(std::vector<T> & output, std::vector<T> & input);
    void get_output(std::vector<T> & output, std::vector<T> & input);
    T get_loss();
  private:
    void init_random(); // initialize weights with Gaussian distribution
    std::array<T, N> * weights;
    std::vector<int> *strides;
    std::vector<int> * dims;
    T learning_rate;
    T loss;
};

template <typename T, std::size_t N>
class Decoder
{
  public:
    Decoder(std::vector<int> & kernel, std::vector<int> & strides);
    ~Decoder();
    void update(std::vector<T> & output, std::vector<T> & input);
    void get_output(std::vector<T> & output, std::vector<T> & input);
    T get_loss();
    std::array<T, N> * weights;
  private:
    void init_random(); // initialize weights with Gaussian distribution    

    std::vector<int> * strides;
    std::vector<int> * dims;
    T learning_rate;
    T loss;
};

#endif