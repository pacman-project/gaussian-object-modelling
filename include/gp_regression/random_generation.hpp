#ifndef GP_REGRESSION___RANDOM_GENERATION_HPP
#define GP_REGRESSION___RANDOM_GENERATION_HPP

#include <random>
#include <cmath>

namespace gp_regression
{

///Simple functions to quickly get uniformely distributed numbers in given
//intervals
std::random_device _device_;
std::mt19937_64 _engine_(_device_());

/**
 * @brief Get an uniformely distributed REAL number in [a, b) if inclusive=false,
 * or in [a, b] if inclusive=true.
 */
double getRandIn(const double a, const double b, bool inclusive=false)
{
        if(inclusive){
                std::uniform_real_distribution<double> dis(a,
                std::nextafter(b, std::numeric_limits<double>::max()));
                return (dis(_engine_));
        }
        else{
                std::uniform_real_distribution<double> dis(a,b);
                return (dis(_engine_));
        }
}
/**
 * @brief Get an uniformely distributed INTEGER number in [a, b].
 */
int getRandIn(const int a, const int b)
{
        std::uniform_int_distribution<int> dis(a,b);
        return (dis(_engine_));
}
}

#endif
