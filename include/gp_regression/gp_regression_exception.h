#ifndef GP_REGRESSION___GP_REGRESSION_EXCEPTION_H
#define GP_REGRESSION___GP_REGRESSION_EXCEPTION_H

#include <exception>

namespace gp_regression
{

class GPRegressionException: public std::exception
{
public:
        GPRegressionException(const std::string& message) : msg(message) {}
        virtual ~GPRegressionException() throw() {}
        virtual const char* what() const throw() {return msg.c_str();}
private:
        std::string msg;
};

}

#endif