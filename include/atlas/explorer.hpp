#ifndef GP_EXPLORER_HPP_
#define GP_EXPLORER_HPP_

#include <memory>
#include <unordered_map>
#include <iostream>
#include <Eigen/Dense>
#include <gp_regression/gp_regression_exception.h>
#include <random_generation.hpp>

namespace gp_atlas_rrt
{

class ExplorerBase
{
    /**
     * \brief get all node ids the given node is connected to
     */
    virtual std::vector<std::size_t> getConnections (const std::size_t &id) const
    {
        auto branch = branches.find(id);
        if (branch != branches.end())
            return branch->second;
        else
            return std::vector<std::size_t>();
    }

    /**
     * \brief Connect two nodes identified by id
     */
    virtual void connect(const std::size_t, const std::size_t)=0;

    ///Connection map
    std::unordered_map<std::size_t, std::vector<std::size_t>> branches;
};
}

#endif
