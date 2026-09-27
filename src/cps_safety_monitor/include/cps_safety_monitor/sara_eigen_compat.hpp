#pragma once

#include <Eigen/Core>

// SaRA-Shield currently spells dynamic column vectors as Eigen::Vector<T, N>,
// an alias introduced by newer Eigen releases. ROS 2 Humble ships an older
// Eigen, so provide the identical alias without modifying the SaRA submodule.
#if !EIGEN_VERSION_AT_LEAST(3, 4, 0)
namespace Eigen {
template <typename Scalar, int Size>
using Vector = Matrix<Scalar, Size, 1>;
}  // namespace Eigen
#endif
