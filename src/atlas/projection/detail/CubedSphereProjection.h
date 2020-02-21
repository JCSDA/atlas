/*
 * (C) British Crown Copyright 2020 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include "atlas/domain.h"
#include "atlas/projection/detail/ProjectionImpl.h"

namespace atlas {
namespace projection {
namespace detail {

template <typename Rotation>
class CubedSphereProjectionT final : public ProjectionImpl {
public:
    // constructor
    CubedSphereProjectionT( const eckit::Parametrisation& p );
    CubedSphereProjectionT();

    // projection name
    static std::string static_type() { return Rotation::typePrefix() + "cubedsphere"; }
    std::string type() const override { return static_type(); }

    // projection and inverse projection
    void xy2lonlat( double crd[] ) const override;
    void lonlat2xy( double crd[] ) const override;

    bool strictlyRegional() const override { return false; } // cubed sphere is global grid
    RectangularLonLatDomain lonlatBoundingBox( const Domain& domain ) const override {
      return ProjectionImpl::lonlatBoundingBox( domain );
    }

    // specification
    Spec spec() const override;

    std::string units() const override { return "degrees"; }

    void hash( eckit::Hash& ) const override;

private:
    double c_;  // stretching factor
    Rotation rotation_;
};

using CubedSphereProjection = CubedSphereProjectionT<NotRotated>;
using RotatedCubedSphereProjection = CubedSphereProjectionT<Rotated>;

}  // namespace detail
}  // namespace projection
}  // namespace atlas
