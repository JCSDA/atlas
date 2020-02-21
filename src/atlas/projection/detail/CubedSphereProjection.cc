/*
 * (C) British Crown Copyright 2020 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <cmath>

#include "eckit/config/Parametrisation.h"
#include "eckit/utils/Hash.h"

#include "atlas/projection/detail/ProjectionFactory.h"
#include "atlas/projection/detail/CubedSphereProjection.h"
#include "atlas/runtime/Exception.h"
#include "atlas/util/Config.h"
#include "atlas/util/Constants.h"

namespace {
static double D2R( const double x ) {
    return atlas::util::Constants::degreesToRadians() * x;
}
static double R2D( const double x ) {
    return atlas::util::Constants::radiansToDegrees() * x;
}
}  // namespace

namespace atlas {
namespace projection {
namespace detail {

// constructor
template <typename Rotation>
CubedSphereProjectionT<Rotation>::CubedSphereProjectionT( const eckit::Parametrisation& params ) :
    ProjectionImpl(),
    rotation_( params ) { }


// constructor
template <typename Rotation>
CubedSphereProjectionT<Rotation>::CubedSphereProjectionT() : ProjectionImpl(), rotation_( util::NoConfig() ) {}

template <typename Rotation>
void CubedSphereProjectionT<Rotation>::xy2lonlat( double crd[] ) const {
    int face_no = (int( crd[0] / M_PI ) + 1)/2;

    crd[0] = crd[0] - 2.0 * M_PI * double(face_no);
    crd[1] = crd[1] - 2.0 * M_PI * double(face_no);

    switch (face_no) {
    case 0:
        crd[0] = crd[0];
        crd[1] = atan( cos(D2R(crd[0])) / tan(D2R(crd[1])) );
        break;
    case 1:
        crd[0] = atan( - 1.0 / tan(D2R(crd[0])) );
        crd[1] = atan( sin(D2R(crd[0])) / tan(crd[1]) );
        break;
    case 2:
        crd[0] = crd[0];
        crd[1] = atan( - cos(D2R(crd[0])) / tan(D2R(crd[1])) );
        break;
    case 3:
        crd[0] = atan( - 1.0 / tan(D2R(crd[0])) );
        crd[1] = atan( - sin(D2R(crd[0]))/tan(crd[1]));
        break;
    case 4:
        break;
    case 5:
        break;

    }
    // convert from colatitude to latitude
    crd[1] = (M_PI / 2.0) -  crd[1];
    // perform rotation
    rotation_.rotate( crd );
}

template <typename Rotation>
void CubedSphereProjectionT<Rotation>::lonlat2xy( double crd[] ) const {

    int face_no;

    // need to calculate some great circles
    if ( crd[0])


    // inverse rotation
    rotation_.unrotate( crd );

    // unstretch
    crd[1] =
        R2D( std::asin( std::cos( 2. * std::atan( c_ * std::tan( std::acos( std::sin( D2R( crd[1] ) ) ) * 0.5 ) ) ) ) );
}

// specification
template <typename Rotation>
typename CubedSphereProjectionT<Rotation>::Spec CubedSphereProjectionT<Rotation>::spec() const {
    Spec proj_spec;
    proj_spec.set( "type", static_type() );
    proj_spec.set( "stretching_factor", c_ );
    rotation_.spec( proj_spec );
    return proj_spec;
}

template <typename Rotation>
void CubedSphereProjectionT<Rotation>::hash( eckit::Hash& hsh ) const {
    hsh.add( static_type() );
    rotation_.hash( hsh );
    hsh.add( c_ );
}

template class CubedSphereProjectionT<NotRotated>;
template class CubedSphereProjectionT<Rotated>;

namespace {
static ProjectionBuilder<CubedSphereProjection> register_1( CubedSphereProjection::static_type() );
static ProjectionBuilder<RotatedCubedSphereProjection> register_2( RotatedCubedSphereProjection::static_type() );
}  // namespace

}  // namespace detail
}  // namespace projection
}  // namespace atlas
