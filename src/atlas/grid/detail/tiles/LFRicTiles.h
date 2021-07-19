/*
 * (C) Crown Copyright 2021, Met Office.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include <array>
#include <string>

#include "atlas/grid/detail/tiles/Tiles.h"
#include "atlas/util/Point.h"

namespace atlas {
namespace cubedspheretiles {

class LFRicCubedSphereTiles : public CubedSphereTiles {
public:

    // constructor
    LFRicCubedSphereTiles( const eckit::Parametrisation& );

    static std::string static_type() { return "cubedsphere_lfric"; }

    virtual std::string type() const override { return static_type(); }

    virtual std::array<std::array<double,6>,2> xy2abOffsets() const override;

    virtual std::array<std::array<double,6>,2> ab2xyOffsets() const override;

    virtual void tile0Rotate( double xyz[] ) const override;

    virtual void tile1Rotate( double xyz[] ) const override;

    virtual void tile2Rotate( double xyz[] ) const override;

    virtual void tile3Rotate( double xyz[] ) const override;

    virtual void tile4Rotate( double xyz[] ) const override;

    virtual void tile5Rotate( double xyz[] ) const override;

    virtual void tile0RotateInverse( double xyz[] ) const override;

    virtual void tile1RotateInverse( double xyz[] ) const override;

    virtual void tile2RotateInverse( double xyz[] ) const override;

    virtual void tile3RotateInverse( double xyz[] ) const override;

    virtual void tile4RotateInverse( double xyz[] ) const override;

    virtual void tile5RotateInverse( double xyz[] ) const override;

    virtual idx_t tileFromXY( const double xy[] ) const override;

    virtual idx_t tileFromLonLat( const double lonlat[] ) const override;

    virtual void enforceXYdomain( double xy[] ) const override;

    virtual atlas::PointXY tileCubePeriodicity (const atlas::PointXY & xyExtended, const atlas::idx_t tile) const override;

    virtual void print( std::ostream& ) const override;

private:
    std::array<atlas::PointXY, 6> botLeftTile_{atlas::PointXY{0., -45.},   atlas::PointXY{90, -45},
                                              atlas::PointXY{180., -45.}, atlas::PointXY{270, -45},
                                              atlas::PointXY{0., 45.},    atlas::PointXY{0, -135.} };

    std::array<atlas::PointXY, 6> botRightTile_{atlas::PointXY{90., -45.},   atlas::PointXY{180., -45},
                                               atlas::PointXY{270., -45.}, atlas::PointXY{360., -45},
                                               atlas::PointXY{90., 45.},    atlas::PointXY{90., -135.} };

    std::array<atlas::PointXY, 6> topLeftTile_{atlas::PointXY{0., 45.},   atlas::PointXY{90, 45},
                                              atlas::PointXY{180., 45.}, atlas::PointXY{270, 45},
                                              atlas::PointXY{0., 135.},    atlas::PointXY{0, -45.} };

    std::array<atlas::PointXY, 6> topRightTile_{atlas::PointXY{90., 45.},   atlas::PointXY{180., 45},
                                               atlas::PointXY{270., 45.}, atlas::PointXY{360., 45},
                                               atlas::PointXY{90., 135.},    atlas::PointXY{90., -45.} };

    bool withinCross(const atlas::idx_t t, const atlas::PointXY & withinRange) const;
    void enforceWrapAround(const atlas::idx_t t, atlas::PointXY & withinRange) const;

};


}  // namespace cubedspheretiles
}  // namespace atlas
