/*
 * (C) Copyright 1996-2014 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */
#ifndef atlas_grid_reduced_lat_lon_grid_H
#define atlas_grid_reduced_lat_lon_grid_H

#include <cstddef>
#include <vector>

#include "atlas/grid/Grid.h"


//-----------------------------------------------------------------------------

namespace atlas {
namespace grid {

//-----------------------------------------------------------------------------

// ==================================================================================
// gribs use the following convention: (from Shahram)
//
// Horizontally:  Points scan in the +i (+x) direction
// Vertically:    Points scan in the -j (-y) direction
//
// The way I verified this was to look at our SAMPLE files (which IFS uses).
// I also verified that IFS does not modify the scanning modes
// so whatever the samples say, is the convention
// ==================================================================================
// Area: Do we check the area.
// Area: Can we assume area is multiple of the grids ?

class ReducedLatLon : public Grid {

public: // methods

	static std::string className()   { return "atlas.grid.ReducedLatLon"; }
	static std::string gridTypeStr() { return "reduced_ll"; }

	ReducedLatLon( const eckit::Params& p );

	virtual ~ReducedLatLon();

	virtual std::string uid() const;
	virtual std::string hash() const { return hash_;}

	virtual BoundBox boundingBox() const { return bbox_;}
	virtual size_t nPoints() const { return points_.size(); }

	virtual void coordinates( std::vector<double>& ) const;
	virtual void coordinates( std::vector<Point>& ) const;

	virtual std::string gridType() const;
	virtual GridSpec spec() const;
	virtual bool same(const Grid&) const;

protected: // methods

	long rows() const { return nptsNS_;}
	double incLat() const { return nsIncrement_; }

private: // members

	std::string hash_;
	BoundBox bbox_;
	double nsIncrement_;                   ///< In degrees
	long nptsNS_;                          ///< No of points along Y axes
	std::vector<long>    rgSpec_;          ///< No of points per latitude
	std::vector< Point > points_;          ///< storage of coordinate points

};

//-----------------------------------------------------------------------------

} // namespace grid
} // namespace eckit

#endif