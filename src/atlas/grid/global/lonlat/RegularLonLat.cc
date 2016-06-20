/*
 * (C) Copyright 1996-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include <typeinfo>
#include "eckit/memory/Builder.h"
#include "atlas/internals/atlas_config.h"
#include "atlas/grid/global/lonlat/RegularLonLat.h"

using eckit::BadParameter;
using eckit::Params;

namespace atlas {
namespace grid {
namespace global {
namespace lonlat {

//------------------------------------------------------------------------------

register_BuilderT1(Grid,RegularLonLat,RegularLonLat::grid_type_str());

std::string RegularLonLat::grid_type_str()
{
  return "regular_lonlat";
}

std::string RegularLonLat::className()
{
  return "atlas.grid.global.lonlat.RegularLonLat";
}

void RegularLonLat::set_typeinfo()
{
  std::stringstream s;
  if( N() ) {
    s << "L" << N();
  } else {
    s << "L" << nlon() << "x" << nlat();
  }
  shortName_ = s.str();
  grid_type_ = grid_type_str();
}

RegularLonLat::RegularLonLat( const eckit::Parametrisation& p )
  : LonLat(Shift::NONE,Domain::makeGlobal())
{
  setup(p);
  set_typeinfo();
}

RegularLonLat::RegularLonLat( const int nlon, const int nlat, const Domain& dom )
 : LonLat(Shift::NONE,dom)
{
  setup( (size_t)nlon, (size_t)nlat );
  set_typeinfo();
}

RegularLonLat::RegularLonLat( const size_t nlon, const size_t nlat, const Domain& dom )
 : LonLat(Shift::NONE,dom)
{
  setup(nlon,nlat);
  set_typeinfo();
}


RegularLonLat::RegularLonLat( const size_t N, const Domain& dom )
 : LonLat(Shift::NONE,dom)
{
  size_t nlon = 4*N;
  size_t nlat = 2*N+1;
  setup(nlon,nlat);
  set_typeinfo();
}

RegularLonLat::RegularLonLat( const double &londeg, const double &latdeg, const Domain& dom )
 : LonLat(Shift::NONE,dom)
{
  setup(londeg,latdeg);
  set_typeinfo();
}


void RegularLonLat::setup(const eckit::Parametrisation& p)
{
  size_t nlon, nlat;

  if( p.get("N",N_ ) )
  {
    nlat = 2*N_+1;
    nlon = 4*N_;
    setup(nlon,nlat);
  }
  else
  {
    if( !p.has("nlon") && !p.has("lon_inc") ) throw BadParameter("nlon or lon_inc missing in Params",Here());
    if( !p.has("nlat") && !p.has("lat_inc") ) throw BadParameter("nlat or lat_inc missing in Params",Here());

    double lon_inc, lat_inc;
    if (p.get("nlon",nlon) && p.get("nlat",nlat))
    {
      setup(nlon,nlat);
    }
    else if (p.get("lon_inc",lon_inc) && p.get("lat_inc",lat_inc))
    {
      setup(lon_inc,lat_inc);
    }
    else
    {
      throw BadParameter("Bad combination of parameters");
    }
  }
}

void RegularLonLat::setup( const size_t nlon, const size_t nlat )
{
    const double latdeg = (domain_.north()-domain_.south())/static_cast<double>(nlat-1);
    const double latmax = domain_.north();
    std::vector<double> lats(nlat);

    std::vector<long>   nlons(nlat,nlon);
    std::vector<double> lonmin(nlat,domain_.west());

    for( size_t jlat=0; jlat<nlat; ++jlat )
    {
        lats[jlat] = latmax - static_cast<double>(jlat)*latdeg;
    }

    if( (nlat-1)%2 == 0 && nlon==2*(nlat-1) )
    {
        Structured::N_ = (nlat-1)/2;
    }
    Structured::setup(nlat,lats.data(),nlons.data(),lonmin.data());
}


void RegularLonLat::setup( const double londeg, const double latdeg )
{
    double nlon_real = (domain_.east() -domain_.west() )/londeg + (domain_.isPeriodicEastWest()? 0:1);
    double nlat_real = (domain_.north()-domain_.south())/latdeg + 1;

    size_t nlon = static_cast<size_t>(nlon_real);
    size_t nlat = static_cast<size_t>(nlat_real);

    std::stringstream msg;
    if( nlon_real - nlon > 0. )
    {
        msg << "Domain range W/E (" << domain_.west() << '/' << domain_.east() << ") is not integer-divisible by londeg " << londeg << '\n';
    }
    if( nlat_real - nlat > 0. )
    {
        msg << "Domain range N/S (" << domain_.north() << '/' << domain_.south() << ") is not integer-divisible by latdeg " << latdeg << '\n';
    }
    if( !msg.str().empty() )
    {
        throw BadParameter(msg.str(),Here());
    }
    setup(nlon,nlat);
}


eckit::Properties RegularLonLat::spec() const
{
  eckit::Properties grid_spec;

  grid_spec.set("grid_type",gridType() );
  grid_spec.set("short_name",shortName());

  grid_spec.set("nlon", nlon() );
  grid_spec.set("nlat", nlat() );


  return grid_spec;
}

//-----------------------------------------------------------------------------

extern "C"
{

Structured* atlas__grid__global__lonlat__RegularLonLat(size_t nlon, size_t nlat)
{
  return new RegularLonLat(nlon,nlat);
}

}

//-----------------------------------------------------------------------------

} // namespace lonlat
} // namespace global
} // namespace grid
} // namespace atlas
