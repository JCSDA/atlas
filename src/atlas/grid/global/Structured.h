/*
 * (C) Copyright 1996-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef ReducedGrid_h
#define ReducedGrid_h

#include "eckit/memory/Builder.h"
#include "eckit/config/Parametrisation.h"
#include "atlas/internals/Parameters.h"
#include "atlas/grid/global/Global.h"

namespace atlas {
namespace grid {
namespace global {

//------------------------------------------------------------------------------

/// @brief Reduced Grid
///
/// This class is a base class for all grids that can be described by
/// constant latitudes with a uniform distribution of points per latitude
/// in zonal direction.
/// This means any full grid and reduced grid, both regular, gaussian or other
/// such distribution can be represented with this class

class Structured: public Global {
public:

  typedef eckit::SharedPtr<Structured> Ptr;

  static Structured* create( const eckit::Parametrisation& );
  static Structured* create( const eckit::Properties& );
  static Structured* create( const std::string& shortName );

public:

  static std::string className();
  static std::string grid_type_str() { return "structured"; }

  Structured();

  // Structured( const eckit::Parametrisation& );

  virtual BoundBox boundingBox() const;

  virtual size_t npts() const;

  virtual void lonlat( std::vector<Point>& ) const;

  virtual std::string gridType() const;

  virtual eckit::Properties spec() const = 0;

  /// number of latitudes in hemisphere
  virtual size_t N() const;

  size_t nlat() const;

  size_t nlon( size_t jlat ) const;

  size_t nlonmax() const;

  // Note that this is not the same type as the constructor
  // We return vector<int> for the fortran
  const std::vector<long>& points_per_latitude() const;
  const std::vector<int>& npts_per_lat() const;

  const std::vector<double>& latitudes() const;

  double lon( const size_t jlat, const size_t jlon ) const;

  double lat( const size_t jlat ) const;

  void lonlat( const size_t jlat, const size_t jlon, double crd[] ) const;

private: // methods

  virtual std::string getOptimalMeshGenerator() const;

  virtual size_t copyLonLatMemory(double* pts, size_t size) const;

  virtual void print(std::ostream&) const;

  /// Human readable name
  /// May not be unique, especially when reduced gauss grids have the same N numbers
  /// but different distribution of latitude points
  virtual std::string shortName() const;

protected:

  /// Hash of the PL array
  virtual void hash(eckit::MD5&) const;

  /// @note Domain is already set when calling setup()
  void setup(const eckit::Parametrisation& );
  /// @note Domain is already set when calling setup()
  void setup( const size_t nlat, const double lats[], const long npts_per_lat[] );
  /// @note Domain is already set when calling setup()
  void setup( const size_t nlat, const double lats[], const long nlons[], const double lonmin[], const double lonmax[] );
  /// @note Domain is already set when calling setup()
  void setup_lat_hemisphere( const size_t N, const double lat[], const long lon[] );

protected:

  size_t              N_;
  size_t              nlonmax_;

  size_t              npts_;          ///<! Total number of unique points in the grid

  std::string         grid_type_;

  std::string         shortName_;

  std::vector<double> lat_;    ///<! Latitude values
  std::vector<long>   nlons_;  ///<! Number of points per latitude
  mutable std::vector<int>    nlons_int_;  ///<! Number of points per latitude (int32 type for Fortran interoperability)

  std::vector<double> lonmin_; ///<! Value of minimum longitude per latitude [default=0]
  std::vector<double> lonmax_; ///<! Value of maximum longitude per latitude [default=~360 (one increment smaller)]

  BoundBox            bounding_box_;  ///<! bounding box cache

private:
  std::vector<double> lon_inc_; ///<! Value of longitude increment

};

//------------------------------------------------------------------------------

inline size_t Structured::nlat() const
{
  return lat_.size();
}

inline size_t Structured::nlon(size_t jlat) const
{
  return nlons_[jlat];
}

inline size_t Structured::nlonmax() const
{
  return nlonmax_;
}

inline const std::vector<long>&  Structured::points_per_latitude() const
{
  return nlons_;
}

inline double Structured::lon(const size_t jlat, const size_t jlon) const
{
  return lonmin_[jlat] + (double)jlon * (lonmax_[jlat]-lonmin_[jlat]) / ( (double)nlon(jlat) - 1. );
}

inline double Structured::lat(const size_t jlat) const
{
  return lat_[jlat];
}

inline void Structured::lonlat( const size_t jlat, const size_t jlon, double crd[] ) const
{
  crd[0] = lon(jlat,jlon);
  crd[1] = lat(jlat);
}


//------------------------------------------------------------------------------
extern "C"
{
  void atlas__ReducedGrid__delete(Structured* This);
  Structured* atlas__new_reduced_grid(char* identifier);
  Structured* atlas__ReducedGrid__constructor(int nlat, double lat[], int nlon[]);
  Structured* atlas__new_gaussian_grid(int N);
  Structured* atlas__new_lonlat_grid(int nlon, int nlat);
  Structured* atlas__new_reduced_gaussian_grid(int nlon[], int nlat);
  int    atlas__ReducedGrid__nlat     (Structured* This);
  int    atlas__ReducedGrid__nlon     (Structured* This, int &jlat);
  void   atlas__ReducedGrid__nlon__all(Structured* This, const int* &nlon, int &size);
  int    atlas__ReducedGrid__nlonmax  (Structured* This);
  int    atlas__ReducedGrid__npts     (Structured* This);
  double atlas__ReducedGrid__lat      (Structured* This, int jlat);
  double atlas__ReducedGrid__lon      (Structured* This, int jlat, int jlon);
  void   atlas__ReducedGrid__lonlat   (Structured* This, int jlat, int jlon, double crd[]);
  void   atlas__ReducedGrid__lat__all (Structured* This, const double* &lats, int &size);
}

//------------------------------------------------------------------------------

} // namespace global
} // namespace grid
} // namespace atlas

#endif // ReducedGrid_h
