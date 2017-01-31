/*
 * (C) Copyright 1996-2017 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <functional>
#include <algorithm>
#include "eckit/geometry/Point2.h"
#include "atlas/grid/Structured.h"
#include "atlas/grid/partitioners/EqualRegionsPartitioner.h"
#include "atlas/internals/Functions.h"
#include "atlas/parallel/mpi/mpi.h"

using atlas::internals::microdeg;

namespace atlas {
namespace grid {
namespace partitioners {

double gamma(const double& x) {
    double p[14];
    double w, y;
    int k, n;
    p[0 ] =   0.999999999999999990e+00;
    p[1 ] =  -0.422784335098466784e+00;
    p[2 ] =  -0.233093736421782878e+00;
    p[3 ] =   0.191091101387638410e+00;
    p[4 ] =  -0.024552490005641278e+00;
    p[5 ] =  -0.017645244547851414e+00;
    p[6 ] =   0.008023273027855346e+00;
    p[7 ] =  -0.000804329819255744e+00;
    p[8 ] =  -0.000360837876648255e+00;
    p[9 ] =   0.000145596568617526e+00;
    p[10] = -0.000017545539395205e+00;
    p[11] = -0.000002591225267689e+00;
    p[12] =  0.000001337767384067e+00;
    p[13] = -0.000000199542863674e+00;
    n = round(x - 2);
    w = x - (n + 2);
    y = ((((((((((((p[13] * w + p[12]) * w + p[11]) * w + p[10]) *
                 w + p[9]) * w + p[8]) * w + p[7]) * w + p[6]) * w + p[5]) *
            w + p[4]) * w + p[3]) * w + p[2]) * w + p[1]) * w + p[0];
    if (n > 0) {
        w = x - 1;
        for( k = 2; k<= n; ++k )
            w = w * (x - k);
    } else {
        w = 1;
        for( k = 0; k<= -n - 1; ++k)
            y = y * (x + k);
    }
    return w/y;
}


double area_of_cap(const double& s_cap) {
    //
    // AREA_OF_CAP Area of spherical cap
    //
    // AREA_OF_CAP(S_CAP) sets AREA to be the area of an S^2 spherical
    // cap of spherical radius S_CAP.
    //
    return 4.0*M_PI * std::pow( std::sin(0.5*s_cap), 2 );
}

double area_of_collar(const double& a_top, const double& a_bot) {
    // AREA_OF_COLLAR Area of spherical collar
    //
    // AREA_OF_COLLAR(A_TOP, A_BOT) sets AREA to be the area of an S^2 spherical
    // collar specified by A_TOP, A_BOT, where A_TOP is top (smaller) spherical radius,
    // A_BOT is bottom (larger) spherical radius.
    //
    return area_of_cap(a_bot) - area_of_cap(a_top);
}

double sradius_of_cap(const double& area) {
    // SRADIUS_OF_CAP(AREA) returns the spherical radius of
    // an S^2 spherical cap of area AREA.
    //
    return 2.*std::asin(0.5*std::sqrt(area/M_PI));
}

double area_of_ideal_region(int N) {
    //
    // AREA_OF_IDEAL_REGION(N) sets AREA to be the area of one of N equal
    // area regions on S^2, that is 1/N times AREA_OF_SPHERE.
    //
    double area_of_sphere = 2.*std::pow(M_PI,1.5)/gamma(1.5);
    return area_of_sphere/static_cast<double>(N);
}

double polar_colat(int N) {
    //
    // Given N, determine the colatitude of the North polar spherical cap.
    //
    double polar_c(0);
    if( N == 1 ) polar_c = M_PI;
    if( N == 2 ) polar_c = 0.5*M_PI;
    if( N > 2  ) polar_c = sradius_of_cap( area_of_ideal_region(N) );
    return polar_c;
}

double ideal_collar_angle(int N) {
    //
    // IDEAL_COLLAR_ANGLE The ideal angle for spherical collars of an EQ partition
    //
    // IDEAL_COLLAR_ANGLE(N) sets ANGLE to the ideal angle for the
    // spherical collars of an EQ partition of the unit sphere S^2 into N regions.
    //
    return std::sqrt( area_of_ideal_region(N) );
}

void ideal_region_list(int N, const double& c_polar, int n_collars, double r_regions[]) {
    //
    // IDEAL_REGION_LIST The ideal real number of regions in each zone
    //
    //  List the ideal real number of regions in each collar, plus the polar caps.
    //
    //  Given N, c_polar and n_collars, determine r_regions, a list of the ideal real
    //  number of regions in each collar, plus the polar caps.
    //  The number of elements is n_collars+2.
    //  r_regions[1] is 1.
    //  r_regions[n_collars+2] is 1.
    //  The sum of r_regions is N.
    //
    // real(wp),intent(out) :: r_regions(n_collars+2)
    double ideal_region_area,ideal_collar_area;
    r_regions[0] = 1.;
    if( n_collars > 0 ) {
        //
        // Based on n_collars and c_polar, determine a_fitting,
        // the collar angle such that n_collars collars fit between the polar caps.
        //
        double a_fitting = (M_PI-2.*c_polar)/static_cast<double>(n_collars);
        ideal_region_area = area_of_ideal_region(N);
        for(int collar_n=0; collar_n < n_collars; ++collar_n) {
            ideal_collar_area = area_of_collar(c_polar+collar_n*a_fitting, c_polar+(collar_n+1)*a_fitting);
            r_regions[1+collar_n] = ideal_collar_area / ideal_region_area;
        }
    }
    r_regions[2+n_collars-1] = 1.;
}

int num_collars(int N, const double& c_polar, const double& a_ideal) {
    //
    // NUM_COLLARS The number of collars between the polar caps
    //
    //  Given N, an ideal angle, and c_polar,
    //  determine n_collars, the number of collars between the polar caps.
    //
    bool enough = (N > 2) && (a_ideal > 0);
    if( enough )
        return std::max(1, static_cast<int>(round((M_PI-2.*c_polar)/a_ideal)));
    else
        return 0;
}

void round_to_naturals(int N, int ncollars, double r_regions[], int n_regions[]) {
    // ROUND_TO_NATURALS Round off a given list of numbers of regions
    //
    //  Given N and r_regions, determine n_regions,
    //  a list of the natural number of regions in each collar and the polar caps.
    //  This list is as close as possible to r_regions, using rounding.
    //  The number of elements is n_collars+2.
    //  n_regions[1] is 1.
    //  n_regions[n_collars+2] is 1.
    //  The sum of n_regions is N.
    //
    double discrepancy = 0.;
    for( int zone_n=0; zone_n<ncollars+2; ++zone_n ) {
        n_regions[zone_n] = round(r_regions[zone_n]+discrepancy);
        discrepancy += r_regions[zone_n]-n_regions[zone_n];
    }
}

void cap_colats(int N, int n_collars, const double& c_polar, int n_regions[], double c_caps[]) {
    // CAP_COLATS Colatitudes of spherical caps enclosing cumulative sum of regions
    //
    //  Given dim, N, c_polar and n_regions, determine c_caps,
    //  an increasing list of colatitudes of spherical caps which enclose the same area
    //  as that given by the cumulative sum of regions.
    //  The number of elements is n_collars+2.
    //  c_caps[1] is c_polar.
    //  c_caps[n_collars+1] is Pi-c_polar.
    //  c_caps[n_collars+2] is Pi.
    //
    //  c_caps = cap_colats(dim,N,c_polar,n_regions);

    c_caps[0] = c_polar;
    double ideal_region_area = area_of_ideal_region(N);
    int subtotal_n_regions = 1;
    for (int collar_n=0; collar_n<n_collars; ++collar_n) {
        subtotal_n_regions = subtotal_n_regions+n_regions[1+collar_n];
        c_caps[collar_n+1] =sradius_of_cap(subtotal_n_regions*ideal_region_area);
    }
    c_caps[n_collars+1] = M_PI;
}

void eq_caps(int N, std::vector<int>& n_regions, std::vector<double>& s_cap) {
    //
    // eq_regions uses the zonal equal area sphere partitioning algorithm to partition
    // the surface of a sphere into N regions of equal area and small diameter.
    //

    if( N == 1 ) {
        //
        // We have only one region, which must be the whole sphere.
        //
        n_regions.resize(1);
        s_cap.resize(1);
        n_regions[0]=1;
        s_cap[0] = M_PI;
        // int n_regions_ns=1;
    } else {
        //
        // Given N, determine c_polar
        // the colatitude of the North polar spherical cap.
        //
        double c_polar = polar_colat(N);

        //
        // Given N, determine the ideal angle for spherical collars.
        // Based on N, this ideal angle, and c_polar,
        // determine n_collars, the number of collars between the polar caps.
        //
        int n_collars = num_collars(N,c_polar,ideal_collar_angle(N));

        // int n_regions_ns=n_collars+2;
        //
        // Given N, c_polar and n_collars, determine r_regions,
        // a list of the ideal real number of regions in each collar,
        // plus the polar caps.
        // The number of elements is n_collars+2.
        // r_regions[0] is 1.
        // r_regions[2+n_collars-1] is 1.
        // The sum of r_regions is N.
        std::vector<double> r_regions(n_collars+2);
        ideal_region_list(N,c_polar,n_collars,r_regions.data());
        //
        // Given N and r_regions, determine n_regions, a list of the natural number
        // of regions in each collar and the polar caps.
        // This list is as close as possible to r_regions.
        // The number of elements is n_collars+2.
        // n_regions[0] is 1.
        // n_regions[2+n_collars-1] is 1.
        // The sum of n_regions is N.
        //
        n_regions.resize(n_collars+2);
        round_to_naturals(N,n_collars,r_regions.data(),n_regions.data());
        //
        // Given dim, N, c_polar and n_regions, determine s_cap,
        // an increasing list of colatitudes of spherical caps which enclose the same area
        // as that given by the cumulative sum of regions.
        // The number of elements is n_collars+2.
        // s_cap[0] is c_polar.
        // s_cap[n_collars]   is Pi-c_polar.
        // s_cap[n_collars+1] is Pi.
        //
        s_cap.resize(n_collars+2);
        cap_colats(N,n_collars,c_polar,n_regions.data(),s_cap.data());
    }
    // int n_regions_ew=maxval(n_regions(:));
}


void eq_regions(int N, double xmin[], double xmax[], double ymin[], double ymax[]) {
    // EQ_REGIONS Recursive zonal equal area (EQ) partition of sphere
    //
    // Syntax
    //  [regions,dim_1_rot] = eq_regions(dim,N,options);
    //
    // Description
    //  REGIONS = EQ_REGIONS(dim,N) uses the recursive zonal equal area sphere
    //  partitioning algorithm to partition S^dim (the unit sphere in dim+1
    //  dimensional space) into N regions of equal area and small diameter.
    //
    //  The arguments dim and N must be positive integers.
    //
    //  The result REGIONS is a (dim by 2 by N) array, representing the regions
    //  of S^dim. Each element represents a pair of vertex points in spherical polar
    //  coordinates.
    //
    //  Each region is defined as a product of intervals in spherical polar
    //  coordinates. The pair of vertex points regions(:,1,n) and regions(:,2,n) give
    //  the lower and upper limits of each interval.
    //
    //  REGIONS = EQ_REGIONS(dim,N,'offset','extra') uses experimental extra
    //  offsets for S^2 and S^3 to try to minimize energy. If dim > 3, extra offsets
    //  are not used.
    //
    //  REGIONS = EQ_REGIONS(dim,N,extra_offset) uses experimental extra offsets
    //  if extra_offset is true or non-zero.
    //
    //  [REGIONS,DIM_1_ROT] = EQ_REGIONS(dim,N) also returns DIM_1_ROT, a cell
    //  array containing N rotation matrices, one per region, each of size dim by dim.
    //  These describe the R^dim rotation needed to place the region in its final
    //  position.
    //
    //  [REGIONS,DIM_1_ROT] = EQ_REGIONS(dim,N,'offset','extra') partitions S^dim
    //  into N regions, using extra offsets, and also returning DIM_1_ROT, as above.
    //

    if (N == 1) {
        //
        // We have only one region, which must be the whole sphere.
        //
        xmin[0]=0.;
        ymin[0]=-0.5*M_PI;
        xmax[0]=2.*M_PI;
        ymax[0]=0.5*M_PI;
        return;
    }
    //
    // Start the partition of the sphere into N regions by partitioning
    // to caps defined in the current dimension.
    //
    std::vector<int> n_regions;
    std::vector<double> s_cap;
    eq_caps(N,n_regions,s_cap);
    //
    // s_cap is an increasing list of colatitudes of the caps.
    //
    //
    // We have a number of zones: two polar caps and a number of collars.
    // n_regions is the list of the number of regions in each zone.
    //
    int n_collars = n_regions.size()-2;
    //
    // Start with the top cap.
    //
    xmin[0]=0.;
    ymin[0]=0.5*M_PI-s_cap[0];
    xmax[0]=2.*M_PI;
    ymax[0]=0.5*M_PI;

    int region_n = 1;
    for( int collar_n=0; collar_n<n_collars; ++collar_n) {
        for( int region_ew=0; region_ew<n_regions[collar_n+1]; ++region_ew) {
            xmin[region_n]=2.*M_PI/(static_cast<double>(n_regions[collar_n+1]))*region_ew;
            ymin[region_n]=0.5*M_PI-s_cap[collar_n+1];
            xmax[region_n]=2.*M_PI/(static_cast<double>(n_regions[collar_n+1]))*(region_ew+1.);
            ymax[region_n]=0.5*M_PI-s_cap[collar_n];
            ++region_n;
        }
    }
    //
    // End with the bottom cap.
    //
    xmin[N-1]=0.;
    ymin[N-1]=-0.5*M_PI;
    xmax[N-1]=2.*M_PI;
    ymax[N-1]=0.5*M_PI-s_cap[s_cap.size()-2];
}

EqualRegionsPartitioner::EqualRegionsPartitioner(const grid::Grid& grid) :
    Partitioner(grid,parallel::mpi::comm().size()),
    N_(parallel::mpi::comm().size()) {
    std::vector<double> s_cap;
    eq_caps(N_, sectors_, s_cap);
    bands_.resize(s_cap.size());
    for (size_t n = 0; n < s_cap.size(); ++n) {
        bands_[n] = 0.5*M_PI-s_cap[n];
    }
}

EqualRegionsPartitioner::EqualRegionsPartitioner(const grid::Grid& grid, int N) :
    Partitioner(grid,N),
    N_(N) {
    std::vector<double> s_cap;
    eq_caps(N_, sectors_, s_cap);
    bands_.resize(s_cap.size());
    for (size_t n = 0; n < s_cap.size(); ++n) {
        bands_[n] = 0.5*M_PI-s_cap[n];
    }
}

int EqualRegionsPartitioner::partition(const double& x, const double& y) const {
    int b = band(y);
    int p = 0;
    for(int n=0; n<b; ++n)
        p += sectors_[n];
    return p + sector(b,x);
}

int EqualRegionsPartitioner::band(const double& y) const {
    return std::distance( bands_.begin(), std::lower_bound(bands_.begin(),bands_.end(),y, std::greater<double>() ) );
}

int EqualRegionsPartitioner::sector(int band, const double& x) const {
    double xreg = x;
    if(x<0.) xreg += 2.*M_PI;
    else if(x>2.*M_PI) xreg -= 2.*M_PI;
    return std::floor(xreg*sectors_[band]/(2.*M_PI+1e-8));
}

void EqualRegionsPartitioner::where(int partition, int& band, int& sector) const {
    int p=0;
    for(size_t b = 0; b < bands_.size(); ++b) {
        for( int s=0; s<sectors_[b]; ++s ) {
            if (partition == p) {
                band = b;
                sector = s;
                return;
            }
            ++p;
        }
    }
}

bool compare_NS_WE(const EqualRegionsPartitioner::NodeInt& node1, const EqualRegionsPartitioner::NodeInt& node2) {
    if( node1.y >  node2.y ) return true;
    if( node1.y == node2.y ) return (node1.x < node2.x);
    return false;
}

bool compare_WE_NS(const EqualRegionsPartitioner::NodeInt& node1, const EqualRegionsPartitioner::NodeInt& node2) {
    if( node1.x <  node2.x ) return true;
    if( node1.x == node2.x ) return (node1.y > node2.y);
    return false;
}

void EqualRegionsPartitioner::partition(int nb_nodes, NodeInt nodes[], int part[]) const {
    // std::clock_t init, final;
    // init=std::clock();
    // std::cout << "partition start (" << nb_nodes << " points)" << std::endl;
    int nb_parts = N_;
    int p;
    int i;
    int begin;
    int end;
    int band;

    /*
    Sort nodes from north to south, and west to east. Now we can easily split
    the points in bands. Note, for RGG, this should not be necessary, as it is
    already by construction in this order, but then sorting is really fast
    */

    std::sort( nodes, nodes+nb_nodes, compare_NS_WE);

    /*
    For every band, now sort from west to east, and north to south. Inside every band
    we can now easily split nodes in sectors.
    */
    int chunk_size = nb_nodes/nb_parts;
    int chunk_remainder = nb_nodes - chunk_size*nb_parts;
    int remainder = chunk_remainder;
    std::vector<int> count;
    count.reserve(nb_parts);
    end=0;
    for( band=0; band<nb_bands(); ++band ) {
        begin = end;
        for( p=0; p<nb_regions(band); ++p ) {
            count.push_back( chunk_size + (remainder-->0?1:0) );
            end += count.back();
        }
        std::sort( nodes+begin, nodes+end, compare_WE_NS );
    }

    /*
    Create list that tells in original node numbering which part the node belongs to
    */
    remainder = chunk_remainder;
    end = 0;
    for( p=0; p<nb_parts; ++p) {
        begin = end;
        end = begin + count[p];
        for( i=begin; i<end; ++i ) {
            part[nodes[i].n] = p;
        }
    }

    /*
    This piece of code is to reorder back along latitudes
    */

    // begin = 0;
    // for( p=0; p<nb_parts; ++p )
    // {
    //   int end = std::floor((p+1)*chunk_size);
    //   std::sort( nodes+begin, nodes+end, compare_NS_WE );
    //   begin = end;
    // }
    // final=std::clock()-init;
    // std::cout << "partition stop (took " << (double)final / ((double)CLOCKS_PER_SEC) << "s)" << std::endl;
}

void EqualRegionsPartitioner::partition(int part[]) const {
    if( N_ == 1 ) { // trivial solution, so much faster
        for(size_t j = 0; j < grid().npts(); ++j)
            part[j] = 0;
    } else {
        std::vector<NodeInt> nodes(grid().npts());
        int n(0);

        if( const grid::Structured* reduced_grid =
                    dynamic_cast<const grid::Structured*>(&grid()) ) {
            for(size_t jlat = 0; jlat < reduced_grid->nlat(); ++jlat) {
                for(size_t jlon = 0; jlon < reduced_grid->nlon(jlat); ++jlon) {
                    nodes[n].x = microdeg(reduced_grid->lon(jlat,jlon));
                    nodes[n].y = microdeg(reduced_grid->lat(jlat));
                    nodes[n].n = n;
                    ++n;
                }
            }
        } else {
            std::vector<eckit::geometry::LLPoint2> points;
            grid().lonlat(points);
            for(size_t j = 0; j < grid().npts(); ++j) {
                nodes[n].x = microdeg(points[j].lon());
                nodes[n].y = microdeg(points[j].lat());
                nodes[n].n = n;
                ++n;
            }
        }
        partition(grid().npts(), nodes.data(), part);
    }
}

} // namespace partitioners
} // namespace grid
} // namespace atlas

namespace {
atlas::grid::partitioners::PartitionerBuilder<
atlas::grid::partitioners::EqualRegionsPartitioner>
__EqualRegions ("EqualRegions");
}







