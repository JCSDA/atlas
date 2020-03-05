/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "atlas/interpolation/method/Method.h"

#include "eckit/linalg/LinearAlgebra.h"
#include "eckit/linalg/Vector.h"
#include "eckit/log/Timer.h"
#include "eckit/thread/AutoLock.h"
#include "eckit/thread/Mutex.h"
#include "eckit/thread/Once.h"

#include "atlas/array.h"
#include "atlas/field/Field.h"
#include "atlas/field/FieldSet.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/functionspace/PointCloud.h"
#include "atlas/interpolation/method/PointIndex3.h"
#include "atlas/mesh/Nodes.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/runtime/Exception.h"
#include "atlas/runtime/Log.h"
#include "atlas/runtime/Trace.h"
#include "atlas/util/LonLatPolygon.h"
#include "atlas/util/Point.h"

namespace atlas {
namespace interpolation {

void Method::check_compatibility( const Field& src, const Field& tgt ) const {
    ATLAS_ASSERT( src.datatype() == tgt.datatype() );
    ATLAS_ASSERT( src.rank() == tgt.rank() );
    ATLAS_ASSERT( src.levels() == tgt.levels() );
    ATLAS_ASSERT( src.variables() == tgt.variables() );

    ATLAS_ASSERT( !matrix_.empty() );
    ATLAS_ASSERT( tgt.shape( 0 ) >= static_cast<idx_t>( matrix_.rows() ) );
    ATLAS_ASSERT( src.shape( 0 ) >= static_cast<idx_t>( matrix_.cols() ) );
}

template <typename Value>
void Method::interpolate_field( const Field& src, Field& tgt ) const {
    check_compatibility( src, tgt );
    if ( src.rank() == 1 ) {
        interpolate_field_rank1<Value>( src, tgt );
    }
    if ( src.rank() == 2 ) {
        interpolate_field_rank2<Value>( src, tgt );
    }
    if ( src.rank() == 3 ) {
        interpolate_field_rank3<Value>( src, tgt );
    }
}

template <typename Value>
void Method::interpolate_field_rank1( const Field& src, Field& tgt ) const {
    const auto outer  = matrix_.outer();
    const auto index  = matrix_.inner();
    const auto weight = matrix_.data();
    idx_t rows        = static_cast<idx_t>( matrix_.rows() );

    if ( use_eckit_linalg_spmv_ ) {
        if ( src.datatype() != array::make_datatype<double>() ) {
            throw_NotImplemented( "Only double precision interpolation is currently implemented with eckit backend",
                                  Here() );
        }
        ATLAS_ASSERT( src.contiguous() );
        ATLAS_ASSERT( tgt.contiguous() );

        eckit::linalg::Vector v_src( const_cast<double*>( array::make_view<double, 1>( src ).data() ), src.shape( 0 ) );
        eckit::linalg::Vector v_tgt( array::make_view<double, 1>( tgt ).data(), tgt.shape( 0 ) );
        eckit::linalg::LinearAlgebra::backend().spmv( matrix_, v_src, v_tgt );
    }
    else {
        auto v_src = array::make_view<Value, 1>( src );
        auto v_tgt = array::make_view<Value, 1>( tgt );

        atlas_omp_parallel_for( idx_t r = 0; r < rows; ++r ) {
            v_tgt( r ) = 0.;
            for ( idx_t c = outer[r]; c < outer[r + 1]; ++c ) {
                idx_t n = index[c];
                Value w = static_cast<Value>( weight[c] );
                v_tgt( r ) += w * v_src( n );
            }
        }
    }
}

template <typename Value>
void Method::interpolate_field_rank2( const Field& src, Field& tgt ) const {
    const auto outer  = matrix_.outer();
    const auto index  = matrix_.inner();
    const auto weight = matrix_.data();
    idx_t rows        = static_cast<idx_t>( matrix_.rows() );

    auto v_src = array::make_view<Value, 2>( src );
    auto v_tgt = array::make_view<Value, 2>( tgt );

    idx_t Nk = src.shape( 1 );

    atlas_omp_parallel_for( idx_t r = 0; r < rows; ++r ) {
        for ( idx_t k = 0; k < Nk; ++k ) {
            v_tgt( r, k ) = 0.;
        }
        for ( idx_t c = outer[r]; c < outer[r + 1]; ++c ) {
            idx_t n = index[c];
            Value w = static_cast<Value>( weight[c] );
            for ( idx_t k = 0; k < Nk; ++k ) {
                v_tgt( r, k ) += w * v_src( n, k );
            }
        }
    }
}


template <typename Value>
void Method::interpolate_field_rank3( const Field& src, Field& tgt ) const {
    const auto outer  = matrix_.outer();
    const auto index  = matrix_.inner();
    const auto weight = matrix_.data();
    idx_t rows        = static_cast<idx_t>( matrix_.rows() );

    auto v_src = array::make_view<Value, 3>( src );
    auto v_tgt = array::make_view<Value, 3>( tgt );

    idx_t Nk = src.shape( 1 );
    idx_t Nl = src.shape( 2 );

    atlas_omp_parallel_for( idx_t r = 0; r < rows; ++r ) {
        for ( idx_t k = 0; k < Nk; ++k ) {
            for ( idx_t l = 0; l < Nl; ++l ) {
                v_tgt( r, k, l ) = 0.;
            }
        }
        for ( idx_t c = outer[r]; c < outer[r + 1]; ++c ) {
            idx_t n = index[c];
            Value w = static_cast<Value>( weight[c] );
            for ( idx_t k = 0; k < Nk; ++k ) {
                for ( idx_t l = 0; l < Nl; ++l ) {
                    v_tgt( r, k, l ) += w * v_src( n, k, l );
                }
            }
        }
    }
}

Method::Method( const Method::Config& config ) {
    std::string spmv = "";
    config.get( "spmv", spmv );
    use_eckit_linalg_spmv_ = ( spmv == "eckit" );
}

void Method::setup( const FunctionSpace& source, const FunctionSpace& target ) {
    ATLAS_TRACE( "atlas::interpolation::method::Method::setup(FunctionSpace, FunctionSpace)" );

    size_t ntasks = atlas::mpi::comm().size();

    // Construct k-d tree with partition centroids
    // FIXME the LonLatPolygon choice should come from the method
    interpolation::method::PointIndex3 search;
    std::vector<util::LonLatPolygon> polygons;

    size_t rank = 0;
    for ( auto& poly : source.polygons() ) {
        ASSERT( !poly->empty() );

        Point3 c{0, 0, 0};
        for ( const Point2& pll : poly->lonlat() ) {
            Point3 p;
            util::Earth::convertSphericalToCartesian( pll, p );
            c = Point3::add( c, p );
        }
        c = Point3::div( c, poly->size() );

        search.insert( {c, rank++} );
        polygons.emplace_back( *poly );
    }

    ASSERT( polygons.size() == size_t( mpi::size() ) );


    // Distribute points to partitions, querying k-d tree for closest partition
    // sorted by distance to centroid
    // FIXME make k configurable
    std::vector< std::vector<double> > sendpoints(ntasks);

    auto lonlat = array::make_view<double, 2>( target.lonlat() );
    size_t k    = 4;  // k-nearest neighbours
    ASSERT( k > 0 );

    for ( int ip = 0; ip < target.size(); ++ip ) {
        Point2 pll{lonlat( ip, LON ), lonlat( ip, LAT )};

        Point3 p;
        util::Earth::convertSphericalToCartesian( pll, p );

        bool found = false;
        for ( auto& n : search.kNearestNeighbours( p, k ) ) {
            auto rank = n.payload();
            ASSERT( rank < size_t( mpi::size() ) );

            if ( ( found = polygons[rank].contains( pll ) ) ) {
                sendpoints[rank].push_back(pll[LON]);
                sendpoints[rank].push_back(pll[LAT]);
                break;
            }
        }

        if ( !found ) {
            std::ostringstream ss;
            ss << "No partition contains " << pll << ", is k=" << k << " too low?";
            throw_AssertionFailed( ss.str(), Here() );
        }
    }

    for ( int rank = 0; rank < mpi::size(); ++rank ) {
        if ( rank == mpi::rank() ) {
            mpi::comm().barrier();
            Log::info() << "--- before distribution"
                           "\nrank="
                        << rank << std::endl;
            for ( int ip = 0; ip < target.size(); ++ip ) {
                Log::info() << Point2{lonlat( ip, LON ), lonlat( ip, LAT )} << std::endl;
            }
            Log::info() << "---" << std::endl;
        }
    }


//  Exchange locations of target points
    std::vector< std::vector<double> > recvpoints(ntasks);
    atlas::mpi::comm().allToAll(sendpoints, recvpoints);

//  Create PointCloud of target points with matching distribution
    std::vector< PointXY > localTargetPoints;
    for (size_t jtask = 0; jtask < ntasks; ++jtask) {
      size_t npts = (recvpoints[jtask].size() + 1) / 2;
      ASSERT( npts * 2 == recvpoints[jtask].size() + 1 );
      for (size_t jpt = 0; jpt < npts; jpt += 2) {
        localTargetPoints.emplace_back(
          PointXY( sendpoints[jtask][jpt+LON], sendpoints[jtask][jpt+LAT] )
        );
      }
    }
    functionspace::PointCloud localtarget(localTargetPoints);

//  Call interpolation setup with local target points
    this->do_setup( source, localtarget );
}

void Method::setup( const Grid& source, const Grid& target ) {
    ATLAS_TRACE( "atlas::interpolation::method::Method::setup(Grid, Grid)" );
    this->do_setup( source, target );
}

void Method::setup( const FunctionSpace& source, const Field& target ) {
    ATLAS_TRACE( "atlas::interpolation::method::Method::setup(FunctionSpace, Field)" );
    this->do_setup( source, target );
}

void Method::setup( const FunctionSpace& source, const FieldSet& target ) {
    ATLAS_TRACE( "atlas::interpolation::method::Method::setup(FunctionSpace, FieldSet)" );
    this->do_setup( source, target );
}

void Method::execute( const FieldSet& source, FieldSet& target ) const {
    ATLAS_TRACE( "atlas::interpolation::method::Method::execute(FieldSet, FieldSet)" );
    this->do_execute( source, target );
}

void Method::execute( const Field& source, Field& target ) const {
    ATLAS_TRACE( "atlas::interpolation::method::Method::execute(Field, Field)" );
    this->do_execute( source, target );
}

void Method::do_setup( const FunctionSpace& /*source*/, const Field& /*target*/ ) {
    ATLAS_NOTIMPLEMENTED;
}

void Method::do_setup( const FunctionSpace& /*source*/, const FieldSet& /*target*/ ) {
    ATLAS_NOTIMPLEMENTED;
}

void Method::do_execute( const FieldSet& fieldsSource, FieldSet& fieldsTarget ) const {
    ATLAS_TRACE( "atlas::interpolation::method::Method::do_execute()" );

    const idx_t N = fieldsSource.size();
    ATLAS_ASSERT( N == fieldsTarget.size() );

    for ( idx_t i = 0; i < fieldsSource.size(); ++i ) {
        Log::debug() << "Method::do_execute() on field " << ( i + 1 ) << '/' << N << "..." << std::endl;
        Method::do_execute( fieldsSource[i], fieldsTarget[i] );
    }
}

void Method::do_execute( const Field& src, Field& tgt ) const {
    haloExchange( src );

    ATLAS_TRACE( "atlas::interpolation::method::Method::do_execute()" );

    if ( src.datatype().kind() == array::DataType::KIND_REAL64 ) {
        interpolate_field<double>( src, tgt );
    }
    if ( src.datatype().kind() == array::DataType::KIND_REAL32 ) {
        interpolate_field<float>( src, tgt );
    }

    tgt.set_dirty();
}

void Method::normalise( Triplets& triplets ) {
    // sum all calculated weights for normalisation
    double sum = 0.0;

    for ( size_t j = 0; j < triplets.size(); ++j ) {
        sum += triplets[j].value();
    }

    // now normalise all weights according to the total
    const double invSum = 1.0 / sum;
    for ( size_t j = 0; j < triplets.size(); ++j ) {
        triplets[j].value() *= invSum;
    }
}

void Method::haloExchange( const FieldSet& fields ) const {
    for ( auto& field : fields ) {
        haloExchange( field );
    }
}
void Method::haloExchange( const Field& field ) const {
    if ( field.dirty() ) {
        source().haloExchange( field );
    }
}

}  // namespace interpolation
}  // namespace atlas
