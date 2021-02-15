/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */


#pragma once

#include "atlas/meshgenerator/MeshGenerator.h"
#include "atlas/meshgenerator/detail/MeshGeneratorImpl.h"
#include "atlas/util/Config.h"
#include "atlas/util/Metadata.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace eckit {
class Parametrisation;
}

namespace atlas {
class StructuredGrid;
class Mesh;
template <typename T>
class vector;
}  // namespace atlas

namespace atlas {
namespace grid {
class Distribution;
}  // namespace grid
}  // namespace atlas
#endif

namespace atlas {
namespace meshgenerator {

//----------------------------------------------------------------------------------------------------------------------

class HealpixMeshGenerator : public MeshGenerator::Implementation {
public:
    HealpixMeshGenerator( const eckit::Parametrisation& = util::NoConfig() );

    virtual void generate( const Grid&, const grid::Distribution&, Mesh& ) const override;
    virtual void generate( const Grid&, Mesh& ) const override;

    using MeshGenerator::Implementation::generate;

    virtual void hash( eckit::Hash& ) const override;
    virtual std::string type() const override { return static_type(); }
    static std::string static_type() { return "healpix"; }

private:
    void configure_defaults();

    void generate_mesh( const StructuredGrid&, const grid::Distribution&, Mesh& m ) const;

private:
    util::Metadata options;
};

//----------------------------------------------------------------------------------------------------------------------

}  // namespace meshgenerator
}  // namespace atlas
