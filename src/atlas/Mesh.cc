/*
 * (C) Copyright 1996-2014 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include <sstream>
#include <stdexcept>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "atlas/Mesh.h"
#include "atlas/FunctionSpace.h"

//------------------------------------------------------------------------------------------------------

namespace atlas {

//------------------------------------------------------------------------------------------------------

Mesh::Ptr Mesh::create()
{
	return Mesh::Ptr( new Mesh( /* eckit::Params ??? */ ) );
}

Mesh::Mesh() :
	grid_(NULL)
{
}

Mesh::~Mesh()
{
}

bool Mesh::has_function_space(const std::string& name) const
{
	return function_spaces_.has(name);
}

FunctionSpace& Mesh::create_function_space(const std::string& name, const std::string& shape_func, const std::vector<int>& shape)
{
	if( has_function_space(name) )
	{
		throw eckit::Exception( "Functionspace '" + name + "' already exists", Here() );
	}

	FunctionSpace::Ptr fs( new FunctionSpace(name,shape_func,shape,*this) );

	function_spaces_.insert(name,fs);
	function_spaces_.sort();

	fs->set_index( function_spaces_.size() - 1 ); ///< @todo revisit this once we can remove functionspaces

	return *fs;
}

FunctionSpace& Mesh::function_space(const std::string& name) const
{
	if( ! has_function_space(name) )
	{
		std::stringstream msg;
		msg << "Could not find FunctionSpace '" << name << "' in mesh";
		throw eckit::OutOfRange(msg.str(),Here());
	}
	return *( function_spaces_[ name ] );
}

FunctionSpace& Mesh::function_space( size_t idx) const
{
	if( idx >= function_spaces_.size() )
		throw eckit::OutOfRange(idx,function_spaces_.size(),Here());
	return *function_spaces_[ idx ];
}


//------------------------------------------------------------------------------------------------------
// C wrapper interfaces to C++ routines

Mesh* atlas__Mesh__new () {
	return new Mesh();
}

void atlas__Mesh__delete (Mesh* This) {
	delete This;
}

void atlas__Mesh__create_function_space(Mesh* This, char* name,char* shape_func,int shape[], int shape_size)
{
	std::vector<int> vshape(shape,shape+shape_size);
	This->create_function_space(std::string(name), std::string(shape_func),vshape);
}

FunctionSpace* atlas__Mesh__function_space (Mesh* This, char* name) {
	return &This->function_space( std::string(name) );
}

Grid* atlas__Mesh__grid (Mesh* This) {
	return &This->grid();
}

//------------------------------------------------------------------------------------------------------

} // namespace atlas
