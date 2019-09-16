! (C) Copyright 2019 UCAR
! 
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0. 

module atlas_FieldSetBundle_module

use atlas_FieldSet_module
use iso_c_binding

implicit none

type atlas_FieldSetBundle
   type(atlas_FieldSet) :: fs
   type(atlas_FieldSet),pointer :: ptr
end type atlas_FieldSetBundle

private
public :: atlas_FieldSetBundle

end module atlas_FieldSetBundle_module
