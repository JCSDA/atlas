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

type atlas_FieldSetBundle4D
  integer :: igrid                                  !> Index of the grid
  integer :: nmga                                   !> Number of gridpoints (on a given MPI task)
  integer :: nl0                                    !> Number of levels
  integer :: nv                                     !> Number of variables
  integer :: nts                                    !> Number of timeslots
  real(kind=c_double),allocatable :: lon(:)         !> Longitude (in degrees: -180 to 180)
  real(kind=c_double),allocatable :: lat(:)         !> Latitude (in degrees: -90 to 90)
  real(kind=c_double),allocatable :: area(:)        !> Area (in m^2)
  real(kind=c_double),allocatable :: vunit(:,:)     !> Vertical unit
  logical,allocatable :: lmask(:,:)                 !> Mask

  type(atlas_FieldSetBundle),allocatable :: afsb(:) !> ATLAS field sets
end type atlas_FieldSetBundle4D

private
public :: atlas_FieldSetBundle4D

end module atlas_FieldSetBundle_module
