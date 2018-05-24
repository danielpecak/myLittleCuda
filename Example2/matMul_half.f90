! Initialize in Fortran
! Calculate in GPU

module matrix
  INTERFACE
    subroutine matmul_wrapper(n,a,b,c) BIND (C, NAME="matmul_wrapper")
      USE ISO_C_BINDING
      implicit none
      integer(c_int), value :: n
      real(c_double) :: a(n,n), b(n,n), c(n,n)
    end subroutine matmul_wrapper
  END INTERFACE
end module matrix

program matMul
  use ISO_C_BINDING
  use matrix
  integer, parameter :: n=2**13
  ! integer, parameter :: n=3
  integer :: i,j,k
  real*8 :: a(n,n), b(n,n), c(n,n)
  ! real*8 :: cF(n,n)
  do i=1,n
    do j=1,n
      a(i,j) = i*j
      b(i,j) = i-j
    enddo
  enddo
  print *, "Data initialized"
  call matmul_wrapper(n,a,b,c)
  print *, "Koniec fortrana"
  ! print *, "Sprawdzenie w fortranie"
  ! do i=1,n
  !   do j=1,n
  !     cF(i,j) = 0.d0
  !     do k=1,n
  !       cF(i,j)=cF(i,j)+a(k,i)*b(j,k)
  !     enddo
  !   enddo
  ! enddo
  ! print *, "Sprawdzenie"
  ! do i=1,n
  !   print "(3F8.2,A,3F8.2)", c(:,i), "        ", cF(i,:)
  ! enddo
  ! print *, "========="
  ! do i=1,n
  !   print "(3F8.2,A,3F8.2)", a(:,i), "        ", b(:,i)
  ! enddo
  ! print *, "========="
  ! do i=1,n
  !   do j=1,n
  !     if(dabs(c(i,j)-cF(j,i))>0.1d0) then
  !       print "(2I3,4F8.2)", i, j, a(i,j), b(i,j), c(j,i), cF(j,i)
  !     endif
  !   enddo
  ! enddo
end program matMul
