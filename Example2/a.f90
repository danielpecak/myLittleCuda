program test

  use ISO_C_BINDING

  implicit none

  real(c_float) ::a(4,5)
  integer :: i,j

  interface
    subroutine p(n1,n2,a) bind(C,name="p")
      import c_int, c_float
      integer (c_int), value :: n1, n2
      real(c_float)   :: a(n1,n2)
    end subroutine p
  end interface

  do i=1,4
    do j=1,5
      a(i,j) = i!*1.2 + j*1.1
    enddo
  enddo

  call p(4,5,a)

end program test
