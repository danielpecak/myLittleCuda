module vector
  INTERFACE
    subroutine vecadd_wrapper(N,a,b,c) BIND (C, NAME='vecadd_wrapper')
      USE ISO_C_BINDING
      import c_int, c_float
      implicit none
      integer(c_int), value :: N
      real(c_float) :: a(N),b(N),c(N)
    end subroutine vecadd_wrapper
  END INTERFACE
end module vector

program vecAdd
    use ISO_C_BINDING
    use vector
    integer, parameter :: N=100000
    real*8 :: a(N),b(N),c(N)
    real*8 :: sum
    integer :: i

    do i=1,N
      a(i) = sin(i)**2
      b(i) = cos(i)**2
    enddo

    call vecadd_wrapper(N,a,b,c)

    sum=0.d0
    do i=1,N
      sum = sum + c(i)
    enddo

    print *, "Final result: ", sum/N


end program vecAdd
