program matMulFortran
  integer i,j,k
  integer, parameter :: n=1000
  real*8 :: a(n,n), b(n,n), c(n,n)
  do i=1,n
    do j=1,n
      a(i,j) = i*j
      b(i,j) = i-j
    enddo
  enddo
  print *, "Data initialized"
  do i=1,n
    do j=1,n
      c(i,j) = 0.d0
      do k=1,n
        c(i,j)=c(i,j)+a(i,k)*b(k,j)
      enddo
    enddo
  enddo

  ! CHECK FOR 3x3 MATRIX
  ! do i=1,n
  !   print *, c(i,1:3)
  ! enddo

end program matMulFortran
