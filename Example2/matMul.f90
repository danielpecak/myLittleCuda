module matrix
  INTERFACE
    subroutine matmul_wrapper() BIND (C, NAME="matmul_wrapper")
      USE ISO_C_BINDING
      implicit none
    end subroutine matmul_wrapper
  END INTERFACE
end module matrix

program matMul
  use ISO_C_BINDING
  use matrix

  call matmul_wrapper()
end program matMul
