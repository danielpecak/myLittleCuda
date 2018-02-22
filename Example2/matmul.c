#include <stdio.h>

void p(int n2,int n1,float a[n1][n2]) {
  for(int i=0; i<n1;i++) {
    for(int j=0; j<n2;j++) {
      printf("(%d %d) -> %f\n",i,j,a[i][j]);
    }
  }
}
