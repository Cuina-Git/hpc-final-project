static char help[] = "Solves a tridiagonal linear system.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers. 
*/
#include <petscksp.h>
int main(int argc,char **args)
{

   Vec            y, z, T;          /* approx solution, RHS, exact solution */
   Mat            A;                /* linear system matrix */
   KSP            ksp;              /* linear solver context */
   PC             pc;               /* preconditioner context */
   PetscReal      norm=0.0,norm_k=1.0,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
   PetscErrorCode ierr;
   PetscInt       i,n = 10,col[3],its=0,rstart,rend,nlocal,rank,max_it=1000;
   PetscScalar    zero = 0.0,one = 1.0,value[3],lamda,delta_x,delta_t;

   ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
   ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

   /* 创建向量对象 */
   ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
   ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
   ierr = VecSetFromOptions(y);CHKERRQ(ierr);
   ierr = VecDuplicate(y,&z);CHKERRQ(ierr);
   ierr = VecDuplicate(y,&T);CHKERRQ(ierr);

   /* 获得局部划分的上下界 */
   ierr = VecGetOwnershipRange(y,&rstart,&rend);CHKERRQ(ierr);
   ierr = VecGetLocalSize(z,&nlocal);CHKERRQ(ierr);

   /* 创建矩阵对象 */
   ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
   ierr = MatSetFromOptions(A);CHKERRQ(ierr);
   ierr = MatSetUp(A);CHKERRQ(ierr);

   /* 给矩阵元素赋值 */
   if (!rstart) 
   {
      rstart = 1;
      i      = 0; col[0] = 0; col[1] = 1; value[0] = -2.0; value[1] = 1.0;
      ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }
  
   if (rend == n) 
   {
      rend = n-1;
      i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = 1.0; value[1] = -2.0;
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   value[0] = 1.0; value[1] = -2.0; value[2] = 1.0;
   for (i=rstart; i<rend; i++) 
   {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   /* 矩阵集聚 */
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
   /*
      给向量z赋值
   */
   ierr = VecSet(z,zero);CHKERRQ(ierr);
   if (rank == 0){
      i    = 0;
      ierr = VecSetValues(z,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
   }

   /* 向量集聚 */
   ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(z);CHKERRQ(ierr);
  
   ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

   /* 迭代求解过程 */
   while ( PetscAbsReal(norm-norm_k) > tol || its < max_it)
   {
      norm_k = norm;
      ierr = MatMult(A,z,y);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecScale(y,one/norm);CHKERRQ(ierr);
      ierr = VecCopy(y,z);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",PetscAbsReal(norm-norm_k),its);CHKERRQ(ierr);
      its++;
   }
   
   ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

   /* 求解特征值 */
   ierr = MatMultTranspose(A,z,T);CHKERRQ(ierr);
   ierr = VecDot(T,z,&lamda);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"eigenvalue is %g\n",lamda);CHKERRQ(ierr);

  
   ierr = VecDestroy(&z);CHKERRQ(ierr); ierr = VecDestroy(&T);CHKERRQ(ierr); 
   ierr = VecDestroy(&y);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);

   ierr = PetscFinalize();
   return ierr;
}

// EOF