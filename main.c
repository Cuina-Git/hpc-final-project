static char help[] = "Solves r the transient heat equation in a one-dimensional \n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers. 
*/
#include <petscksp.h>
int main(int argc,char **args)
{

   Vec            u, u_new, f;      /* approx solution, RHS, exact solution */
   Mat            A;                /* linear system matrix */
   KSP            ksp;              /* linear solver context */
   PC             pc;               /* preconditioner context */
   PetscReal      norm=0.0,norm_k=1.0,time,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
   PetscErrorCode ierr;
   PetscInt       i,m = 100,n = 1000,col[3],rstart,rend,nlocal,rank;
   PetscScalar    zero = 0.0,T = 1.0,value[3],delta_x = 0.01,delta_t = T/n,r=delta_t/(delta_x*delta_x);

   /* Initialize */
   ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
   ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

   /* create vector object */
   ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
   ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
   ierr = VecSetFromOptions(u);CHKERRQ(ierr);
   ierr = VecDuplicate(u,&u_new);CHKERRQ(ierr);
   ierr = VecDuplicate(u,&f);CHKERRQ(ierr);

   /* Identify the starting and ending mesh points on each
      processor for the interior part of the mesh. We let PETSc decide
      above. */
   ierr = VecGetOwnershipRange(y,&rstart,&rend);CHKERRQ(ierr);
   ierr = VecGetLocalSize(z,&nlocal);CHKERRQ(ierr);

   /* Set vector u & f */
   ierr = VecSet(u,zero);CHKERRQ(ierr);
   ierr = VecSet(f,zero);CHKERRQ(ierr);
   if (rank == 0){
      for (i = 0; i < m; i++) {
         if (i == 0 || i == m - 1) {
	         ierr = VecSetValues(u,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
         } 
         else {
            u0   = exp(i*delta_x);
            ierr = VecSetValues(u,1,&i,&u0,INSERT_VALUES);CHKERRQ(ierr);
         }
         f0   = sin(l**i*delta_x);
         ierr = VecSetValues(f,1,&i,&f0,INSERT_VALUES);CHKERRQ(ierr);
      }
   }

   /* Assemble vector */
   ierr = VecAssemblyBegin(z);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(z);CHKERRQ(ierr);
  
   ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

   /* create matrix object */
   ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr = MatSetSizes(A,nlocal,nlocal,m,m);CHKERRQ(ierr);
   ierr = MatSetFromOptions(A);CHKERRQ(ierr);
   ierr = MatSetUp(A);CHKERRQ(ierr);

   /* Set matrix value */
   if (!rstart) 
   {
      rstart = 1;
      i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0-2.0*r; value[1] = r;
      ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }
  
   if (rend == m) 
   {
      rend = m-1;
      i    = m-1; col[0] = m-2; col[1] = n-1; value[0] = r; value[1] = 1.0-2.0*r;
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   value[0] = r; value[1] = 1.0-2.0*r; value[2] = r;
   for (i=rstart; i<rend; i++) 
   {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   /* Assemble matrix */
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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