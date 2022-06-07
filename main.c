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
   PetscReal      time;             /* norm of solution error */
   PetscErrorCode ierr;
   PetscInt       i,m = 100,n = 1000,l = 1,col[3],rstart,rend,nlocal,rank,its;
   PetscScalar    zero = 0.0,T = 1.0,value[3],ui,fi,delta_x = 0.01,delta_t = T/n,r=delta_t/(delta_x*delta_x);

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
   ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
   ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);

   /* Set vector u & f */
   ierr = VecSet(u,zero);CHKERRQ(ierr);
   ierr = VecSet(f,zero);CHKERRQ(ierr);
   if (rank == 0){
      for (i = 0; i <= m; i++) {
         if (i == 0 || i == m) {
	         ierr = VecSetValues(u,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
         } 
         else {
            ui   = exp(i*delta_x);
            ierr = VecSetValues(u,1,&i,&ui,INSERT_VALUES);CHKERRQ(ierr);
         }
         fi   = sin(l*PETSC_PI*i*delta_x);
      }
   }

   /* Assemble vector */
   ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
   ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  
   ierr = VecSetValues(f,1,&i,&fi,INSERT_VALUES);CHKERRQ(ierr);
   ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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

   /* Slove the heat equation */
    while (its < n){
    ierr = VecScale(f,(PetscScalar)delta_t);CHKERRQ(ierr);
    ierr = MatMultAdd(A,u,f,u_new);CHKERRQ(ierr);
    i = 0;
    ierr = VecSetValues(u_new,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
    i = m-1;
    ierr = VecSetValues(u_new,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
    ierr = VecCopy(u_new,u);CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    its++;
   }

   ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&u_new);CHKERRQ(ierr); 
   ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);

   ierr = PetscFinalize();
   return ierr;
}

// EOF