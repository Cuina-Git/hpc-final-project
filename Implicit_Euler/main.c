static char help[] = "Solves r the transient heat equation in a one-dimensional \n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers. 
*/
#include <petscksp.h>
#include <assert.h>

int main(int argc,char **args)
{

   Vec            u, u_new,u_steady, f;      /* approx solution, RHS, exact solution */
   Mat            A;                /* linear system matrix */
   KSP            ksp;              /* linear solver context */
   PC             pc;               /* preconditioner context */
   PetscErrorCode ierr;
   PetscInt       i,m = 101,n = 100000,col[3],rstart,rend,nlocal,rank,its;
   PetscScalar    zero = 0.0,t = 1.0,rho = 1.0,c = 1.0,k = 1.0,l = 1.0,value[3],ui,fi;

   /* Initialize */
   ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
   ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
   PetscReal      norm,time,delta_x = 0.01,delta_t = t/n,r=k*delta_t/(rho*c*delta_x*delta_x);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"delta_t %f \n",delta_t);CHKERRQ(ierr);   

   /* Assert parameters are positive */
   assert(t>0.0);
   assert(c>0.0);
   assert(rho>0.0);
   assert(l>0.0);
   assert(delta_x>0.0);
   assert(delta_t>0.0);

   /* create vector object */
   ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
   ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
   ierr = VecSetFromOptions(u);CHKERRQ(ierr);
   ierr = VecDuplicate(u,&u_new);CHKERRQ(ierr);
   ierr = VecDuplicate(u,&f);CHKERRQ(ierr);
   ierr = VecDuplicate(u,&u_steady);CHKERRQ(ierr);

   /* Identify the starting and ending mesh points on each
      processor for the interior part of the mesh. We let PETSc decide
      above. */
   ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
   ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);

   /* Set vector u & f */
   ierr = VecSet(u,zero);CHKERRQ(ierr);
   ierr = VecSet(u_steady,zero);CHKERRQ(ierr);
   ierr = VecSet(f,zero);CHKERRQ(ierr);
   if (rank == 0){
      for (i = 0; i < m; i++) {
         if (i == 0 || i == m-1) {
            ierr = VecSetValues(u,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
            ierr = VecSetValues(u_steady,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
         } else {
            ui   = exp(i*delta_x);
            ierr = VecSetValues(u,1,&i,&ui,INSERT_VALUES);CHKERRQ(ierr);
            ui   = sin(l*PETSC_PI*i*delta_x)/(l*l*PETSC_PI*PETSC_PI);
            ierr = VecSetValues(u,1,&i,&ui,INSERT_VALUES);CHKERRQ(ierr);
         }
            fi   = sin(l*PETSC_PI*i*delta_x);
            ierr = VecSetValues(f,1,&i,&fi,INSERT_VALUES);CHKERRQ(ierr);
      }
   }

   /* Assemble vector */
   ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
   ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
   ierr = VecAssemblyBegin(u_steady);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(u_steady);CHKERRQ(ierr); 
  
   ierr = VecScale(f,(PetscScalar)delta_t/(rho*c));CHKERRQ(ierr);
   ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

   /* create matrix object */
   ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr = MatSetSizes(A,nlocal,nlocal,m,m);CHKERRQ(ierr);
   ierr = MatSetFromOptions(A);CHKERRQ(ierr);
   ierr = MatSetUp(A);CHKERRQ(ierr);

   /* Set matrix value */
   if (!rstart) 
   {
      rstart = 1;
      i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0+2.0*r; value[1] = -r;
      ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }
  
   if (rend == m) 
   {
      rend = m-1;
      i    = m-1; col[0] = m-2; col[1] = m-1; value[0] = -r; value[1] = 1.0+2.0*r;
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   value[0] = -r; value[1] = 1.0+2.0*r; value[2] = -r;
   for (i=rstart; i<rend; i++) 
   {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
   }

   /* Assemble matrix */
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* Create slover KSP */
   ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

   /* Set operators. */
   ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

   /* Set default of KSP */
   ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
   ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
   ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

   /* Set options of KSP,we can replace it in script */
   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

   /* Slove the heat equation */
    while (its < n){
      ierr = VecAYPX(u,1,f);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,u,u_new);CHKERRQ(ierr);
      i = 0;
      ierr = VecSetValues(u_new,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
      i = m-1;
      ierr = VecSetValues(u_new,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(u_new);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(u_new);CHKERRQ(ierr);
      ierr = VecCopy(u_new,u);CHKERRQ(ierr);
      //ierr = VecView(u_new,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      its++;
   }
   //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = VecAXPY(u_steady,-1,u);CHKERRQ(ierr);
   ierr = VecNorm(u_steady,NORM_2,&norm);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Error is %f \n",norm/n);CHKERRQ(ierr);

   ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&u_new);CHKERRQ(ierr); 
   ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
   ierr = KSPDestroy(&ksp);CHKERRQ(ierr); ierr = VecDestroy(&u_steady);CHKERRQ(ierr); 

   ierr = PetscFinalize();
   return ierr;
}

// EOF
