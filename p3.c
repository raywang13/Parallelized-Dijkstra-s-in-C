/* File:     p3.c
 * Author:   Ray Wang
 * Purpose:  Implement I/O functions that will be useful in an
 *           an MPI implementation of Dijkstra's algorithm.  
 *           In particular, the program creates an MPI_Datatype
 *           that can be used to implement input and output of
 *           a matrix that is distributed by block columns.  It
 *           also implements input and output functions that use
 *           this datatype.  Finally, it implements a function
 *           that prints out a process' submatrix as a string.
 *           This makes it more likely that printing the submatrix 
 *           assigned to one process will be printed without 
 *           interruption by another process.
 *
 * Compile:  mpicc -g -Wall -o p3 p3.c
 * Run:      mpiexec -n <p> ./mpi_io (on lab machines)
 *           csmpiexec -n <p> ./mpi_io (on the penguin cluster)
 *
 * Input:    n:  the number of rows and the number of columns 
 *               in the matrix
 *           mat:  the matrix:  note that INFINITY should be
 *               input as 1000000
 * Output:   The submatrix assigned to each process and the
 *           complete matrix printed from process 0.  Both
 *           print "i" instead of 1000000 for infinity.
 *
 * Notes:
 * 1.  The number of processes, p, should evenly divide n.
 * 2.  Example:  Suppose the matrix is
 *
 *        0 1 2 3
 *        4 0 5 6 
 *        7 8 0 9
 *        8 7 6 0
 *
 *     Then if there are two processes, the matrix will be
 *     distributed as follows:
 *
 *        Proc 0:  0 1    Proc 1:  2 3
 *                 4 0             5 6
 *                 7 8             0 9
 *                 8 7             6 0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_STRING 10000
#define INFINITY 1000000

int Read_n(int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n, 
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
void Print_matrix(int loc_mat[], int n, int loc_n, 
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
int Find_min_dist(int dist[], int known[], int loc_n, int my_rank, 
   MPI_Comm comm);
void Dijkstra(int mat[], int loc_dist[], int loc_pred[], int loc_n, int my_rank,
   int n, MPI_Comm comm);
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, 
   MPI_Comm comm);
void Print_paths(int loc_pred[], int n, int loc_n, int my_rank, 
   MPI_Comm comm);

/* -------------------------------------------------------------------------- */
/* ------------------------------ main -------------------------------------- */

int main(int argc, char* argv[]) {
   int *loc_mat;
   int n, loc_n, p, my_rank;
   int *loc_dist, *loc_pred;
   MPI_Comm comm;
   MPI_Datatype blk_col_mpi_t;

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &my_rank);
   
   n = Read_n(my_rank, comm);
   loc_n = n/p;
   loc_mat = malloc(n*loc_n*sizeof(int));
   loc_dist = malloc(loc_n*sizeof(int));
   loc_pred = malloc(loc_n*sizeof(int));
   
   /* Build the special MPI_Datatype before doing matrix I/O */
   blk_col_mpi_t = Build_blk_col_type(n, loc_n);

   Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
   
   #ifdef DEBUG
      Print_local_matrix(loc_mat, n, loc_n, my_rank);
      Print_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
   #endif

   Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, my_rank, n, comm);
   Print_dists(loc_dist, n, loc_n, my_rank, comm);
   Print_paths(loc_pred, n, loc_n, my_rank, comm);
   
   /* Frees malloc'd space */
   free(loc_mat);
   free(loc_dist);
   free(loc_pred);

   /* Frees the MPI Data Type*/
   MPI_Type_free(&blk_col_mpi_t);

   MPI_Finalize();
   return 0;
}  /* main */


/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */
int Read_n(int my_rank, MPI_Comm comm) {
   int n;

   if (my_rank == 0)
      printf("Please enter the number of vertices in your matrix\n");
      scanf("%d", &n);
   MPI_Bcast(&n, 1, MPI_INT, 0, comm);
   return n;
}  /* Read_n */


/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
   MPI_Aint lb, extent;
   MPI_Datatype block_mpi_t;
   MPI_Datatype first_bc_mpi_t;
   MPI_Datatype blk_col_mpi_t;

   MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
   MPI_Type_get_extent(block_mpi_t, &lb, &extent);

   MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
   MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
         &blk_col_mpi_t);
   MPI_Type_commit(&blk_col_mpi_t);

   MPI_Type_free(&block_mpi_t);
   MPI_Type_free(&first_bc_mpi_t);

   return blk_col_mpi_t;
}  /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be 
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n, 
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) {
      mat = malloc(n*n*sizeof(int));
      for (i = 0; i < n; i++)
         for (j = 0; j < n; j++)
            scanf("%d", &mat[i*n + j]);
   }

   MPI_Scatter(mat, 1, blk_col_mpi_t,
           loc_mat, n*loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) free(mat);
}  /* Read_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_local_matrix
 * Purpose:   Store a process' submatrix as a string and print the
 *            string.  Printing as a string reduces the chance 
 *            that another process' output will interrupt the output.
 *            from the calling process.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  the number of rows in the submatrix
 *            loc_n:  the number of cols in the submatrix
 *            my_rank:  the calling process' rank
 */
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank) {
   char temp[MAX_STRING];
   char *cp = temp;
   int i, j;

   sprintf(cp, "Proc %d >\n", my_rank);
   cp = temp + strlen(temp);
   for (i = 0; i < n; i++) {
      for (j = 0; j < loc_n; j++) {
         if (loc_mat[i*loc_n + j] == INFINITY)
            sprintf(cp, " i ");
         else
            sprintf(cp, "%2d ", loc_mat[i*loc_n + j]);
         cp = temp + strlen(temp);
      }
      sprintf(cp, "\n");
      cp = temp + strlen(temp);
   }

   printf("%s\n", temp);
}  /* Print_local_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the matrix that's been distributed among the 
 *            processes.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  number of rows in the matrix and the submatrices
 *            loc_n:  the number of cols in the submatrix
 *            blk_col_mpi_t:  MPI_Datatype used on process 0 to
 *               receive a process' submatrix
 *            my_rank:  the calling process' rank
 *            comm:  Communicator consisting of all the processes
 */
void Print_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) {
   int* mat = NULL, i, j;

   if (my_rank == 0) mat = malloc(n*n*sizeof(int));
   MPI_Gather(loc_mat, n*loc_n, MPI_INT,
         mat, 1, blk_col_mpi_t, 0, comm);
   if (my_rank == 0) {
      for (i = 0; i < n; i++) {
         for (j = 0; j < n; j++)
            if (mat[i*n + j] == INFINITY)
               printf(" i ");
            else
               printf("%2d ", mat[i*n + j]);
         printf("\n");
      }
      free(mat);
   }
}  /* Print_matrix */

/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     mat: mat[] = submatrix from each process
 *              loc_n = size of loc_dist[] and loc_pred[]
 *              my_rank = rank of process
 *              MPI_Comm = MPI Communicator
 * In/Out args: loc_dist: loc_dist[] = subarray of distances
 *              loc_pred: loc_pred[] = subarray of predecessors
 *         
 */
void Dijkstra(int mat[], int loc_dist[], int loc_pred[], int loc_n, int my_rank,
   int n, MPI_Comm comm) {
   int i, loc_u, u, v, *known, new_dist;

   /* known[v] = true, if the shortest path 0->v is known */
   /* known[v] = false, otherwise                         */
   known = malloc(loc_n*sizeof(int));

   for (v = 0; v < loc_n; v++) {
      loc_dist[v] = mat[0*loc_n + v];
      loc_pred[v] = 0;
      known[v] = 0;
   }

   if (my_rank == 0) {
      known[0] = 1;
   }

   /* On each pass find an additional vertex */
   /* whose distance to 0 is known           */
   for (i = 1; i < n; i++) {

      /* Finds the minimum distance of each dist subarray */
      loc_u = Find_min_dist(loc_dist, known, loc_n, my_rank, comm);

      int my_min[2], glbl_min[2];

      if (loc_u < INFINITY) {
         my_min[0] = loc_dist[loc_u];
         my_min[1] = loc_u + my_rank * loc_n;
      } else {
         my_min[0] = INFINITY;
         my_min[1] = INFINITY;
      }

      /* Finds the minimum distance between each processes' subarray */
      MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);

      /* Stores the global min dist and where it was found */
      int min_dist = glbl_min[0];
      u = glbl_min[1];

      /* Sets known to 1 for appropriate processor */
      if(u/loc_n == my_rank) {

         loc_u = u % loc_n;

         known[loc_u] = 1;
      }

      /* Checks to see if new min is less than existing distance */
      for (v = 0; v < loc_n; v++) {
         if (!known[v]) {
            new_dist = min_dist + mat[u*loc_n + v];
            if (new_dist < loc_dist[v]) {
               loc_dist[v] = new_dist;
               loc_pred[v] = u;
            }
         }
      }
   } /* for i */

   free(known);
}  /* Dijkstra */ 

/*-------------------------------------------------------------------
 * Function:    Find_min_dist
 * Purpose:     Find the vertex u with minimum distance to 0
 *              (loc_dist[u]) among the vertices whose distance 
 *              to 0 is not known.
 * In args:     loc_dist: loc_dist[v] = current estimate of distance
 *                 0->v
 *              loc_known: whether the minimum distance 0-> is
 *                 known
 *              loc_n:  the total number of vertices in each process
 * Ret val:     The vertex loc_u whose distance to 0, loc_dist[u]
 *              is a minimum among vertices whose distance
 *              to 0 is not known.
 */
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int my_rank, 
   MPI_Comm comm) {

   int loc_v;

   int loc_u = INFINITY;
   int loc_min_dist = INFINITY;
   for (loc_v = 0; loc_v < loc_n; loc_v++) {
      if (!loc_known[loc_v]) {
         if (loc_dist[loc_v] < loc_min_dist) {
            loc_u = loc_v;
            loc_min_dist = loc_dist[loc_v];
         }
      }
   }

   return loc_u;
}  /* Find_min_dist */

   /*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, 
   MPI_Comm comm) {
   int v;

   int* dist = NULL;

   if (my_rank == 0) {
      dist = malloc(n*sizeof(int));
   }

   MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) {
      printf("The distance from 0 to each vertex is:\n");
      printf("  v    dist 0->v\n");
      printf("----   ---------\n");
                     
      for (v = 1; v < n; v++)
         printf("%3d       %4d\n", v, dist[v]);
      printf("\n");

      free(dist);
   }
      
} /* Print_dists */  

/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int loc_pred[], int n, int loc_n, int my_rank, 
   MPI_Comm comm) {
   int v, w, *path, count, i;

   int* pred = NULL;

   if (my_rank == 0) {
      pred = malloc(n*sizeof(int));
   }

   MPI_Gather(loc_pred, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);

   if (my_rank == 0) {
      path =  malloc(n*sizeof(int));

      printf("The shortest path from 0 to each vertex is:\n");
      printf("  v     Path 0->v\n");
      printf("----    ---------\n");
      for (v = 1; v < n; v++) {
         printf("%3d:    ", v);
         count = 0;
         w = v;
         while (w != 0) {
            path[count] = w;
            count++;
            w = pred[w];
         }
         printf("0 ");
         for (i = count-1; i >= 0; i--)
            printf("%d ", path[i]);
         printf("\n");
      }

      free(path);
      free(pred);
   }
}  /* Print_paths */