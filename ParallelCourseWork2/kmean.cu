#include <stdio.h>
#include "check.h"

const int MAX_ITERATIONS = 100; 

struct Point2D
{
  float x;
  float y;
};

int readPoints(const char * fn, int pNr,  Point2D * ps)
{
  FILE * fp = fopen(fn, "rb");    // open a binary input file to read data from  
  if (fp == NULL) return 0;      
  int tmp = fread(ps, sizeof(float), 2*pNr, fp);  // binary read 
  fclose(fp);                      
  return tmp;                     // return the number of successfully read elements 
}

void printClustres(int cNr, int pNr, const Point2D * ctds, Point2D * ps, const int * p2ctds, bool details)
{
  for (int i = 0; i < cNr; i++)
    printf("center %d: %f %f\n", i, ctds[i].x, ctds[i].y);
  
  if (details){
    printf ("\n------------------- details --------------------------\n");
    for (int i = 0; i < cNr; i++)
	{
      printf("center %d: %f %f\n", i, ctds[i].x, ctds[i].y);
      printf( "Points: \n");
      int k = 0; 
      for (int j = 0; j < pNr; j++)
	  {
        if (p2ctds[j] == i)
		{
          switch (k)
		  {
            case 8 : printf ("\n(%.3f,%.3f)\t",  ps[j].x, ps[j].y);
                   break; 
            default: printf ("(%.3f,%.3f)\t",  ps[j].x, ps[j].y);
          }
          k = (k == 8) ? 0: k+1; 
        } 
      }
      printf("\n");
    }  
  }
}

// Initialize cluster centroids to the first K points from the dataset 
void initializeClusters(int n, Point2D const * ps, Point2D * ctds)
{
  for (int i = 0; i < n; i++)
  {
    ctds[i].x = ps[i].x; 
    ctds[i].y = ps[i].y; 
  }
} 

//Kernel that assigns each point to the closest cluster
__global__
void assignClusters(Point2D * ps, int pNr, Point2D *ctds, int cNr, int * p2ctds)
{
	//Thread index and stride size
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	//Go over all points using striding
	while(tid < pNr)
	{	
		float cdis;		//Distance to cluster
		int cIdx;		//Id of the cluster	
		float tmp, tmp_x, tmp_y;	//Temp variables to compute and compare distances

		//Set the first cluster as the starting value of closest cluster
		tmp_x = ps[tid].x - ctds[0].x;	
		tmp_y = ps[tid].y - ctds[0].y;	
		cdis = tmp_x * tmp_x + tmp_y * tmp_y;	
		cIdx = 0;	

		//Foreach cluster
		for (int c = 1; c < cNr; c++)
		{
			//Compute the distance
			tmp_x = ps[tid].x - ctds[c].x;
			tmp_y = ps[tid].y - ctds[c].y;
			tmp = tmp_x*tmp_x + tmp_y*tmp_y; 

			//If the current cluster distance is larger set the current cluster
			//as the closest one
			if ( cdis > tmp)
			{   
				cdis = tmp; 
				cIdx = c;  
			}
		}

		//Set the points to clusters map for the current point
		p2ctds[tid] = cIdx;
	
		//Use stride to handle points that are out of the range of total threads
		tid += stride;
	}
	
}

//Kernel that computes the centroids' position as the average of the points assigned to them
__global__
void centroids(Point2D * ps, int pNr, Point2D * ctds, int cNr, int * p2ctds, int * counters)
{
	//Array in shared memory that contains the number of points of each cluster
	__shared__ int shared_counters[256];

	//Thread index and stride size
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	//Reset the centroids' positions and number of points 
	if(tid < cNr)
	{
		ctds[tid].x = ctds[tid].y = 0.0f;
		counters[tid] = 0;
	}

	//Initialize the shared array in each block
	if(threadIdx.x < cNr)
	{
		shared_counters[threadIdx.x] = 0;
	}

	__syncthreads();

	//Go over all points using striding 
	while(tid < pNr)
	{
		//Use atomicAdd to add the coords of the points to the clusters
		int ctdId = p2ctds[tid];
		atomicAdd(&ctds[ctdId].x, ps[tid].x);
		atomicAdd(&ctds[ctdId].y, ps[tid].y);
		//Use atomicAdd to add the number of clusters to the shared array
		atomicAdd(&shared_counters[ctdId],1);

		//Use stride to handle points that are out of the range of total threads
		tid += stride;
	}

	__syncthreads();

	//Add all the shared arrays to the global array
	if(threadIdx.x < cNr)
	{
		atomicAdd(&counters[threadIdx.x],shared_counters[threadIdx.x]);
	}

}

int main(int argc, char **argv)
{
	//Check if the path of the file is provided
	if (argc < 2)
	{
		printf("Provide the address of input data file.\n");
		return 0; 
	}
	
	int pNr;	//Number of points
	int cNr;	//Number of centroids

	// Extract # of points and # of clusters 
	// from the name of the input file 
	if (sscanf(argv[1], "2DPoints_%d_C_%d.data", &pNr, &cNr) != 2)
	{
		printf("Error! Unexpected file name.\n");
		return 0; 
	}
	
	printf("Number of Points: %d\nNumber of Clusters: %d\n", pNr, cNr); 

	Point2D * ps = new Point2D[pNr];	//Points
	Point2D * ctds = new Point2D[cNr];	//Centroids
	int * p2ctds = new int [pNr];		//Point to cluster map
	int counters[256];					//Number of points assigned to each cluster

	// Read observed points from the input file 
	if (readPoints(argv[1], pNr, ps) == 0)
		printf("Error! Unable to open input file.");

   
	cudaDeviceProp  prop;	 //Device properties
    CHECK( cudaGetDeviceProperties( &prop, 0 ) );	//Get the device properties

	//Device must have compute capability above 2 to use atomicAdd
	if(prop.major < 2)
	{
		printf("GPU's compute capability doesn't support 32-bit floating-point atomicAdd.");
		return 0;	
	}
 
	// Initalize the cluster centroids 
	initializeClusters(cNr, ps, ctds);
	
	Point2D * dev_ps, *dev_ctds;	//Device points, centroids
	int * dev_p2ctds;				//Device point to cluster map
	int * dev_counters;				//Device number of points assigned to each cluster

	int blocks = 32;						//Set the number of blocks to 32
	int threads = prop.maxThreadsPerBlock;	//Set the number of threads to the maximum possible
	
	//Cuda events for timing
	cudaEvent_t start, stop;

	//Allocate memory on device for device variables
	CHECK( cudaMalloc( (void**)&dev_ps, pNr*sizeof(Point2D)));
	CHECK( cudaMalloc( (void**)&dev_ctds, cNr*sizeof(Point2D)));
	CHECK( cudaMalloc( (void**)&dev_p2ctds, pNr*sizeof(int)));
	CHECK( cudaMalloc( (void**)&dev_counters, cNr*sizeof(int)));
	
	//Create events and start timer
	CHECK( cudaEventCreate(&start));
	CHECK( cudaEventCreate(&stop));
	CHECK( cudaEventRecord(start,0));
	
	//Copy points to device
	CHECK( cudaMemcpy( dev_ps, ps, pNr*sizeof(Point2D), cudaMemcpyHostToDevice));

	//For the number of iterations given
	for(int i =0 ; i < MAX_ITERATIONS ; i++)
	{
		//Copy the centroids position to device
		CHECK( cudaMemcpy( dev_ctds, ctds, cNr*sizeof(Point2D), cudaMemcpyHostToDevice));

		//Call the kernel that assigns points to clusters
		assignClusters<<<blocks,threads>>>(dev_ps,pNr,dev_ctds,cNr,dev_p2ctds);

		//Call the kernel that computes the sum of the points position assigned to each cluster
		centroids<<<blocks,threads>>>(dev_ps,pNr,dev_ctds,cNr,dev_p2ctds,dev_counters);
		
		//Copy the number of points assigned to each cluster and the centroids positions to the host
		CHECK( cudaMemcpy( counters, dev_counters, cNr*sizeof(int), cudaMemcpyDeviceToHost));
		CHECK( cudaMemcpy( ctds, dev_ctds, cNr*sizeof(Point2D), cudaMemcpyDeviceToHost));

		//Do the division to find the average
		for(int i =0 ; i < cNr ; i++)
		{
			if(counters[i] > 0)
			{
				ctds[i].x = ctds[i].x/counters[i];
				ctds[i].y = ctds[i].y/counters[i];
			}
			else
			{	
				ctds[i].x = ps[0].x;
				ctds[i].y = ps[0].y;
			}
		}

	}
	
	//Copy the points to clusters map to host
	CHECK( cudaMemcpy(p2ctds, dev_p2ctds, pNr*sizeof(int), cudaMemcpyDeviceToHost));
	
	//Stop the timer
	CHECK( cudaEventRecord(stop, 0));
	CHECK( cudaEventSynchronize(stop));

	float elapsedTime;

	//Get elapsed time from event and print it
	CHECK( cudaEventElapsedTime( &elapsedTime, start, stop));
	printf("Time : %3.1f ms \n", elapsedTime);

	//Destroy the timing events
	CHECK( cudaEventDestroy( start));
	CHECK( cudaEventDestroy( stop));

	//Clear the device resources
	cudaFree(dev_ps);
	cudaFree(dev_ctds);
	cudaFree(dev_p2ctds);
	cudaFree(dev_counters);
	
	//Print the clusters
	printClustres(cNr, pNr, ctds, ps, p2ctds, false);

	return 0;
}
