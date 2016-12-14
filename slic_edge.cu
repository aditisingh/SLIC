#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>


using namespace std;

//handlerror declaration : to display file and line numbers of erroneous lines
static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout<<cudaGetErrorString(err)<<" in "<< file <<" at line "<< line<<endl;
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define min(a, b) ((a) > (b))? (b): (a)
#define max(a, b) ((a) > (b))? (a): (b)

// storing RGB values for rgb colorspace images
struct pixel_RGB
{
	unsigned char r;  //Red values
	unsigned char g;  //Green values
	unsigned char b;  //Blue Values
};

// storing values for xyz and lab colorspace images
struct pixel_XYZ
{
	float x;  //X for XYZ colorspace, L for LAB colorspace
	float y;  //Y for XYZ colorspace, A for LAB colorspace
	float z;  //Z for XYZ colorspace, B for LAB colorspace
};

//store coordinates for each cluster centres
struct point
{ 
	int x;  //x-ccordinate
	int y;  //y-coordinate
};

__global__ void RGB2LAB(pixel_RGB* img, int img_wd, int img_ht, pixel_XYZ* LAB_img)
{
	unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	//row value using x-index of current thread
	unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;	//column value using y-index of current thread
	unsigned int idx=r*img_wd+c;

	if(idx>img_wd*img_ht)
		return;

	int R=img[idx].r;
	int G=img[idx].g;
	int B=img[idx].b;

	double var_R=double(R)/255;
	double var_G=double(G)/255;
	double var_B=double(B)/255;

	double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
	double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
	double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

	X=X/0.95047;
	Y=Y/1.00000;
	Z=Z/1.088969;

	double Y3=pow(Y,1/3);

	double T=0.008856;
	double fx=(X>T)?(powf(X,double(1)/3)):(7.787*X+(16/116));
	double fy=(Y>T)?(powf(Y,double(1)/3)):(7.787*Y+(16/116));
	double fz=(Z>T)?(powf(Z,double(1)/3)):(7.787*Z+(16/116));


	double L=(Y>T)?(116*Y3 - 16):(903.3*Y);
	double a = 500 * (fx - fy);
	double b = 200 * (fy - fz);

	LAB_img[idx].x=L;
	LAB_img[idx].y=a;
	LAB_img[idx].z=b;
}

int min_index(float* array, int size, int x1, int x2, int y1, int y2, int img_wd) //find the index of min value a given region
{
	int index=(x1+1)+(y1+1)*img_wd;;
  	for(int i=0;i<size;i++)
  	{
		if(int(i%img_wd)>=x1 && int(i%img_wd)<=x2 && int(i/img_wd)>=y1 && int(i/img_wd)<=y2)//check if it is in the region of search
		{ 
		  if(array[i]<array[index])
		    index=i;
		}
	}
	return index;
}

__global__ void label_assignment(int* labels_gpu, pixel_XYZ* Pixel_LAB_gpu, point* centers_gpu, int S, int img_wd, int img_ht, int m, float* d_gpu, int k1)
{

	unsigned int index = blockIdx.x*blockDim.x+ threadIdx.x; //find threadindex of cluster center
	if(index>=k1) //for degenerate cases
		return;//}	

	// // finding centre coordinates
	int x_center=centers_gpu[index].x;
	int y_center=centers_gpu[index].y;

	int centre_idx=y_center*img_wd+x_center;//find index in image row major form
	int L_x1=max(0, x_center-S), L_x2=min(x_center+S,img_wd-1),L_y1= max(0,y_center-S),L_y2= min(y_center+S, img_ht-1);

	for(int x_coord=L_x1;x_coord<=L_x2;x_coord++) //look in 2S x 2S neighborhood
	{
		for(int y_coord=L_y1;y_coord<=L_y2;y_coord++)
	 	{

			int j=y_coord*img_wd+x_coord; // find global index of the pixel

			float d_c = sqrt(pow((Pixel_LAB_gpu[centre_idx].x-Pixel_LAB_gpu[j].x),2) + pow((Pixel_LAB_gpu[centre_idx].y-Pixel_LAB_gpu[j].y),2) + pow((Pixel_LAB_gpu[centre_idx].z-Pixel_LAB_gpu[j].z),2)); //color proximity;
			float d_s = sqrtf((x_coord-x_center)*(x_coord-x_center)+(y_coord-y_center)*(y_coord-y_center)); //spatial proximity
			float D=powf(powf(d_c,2)+powf(m*d_s/S,2),0.5);
			
			if(D<d_gpu[j])
			 {
				d_gpu[j]=D;
				labels_gpu[j]=index;
			}

		}
	}
}

__global__ void update_centres(int* labels_gpu, point* centers_gpu, int S, int img_wd, int img_ht, int k1)
{
	size_t index = blockIdx.x*blockDim.x+ threadIdx.x; //thread index
	
	if(index>=k1)
		return;

	// finding centre coordinates
	int centre_x=centers_gpu[index].x;
	int centre_y=centers_gpu[index].y;
	
	//finding the label of cluster, this will be center's label
	int i=labels_gpu[centre_y*img_wd+centre_x]; //finding the label of centre

	int x_mean=0, y_mean=0, count=0, flag=0;
	int L_x1=max(0, centre_x-S), L_x2=min(centre_x+S,img_wd-1),L_y1= max(0,centre_y-S),L_y2= min(centre_y+S, img_ht-1);

	for(int x_coord=L_x1;x_coord<=L_x2;x_coord++) //look in 2S x 2S neighborhood
	{
		for(int y_coord=L_y1;y_coord<=L_y2;y_coord++)
	 	{
			int pt_idx=y_coord*img_wd+x_coord;

			if(labels_gpu[pt_idx]==i)
			{
				x_mean+=x_coord; 
				y_mean+=y_coord;
				flag++;
				count++;
			}
		}
	}
	if(flag)
	{
		centers_gpu[index].x=x_mean/count;
		centers_gpu[index].y=y_mean/count;
	}	

}

float error_calculation(point* centers_curr,point* centers_prev,int N)
{
  float err=0;
  for(int i=0;i<N;i++)
  {
    err+=pow((centers_curr[i].x-centers_prev[i].x),2) + pow((centers_curr[i].y-centers_prev[i].y),2);
  }

  err=((float)err)/N;
  return err;
}


int main(int argc, char* argv[])
{
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	if(argc != 4) //there should be three arguments
	{
		cout<<" program_name image_name num_superpixels control_constant"<<endl;
		return 1; //exit and return an error
	}
	//READING FILE

	ifstream infile;
	infile.open(argv[1]);  //opening the file
	string line;

	int img_wd, img_ht;
	int max_pixel_val;

	HANDLE_ERROR(cudaEventRecord(start));

	//line one contains P6, line 2 mentions about gimp version, line 3 stores the height and width
	getline(infile, line);
	istringstream iss1(line);

	//reading first line to check format
	int word;
	string str1;
	iss1>>str1;

	if(str1.compare("P6")!=0) //comparing magic number
	{
		cout<<"wrong file format"<<endl;
		return 1;
	}


	getline(infile,line); //this line has version related comment, hence ignoring
	getline(infile,line); //this stores image dims

	istringstream iss2(line);
	iss2>>word;// this will be image width
	img_wd=word;
	iss2>>word;// this will be image height
	img_ht=word;

	// cout<<img_ht<<" "<<img_wd<<endl;

	//storing the pixels as 1d images(row major)
	pixel_RGB *Pixel = (pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));

	int pix_cnt=0, cnt=0;

	getline(infile,line); //this stores max value
	istringstream iss3(line);
	iss3>>word;

	max_pixel_val=word;//max pixel value
	// cout<<max_pixel_val<<endl;
	unsigned int val;

	while (getline(infile, line))
	{
		istringstream iss4(line);
		for (int i=0; i<=line.length();i++)
		{
			if(pix_cnt<img_ht*img_wd)
			{
				val =((int)line[i]);
				if(cnt%3==0)
				{
					Pixel[pix_cnt].r=val;
				}
				else if(cnt%3==1)
				{
					Pixel[pix_cnt].g=val;
				}
				else
				{
					Pixel[pix_cnt].b=val;
					pix_cnt++;
				}
				cnt++;
			}
		}
	}

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float milliseconds=0;
	HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
	//get the time in milliseconds


	cout<<"File read in "<<milliseconds<<" ms"<<endl;

	//COLOR CONVERSION
	//RGB->XYZ->CIE-L*ab

	//RGB to XYZ
	// time_t t9= time(NULL);
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));

	float thread_block=sqrt(prop.maxThreadsPerBlock);	//2D blocks used
	dim3 DimGrid(ceil(img_wd/thread_block),ceil(img_ht/thread_block),1); //image saved as a 2D grid
	dim3 DimBlock(thread_block,thread_block,1);

	pixel_RGB* Pixel_gpu; //to copy img to gpu
	HANDLE_ERROR(cudaMalloc(&Pixel_gpu,img_ht*img_wd*sizeof(pixel_RGB)));
	HANDLE_ERROR(cudaMemcpy(Pixel_gpu,Pixel,img_wd*img_ht*sizeof(pixel_RGB),cudaMemcpyHostToDevice));

	pixel_XYZ* Pixel_lab_gpu; 
	pixel_XYZ* Pixel_LAB=(pixel_XYZ*)malloc(img_ht*img_wd*sizeof(pixel_XYZ));

	HANDLE_ERROR(cudaMalloc(&Pixel_lab_gpu,img_ht*img_wd*sizeof(pixel_XYZ)));
	HANDLE_ERROR(cudaMemcpy(Pixel_lab_gpu,Pixel_LAB,img_wd*img_ht*sizeof(pixel_XYZ),cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaEventRecord(start));
	// pixel_XYZ *Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
	RGB2LAB<<<DimGrid,DimBlock>>>(Pixel_gpu, img_wd, img_ht, Pixel_lab_gpu);
	HANDLE_ERROR(cudaMemcpy(Pixel_LAB,Pixel_lab_gpu,img_wd*img_ht*sizeof(pixel_XYZ),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(Pixel_lab_gpu));
	//XYZ TO CIE-L*ab
	// pixel_XYZ* Pixel_LAB1=RGB_LAB(Pixel, img_ht, img_wd);

	// cout<<Pixel_LAB[231012].x<<" "<<Pixel_LAB[231012].y<<" "<<Pixel_LAB[231012].z<<endl;

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));//get the time in milliseconds

	cout<<"Colorspace conversion done in "<<milliseconds<<" ms"<<endl;
	//IMPLEMENTING SLIC ALGORITHM
	int N = img_ht*img_wd;  //number of pixels in the images
	int K = atoi(argv[2]);    //number of superpixels desired

	int S= floor(sqrt(N/K));//size of each superpixel
	int m=atoi(argv[3]);    //compactness control constant
	int k1=ceil(img_ht*1.0/S)*ceil(img_wd*1.0/S);//actual number of superpixels

	cout<<"Image size: "<<img_wd<<" x "<<img_ht<<endl;
	cout<<"Using SLIC algorithm to get "<<k1<<" superpixels of approximate size "<<S<<" x "<<S<<", area "<<S*S<<" each, also m/S="<<1.0*m/S<<endl;
	// cout<<k1<<" "<<S<<" "<<float(img_ht*1.0/S)<<" "<<float(img_wd*1.0/S)<<endl;
	point* centers_curr=(point*)malloc(k1*sizeof(point));

	//initialize centers

	HANDLE_ERROR(cudaEventRecord(start));

	int center_ctr=0;
	for(int j=S/2;j<S*ceil(img_ht*1.0/S);j=j+S)
	{
		for(int i=S/2;i<S*ceil(img_wd*1.0/S);i=i+S)
		{
			int val1=((i>=img_wd)?(img_wd+j-S)/2:i);
			int val2=((j>=img_ht)?(img_ht+i-S)/2:j);
			centers_curr[center_ctr].x=val1;
			centers_curr[center_ctr].y=val2;
			// cout<<center_ctr<<" "<<centers_curr[center_ctr].x<<" "<<centers_curr[center_ctr].y<<" "<<val1<<" "<<val2<<endl;
			center_ctr++;

		}
	}

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));//get the time in milliseconds
	cout<<"centres initialized in "<<milliseconds<<" ms"<<endl;
	//perturb centers

	HANDLE_ERROR(cudaEventRecord(start));
	float* G=(float*)malloc(N*sizeof(float)); 
	for(int i=0; i<img_wd;i++)
	{
	  for(int j=0; j<img_ht;j++)
	  {
	    int index=j*img_wd+i;
	    float L1, L2, L3, L4, a1, a2, a3, a4, b1, b2, b3, b4;
	    L1=L2=L3=L4=a1=a2=a3=a4=b1=b2=b3=b4=0;

		// pt1 is point(x+1, y),pt 2 is point(x-1,y),pt3 is point(x,y+1), pt4 is point(x,y-1)

	    if(i+1<img_wd)
	      L1=Pixel_LAB[j*img_wd+i+1].x, a1=Pixel_LAB[j*img_wd+i+1].y, b1=Pixel_LAB[j*img_wd+i+1].z;
	    if(i-1>0)
	      L2=Pixel_LAB[j*img_wd+i-1].x, a2=Pixel_LAB[j*img_wd+i-1].y, b2=Pixel_LAB[j*img_wd+i-1].z;
	    if(j+1<img_ht)
	      L3=Pixel_LAB[(j+1)*img_wd+i].x, a3=Pixel_LAB[(j+1)*img_wd+i].y, b3=Pixel_LAB[(j+1)*img_wd+i].z;
	    if(j-1>0)
	      L4=Pixel_LAB[(j-1)*img_wd+i].x, a4=Pixel_LAB[(j-1)*img_wd+i].y, b4=Pixel_LAB[(j-1)*img_wd+i].z;

	    G[index]=pow(L1-L2,2) + pow(a1-a2,2) + pow(b1-b2,2) + pow(L3-L4,2) + pow(a3-a4,2) + pow(b3-b4,2);
	  }
	}

	for(int i=0; i<k1;i++)  //for every component
	{
	  int x1=centers_curr[i].x-1;
	  int x2=centers_curr[i].x+1;
	  int y1=centers_curr[i].y-1;
	  int y2=centers_curr[i].y+1;

	  int index = min_index(G, N, x1, x2, y1, y2, img_wd);
	  // cout<<index<<endl;
	  centers_curr[i].x=(floor)(index%img_wd);
	  centers_curr[i].y=(floor)(index/img_wd);

	} 	

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));//get the time in milliseconds
	cout<<"centres perturbed, step 1 in "<<milliseconds<<" ms"<<endl;


	int* labels=(int*)malloc(N*sizeof(int));
	float* d=(float*)malloc(N*sizeof(float));

	for(int idx=0;idx<N;idx++)
	{
	  labels[idx]=-1; 
	  d[idx]=60000;
	}

	float error=100;
	//label assignment

	point* centers_gpu;
	float* d_gpu;
	int* labels_gpu;
	pixel_XYZ* Pixel_LAB_gpu;
	
	point* centers_prev=(point*)malloc(k1*sizeof(point));

	HANDLE_ERROR(cudaMalloc(&centers_gpu, k1*sizeof(point)));

	HANDLE_ERROR(cudaMalloc(&labels_gpu, N*sizeof(int)));

	HANDLE_ERROR(cudaMalloc(&Pixel_LAB_gpu, N*sizeof(pixel_XYZ)));

	HANDLE_ERROR(cudaMalloc(&d_gpu, N*sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(Pixel_LAB_gpu, Pixel_LAB, N*sizeof(pixel_XYZ), cudaMemcpyHostToDevice));
	unsigned int thread_block1=prop.maxThreadsPerBlock;
	// cout<<N<<" "<<S<<" "<<K<<" "<<k1<<" "<<thread_block1<<endl;

	// cout<<DimGrid1.x<<" "<<DimBlock1.x<<endl;

	// float error=100;
	int epoch=0;
	while(error>1)
	{
		cout<<"Epoch = "<<epoch<<endl;
		for(int i=0; i<k1;i++)
		{
	    	centers_prev[i].x=centers_curr[i].x; //saving current centres, before any recalculation
			centers_prev[i].y=centers_curr[i].y;
		}
		HANDLE_ERROR(cudaMemcpy(labels_gpu, labels, N*sizeof(int), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(centers_gpu, centers_curr, k1*sizeof(point), cudaMemcpyHostToDevice));
		// HANDLE_ERROR(cudaMemset(Pixel_LAB_gpu, (112.3,1,1), N*sizeof(pixel_XYZ)));
		HANDLE_ERROR(cudaMemcpy(d_gpu, d , N*sizeof(float), cudaMemcpyHostToDevice));
		// for(int i=0; i<N;i++)
		// cout<<labels[i]<<endl;
		// HANDLE_ERROR(cudaEventRecord(start));

			
		dim3 DimGrid1(1+(k1/thread_block1),1,1); 
		dim3 DimBlock1(thread_block1,1,1);
		cout<<DimGrid1.x<<" "<<DimBlock1.x<<endl;

		label_assignment<<<DimGrid1,DimBlock1>>>(labels_gpu,Pixel_LAB_gpu,centers_gpu,S,img_wd, img_ht,m, d_gpu, k1);
			
		update_centres<<<DimGrid1,DimBlock1>>>(labels_gpu, centers_gpu, S, img_wd, img_ht, k1);

		HANDLE_ERROR(cudaMemcpy(centers_curr, centers_gpu, k1*sizeof(point), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(d, d_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(labels, labels_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
		error= error_calculation(centers_curr, centers_prev,k1);

		cout<<"MSE = "<<error<<endl;
		epoch++;
	}

	// for(int i=0;i<N;i++)
	// 	cout<<d[i]<<" "<<labels[i]<<endl;
	//OUTPUT STORAGE


	pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));

	///enforce connectivity
	//for every point, look into its 4 neighbour labels, if all are same and different from pixel's label, change its label
	for(int x=0; x<img_wd; x++)
	{
	  for(int y=0; y<img_ht; y++)
	  {
	    //int L_0=labels[y*img_wd+x];
	    int L_t=labels[max(y-1,0)*img_wd+x];
	    int L_b=labels[min(y+1,img_ht)*img_wd+x];
	    int L_r=labels[y*img_wd+max(img_wd,x+1)];
	    int L_l=labels[y*img_wd+min(0,x-1)];

	    if(L_t==L_b && L_b==L_r && L_r==L_l)
	    {
	      labels[y*img_wd+x]=L_t;
	// cout<<"stray pixel found"<<endl;
	    }
	  }
	}



	cout<<"connectivity enforced "<<endl;//time_span.count() <<" s"<<endl;

	//randomly shuffle the labels
	random_shuffle(labels,labels+k1);

	float alpha=0;
	for(int i=0;i<img_ht*img_wd;i++)
	{
	  int label_val=labels[i];
	// cout<<label_val<<endl;
	  rgb[i].r=alpha*(21*label_val%255) + (1-alpha)*Pixel[i].r;
	  rgb[i].g=alpha*(47*label_val%255) + (1-alpha)*Pixel[i].g;
	  rgb[i].b=alpha*(173*label_val%255) + (1-alpha)*Pixel[i].b;
	}

	//sobel edge detection
	int valX, valY = 0; 
	int GX [3][3];
	int GY [3][3];

	//Sobel Horizontal Mask     
	GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
	GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
	GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

	//Sobel Vertical Mask   
	GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =   1;    
	GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =   0;    
	GY[2][0] = -1; GY[2][1] =-2; GY[2][2] =  -1;

	double val1;
	for(int i=0;i<img_wd;i++)
	{
		for(int j=0;j<img_ht;j++)
		{
			if(i==0||i==img_wd-1||j==0||j==img_ht-1)
			{
				valX=0;
				valY=0;
			}
			else
			{
				valX=0, valY=0;
				for (int x = -1; x <= 1; x++)
				{
					for (int y = -1; y <= 1; y++)
					{
					  valX = valX + labels[i+x+(j+y)*img_wd] * GX[1+x][1+y];
					  valY = valY + labels[i+x+(j+y)*img_wd]  * GY[1+x][1+y];
					}
				}

			}
			val1=sqrt(valX*valX + valY*valY);
			if(val1>0)
			{
				rgb[j*img_wd+i].r=0;
				rgb[j*img_wd+i].g=0;
				rgb[j*img_wd+i].b=0;
			}
		}
	}
	  


	cout<<"Output image prepared"<<endl;// in "<<time_span.count() <<" s"<<endl;


	// //OUTPUT STORAGE
	ofstream ofs;
	ofs.open("output_gpu.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";

	for(int j=0; j <img_ht*img_wd;j++)
	{
		ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;//labelled_ini[j]<<0<<0;//ofs<<Pixel_LAB[j].x<<Pixel_LAB[j].y<<Pixel_LAB[j].z; //write as ascii
	}

	ofs.close();
		// cudaEventRecord(stop);
		// cudaEventSynchronize(stop);
		// cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds

	cout<<"Image saved "<<endl;//milliseconds<<" ms"<<endl;
	return 0;
}