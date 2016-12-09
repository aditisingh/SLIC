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



	  //color space conversion from RGB to XYZ
	  pixel_XYZ* RGB_XYZ(pixel_RGB* img ,int ht ,int wd)
	  { 
	    pixel_XYZ *XYZ=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ))); //declaring same sized output image

	    for(int i=0; i<ht*wd;i++)
	    {
	    	int R=img[i].r;
	    	int G=img[i].g;
	    	int B=img[i].b;

	    	float var_R=R/255;
	    	float var_G=G/255;
	    	float var_B=B/255;

	    	if(var_R>0.04045)
	    		var_R =pow((var_R + 0.055)/1.055,2.4);
	    	else                  
	    		var_R = var_R / 12.92;

	    	if ( var_G > 0.04045 ) 
	    		var_G = pow((var_G + 0.055)/1.055,2.4);
	    	else                   
	    		var_G = var_G / 12.92;

	    	if ( var_B > 0.04045 ) 
	    		var_B = pow((var_B + 0.055 )/1.055, 2.4);
	    	else                   
	    		var_B = var_B / 12.92;

	    	var_R = var_R * 100;
	    	var_G = var_G * 100;
	    	var_B = var_B * 100;

	    	float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
	    	float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
	    	float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

	    	XYZ[i].x=X;
	    	XYZ[i].y=Y;
	    	XYZ[i].z=Z;

	    }

	    return XYZ;
	  }
	  //colorspace conversion from XYZ to LAB
	  pixel_XYZ* XYZ_LAB(pixel_XYZ* img ,int ht ,int wd)
	  { 
	  	pixel_XYZ *LAB_img=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ)));

	  	for(int i=0; i<ht*wd;i++)
	  	{
	  		float X=img[i].x;
	  		float Y=img[i].y;
	  		float Z=img[i].z;

	  		float ref_X =  95.047;
	  		float ref_Y = 100.000;
	  		float ref_Z = 108.883;

	      float var_X = X/ref_X;          //  Observer= 2°, Illuminant= D65
	      float var_Y = Y/ref_Y;
	      float var_Z = Z/ref_Z;          

	      if(var_X > 0.008856) 
	      	var_X = pow(var_X,1/3);
	      else                    
	      	var_X = (7.787*var_X ) + (16/116);

	      if(var_Y > 0.008856) 
	      	var_Y = pow(var_Y,1/3);
	      else                    
	      	var_Y = (7.787*var_Y) + (16/116);

	      if(var_Z > 0.008856) 
	      	var_Z = pow(var_Z,1/3);
	      else                    
	      	var_Z = (7.787*var_Z) + (16/116);

	      float L = (116 * var_Y) - 16;
	      float A = 500 * (var_X - var_Y);
	      float B = 200 * (var_Y - var_Z);

	      LAB_img[i].x=L;
	      LAB_img[i].y=A;
	      LAB_img[i].z=B;
	    }
	    return LAB_img;
	  }

	  int min_index(float* array, int size, int x1, int x2, int y1, int y2, int img_wd) //find the index of min value a given region
	  {
	  	int index=0;
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


	  __global__ void label_assignment(int* labels_gpu, pixel_XYZ* Pixel_LAB_gpu, point* centers_gpu, int S, int img_wd,int m, float* d_gpu, int k1)
	  {
	  size_t index = blockIdx.x*blockDim.x+ threadIdx.x; //find threadindex of cluster center
		// finding centre coordinates
	  int x_center=centers_gpu[index].x;
	  int y_center=centers_gpu[index].y;
	  int centre_idx=y_center*img_wd+x_center;//find index in image row major form

	  if(index>=k1) //for degenerate cases
	  	return;	

		for(int x_coord=x_center-S;x_coord<=x_center+S;x_coord++) //look in 2S x 2S neighborhood
		{
			for(int y_coord=y_center-S;y_coord<=y_center+S;y_coord++)
			{
				int j=y_coord*img_wd+x_coord; // find global index of the pixel
			  float d_c = powf(powf((Pixel_LAB_gpu[centre_idx].x-Pixel_LAB_gpu[j].x),2) + powf((Pixel_LAB_gpu[centre_idx].y-Pixel_LAB_gpu[j].y),2) + powf((Pixel_LAB_gpu[centre_idx].z-Pixel_LAB_gpu[j].z),2),0.5); //color proximity;
	   		float d_s = powf(powf(x_coord-x_center,2)+powf(y_coord-y_center,2),0.5); //spatial proximity
	   		float D=powf(powf(d_c,2)+powf(m*d_s/S,2),0.5);

	   		if(D<d_gpu[j])
	   		{
	   			d_gpu[j]=D;
	   			labels_gpu[j]=index;
	   		}
	   	}
	   }
	 }

	 __global__ void update_centres(int* labels_gpu, point* centers_gpu, int S, int img_wd, int k1)
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
	  for(int x_coord=centre_x-S;x_coord<=centre_x+S;x_coord++)
	  {
	  	for(int y_coord=centre_y-S;y_coord<=centre_y+S;y_coord++)
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
	  // printf("index: %d, initial values : %d %d , new values : %d %d \n",index, centre_x,centre_y,centers_gpu[index].x,centers_gpu[index].y);

	}

	__host__ __device__ float padding(float* Pixel_val, int x_coord, int y_coord, int img_width, int img_height) 
	{ float Px;
		Px=0;
		if(x_coord< img_width && y_coord <img_height && x_coord>=0 && y_coord>=0)
		{
			Px=Pixel_val[y_coord*img_width+x_coord];
		}
		return Px;
	}

	__global__ void vertical_conv(float* Pixel_in, float* Pixel_out,int img_wd, int img_ht, float* kernel, int k)
	{
		size_t col=blockIdx.x*blockDim.x + threadIdx.x;
		size_t row=blockIdx.y*blockDim.y + threadIdx.y;

		size_t idx=row*img_wd+col;

		float tmp=0;    

		if(row<img_ht && col<img_wd){

			for(int l=0;l<k;l++)
			{
				float val=padding(Pixel_in, col, (row+l-(k-1)/2), img_wd, img_ht);
				tmp+=(val) * kernel[l]/3;
			}

			Pixel_out[idx]=tmp;
		}
	}     

	__global__ void horizontal_conv(float* Pixel_in, float* Pixel_out, int img_wd, int img_ht, float* kernel, int k)
	{
		size_t col=blockIdx.x*blockDim.x + threadIdx.x;
		size_t row=blockIdx.y*blockDim.y + threadIdx.y;
		size_t idx=row*img_wd+col;

		float tmp=0;

		if(row<img_ht && col<img_wd)
		{
			for(int l=0; l<k;l++)
			{
				float val=padding(Pixel_in, col+ l-(k-1)/2, row, img_wd, img_ht);
				tmp+=(val) * kernel[l]/3;
			}
			Pixel_out[idx]=tmp;
		}
	}

	__global__ void squared_elem_add(float* G1_gpu, float* G2_gpu, float* G_gpu,int img_wd, int img_ht)
	{
	  size_t col=blockIdx.x*blockDim.x + threadIdx.x; //column
	  size_t row=blockIdx.y*blockDim.y + threadIdx.y; //row

	  size_t idx=row*img_wd+col;  //index

	  if(col>img_wd || row>img_ht)
	  	return;

	  G_gpu[idx]=G1_gpu[idx]*G1_gpu[idx] + G2_gpu[idx]*G2_gpu[idx]; //adding G1 and G2
	}

	float error_calculation(point* centers_curr,point* centers_prev,int N)
	{
		float err=0;
		for(int i=0;i<N;i++)
		{
			err+=pow((centers_curr[i].x-centers_prev[i].x),2) + pow((centers_curr[i].y-centers_prev[i].y),2);
	    // cout<<i<<" "<<"curr = ("<<centers_curr[i].x<<","<<centers_curr[i].y<<") , prev= ("<<centers_prev[i].x<<","<<centers_prev[i].y<<")"<<endl;
		}

		err=((float)err)/N;
		return err;
	}

	int main(int argc, char* argv[])
	{
	    // time_t start=time(NULL);
		cudaEvent_t start, stop;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
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
	    
	    cudaEventRecord(start);

	    getline(infile,line); //this line has version related comment, hence ignoring
	    getline(infile,line); //this stores image dims

	    istringstream iss2(line);
	    iss2>>word;// this will be image width
	    img_wd=word;
	    iss2>>word;// this will be image height
	    img_ht=word;
	    
	    cout<<img_ht<<" "<<img_wd<<endl;

	    //storing the pixels as 1d images
	    pixel_RGB *Pixel = (pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));
	    
	    int pix_cnt=0, cnt=0;

	    getline(infile,line); //this stores max value
	    istringstream iss3(line);
	    iss3>>word;

	    max_pixel_val=word;//max pixel value
	    cout<<max_pixel_val<<endl;
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
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);

	    float milliseconds=0;
			cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds


			cout<<"File read in "<<milliseconds<<" ms"<<endl;

	    //COLOR CONVERSION
	    //RGB->XYZ->CIE-L*ab

	    //RGB to XYZ
	    // time_t t9= time(NULL);
			cudaEventRecord(start);
			pixel_XYZ *Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
			
	    //XYZ TO CIE-L*ab
			pixel_XYZ* Pixel_LAB=XYZ_LAB(Pixel_XYZ, img_ht, img_wd);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds

	    cout<<"Colorspace conversion done in "<<milliseconds<<" ms"<<endl;
	    //IMPLEMENTING SLIC ALGORITHM
	    int N = img_ht*img_wd;  //number of pixels in the images
	    int K = atoi(argv[2]);    //number of superpixels desired

	    int S= floor(sqrt(N/K));//size of each superpixel
	    float m=atof(argv[3]);    //compactness control constant
	    
	    int k1=ceil(img_ht*1.0/S)*ceil(img_wd*1.0/S);//actual number of superpixels
	    // cout<<k1<<" "<<S<<" "<<float(img_ht*1.0/S)<<" "<<float(img_wd*1.0/S)<<endl;
	    point* centers_curr=(point*)malloc(k1*sizeof(point));

	    //initialize centers

	    cudaEventRecord(start);

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
	    
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    cout<<"centres initialized in "<<milliseconds<<" ms"<<endl;
	    //perturb centers

	    cudaEventRecord(start);
	    float* G=(float*)malloc(N*sizeof(float)); 
	    for(int i=0; i<img_wd;i++)
	    {
	    	for(int j=0; j<img_ht;j++)
	    	{
	    		int index=j*img_wd+i;
	    		float L1, L2, L3, L4, a1, a2, a3, a4, b1, b2, b3, b4;
	    		L1=L2=L3=L4=a1=a2=a3=a4=b1=b2=b3=b4=0;

	     		// cout<<i<<" "<<j<<endl;
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
	    
	    cudaDeviceProp prop;

	    for(int i=0; i<k1;i++)  //for every component
	    {
	    	int x1=centers_curr[i].x-1;
	    	int x2=centers_curr[i].x+1;
	    	int y1=centers_curr[i].y-1;
	    	int y2=centers_curr[i].y+1;

	    	int index = min_index(G, N, x1, x2, y1, y2, img_wd);
	    	centers_curr[i].x=(floor)(index%img_wd);
	    	centers_curr[i].y=(floor)(index/img_wd);

	    } 
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    cout<<"centres perturbed in "<<milliseconds<<" ms"<<endl;
	    
	    ///label initialized to all -1
	    int* labels=(int*)malloc(N*sizeof(int));
	    float* d=(float*)malloc(N*sizeof(float));
	    
	    cudaEventRecord(start);	    
	    for(int idx=0;idx<N;idx++)
	    {
	    	labels[idx]=-1; 
	    	d[idx]=60000;
	    }
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    
	    cout<<"labels and distance measures initialized in "<<milliseconds<<" ms"<<endl;

	    float error=10;

	    point* centers_prev=(point*)(malloc(k1*sizeof(point)));
	    int epoch=0;
	    while(error>0.005)
	    {
	    	cout<<"Epoch number "<<epoch<<endl;


	    	point* centers_gpu;
	    	float* d_gpu;
	    	int* labels_gpu;
	    	pixel_XYZ* Pixel_LAB_gpu;
	    	HANDLE_ERROR(cudaMalloc(&centers_gpu, k1*sizeof(point)));

	    	HANDLE_ERROR(cudaMalloc(&labels_gpu, N*sizeof(int)));

	    	HANDLE_ERROR(cudaMalloc(&Pixel_LAB_gpu, N*sizeof(pixel_XYZ)));
	    	HANDLE_ERROR(cudaMalloc(&d_gpu, N*sizeof(float)));
	    	cudaDeviceProp prop;

	    	float thread_block1=prop.maxThreadsPerBlock;
	    	dim3 DimGrid1(ceil(k1/thread_block1),1,1); 
	    	dim3 DimBlock1(ceil(thread_block1),1,1);

	    // cout<<N<<" "<<S<<" "<<K<<" "<<k1<<" "<<DimGrid1.x<<" "<<DimBlock1.x<<endl;

	    	HANDLE_ERROR(cudaMemcpy(labels_gpu, labels, N*sizeof(int), cudaMemcpyHostToDevice));
	    	HANDLE_ERROR(cudaMemcpy(centers_gpu, centers_curr, k1*sizeof(point), cudaMemcpyHostToDevice));
	    	HANDLE_ERROR(cudaMemcpy(Pixel_LAB_gpu, Pixel_LAB, N*sizeof(pixel_XYZ), cudaMemcpyHostToDevice));
	    	HANDLE_ERROR(cudaMemcpy(d_gpu, d , N*sizeof(int), cudaMemcpyHostToDevice));

	    	cudaEventRecord(start);

	    	label_assignment<<<DimGrid1,DimBlock1>>>(labels_gpu,Pixel_LAB_gpu,centers_gpu,S,img_wd,m, d_gpu, k1);

	    	cudaEventRecord(stop);
	    	cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    

	    	cout<<"Distances calculated for all points neighboured to centres in "<<milliseconds<<" ms"<<endl;// in "<<double(t2-t1)<<" secs"<<endl;

	    //update cluster centres
	    	// t1=time(NULL);

	    	for(int i=0; i<k1;i++)
	    	{
	    centers_prev[i].x=centers_curr[i].x; //saving current centres, before any recalculation
	    centers_prev[i].y=centers_curr[i].y;
	  }
	  cudaEventRecord(start);
	  update_centres<<<DimGrid1,DimBlock1>>>(labels_gpu, centers_gpu, S, img_wd, k1);
	  cudaEventRecord(stop);
	  cudaEventSynchronize(stop);
	  cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    
	//copy back centres, labels, d
	    HANDLE_ERROR(cudaMemcpy(centers_curr, centers_gpu, k1*sizeof(point), cudaMemcpyDeviceToHost));
	    HANDLE_ERROR(cudaMemcpy(d, d_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
	    HANDLE_ERROR(cudaMemcpy(labels, labels_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));

	  // t2=time(NULL);
	    cout<<"cluster centers updated in "<<milliseconds<<" ms"<<endl;

	    //error calculation
	    cudaEventRecord(start);
	    error= error_calculation(centers_curr, centers_prev,k1);
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);

	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds
	    
	    cout<<"error = "<<error<<" and is calculated in "<<milliseconds<<" ms"<<endl;
	    HANDLE_ERROR(cudaFree(labels_gpu));
	    HANDLE_ERROR(cudaFree(Pixel_LAB_gpu));
	    HANDLE_ERROR(cudaFree(centers_gpu));
	    HANDLE_ERROR(cudaFree(d_gpu));
	    epoch++;
	  }

	  pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));

	//randomly shuffle the labels
	  random_shuffle(labels,labels+k1);
	  float alpha=0.4;
	  cudaEventRecord(start);
	  for(int i=0;i<img_ht*img_wd;i++)
	  {
	  	int label_val=labels[i];
	      // cout<<label_val<<endl;
	  	rgb[i].r=alpha*(21*label_val%255) + (1-alpha)*Pixel[i].r;
	  	rgb[i].g=alpha*(137*label_val%255) + (1-alpha)*Pixel[i].g;
	  	rgb[i].b=alpha*(23*label_val%255) + (1-alpha)*Pixel[i].b;
	  }

	    //labelling the centers
	  for(int i=0; i<k1;i++)  
	  {
	  	int x_coord=centers_curr[i].x;
	  	int y_coord=centers_curr[i].y;
	      // cout<<x_coord<<" "<<y_coord<<endl;
	  	for (int x=x_coord-5; x<x_coord+5; x++)
	  	{
	  		for(int y=y_coord-5; y<y_coord+5; y++)
	  		{
	  			int idx=img_wd*y_coord + x_coord;
	          rgb[idx].r= 0;//NULL;//(unsigned char) 0; 
	          rgb[idx].g= 0;//(unsigned char) 0; 
	          rgb[idx].b= 0;//(unsigned char) 0;
	        }
	      }
	    }
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds

	    cout<<"Image prepared in "<<milliseconds<<" ms"<<endl;

	    //OUTPUT STORAGE
	    cudaEventRecord(start);
	    ofstream ofs;
	    ofs.open("output1.ppm", ofstream::out);
	    ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";

	    for(int j=0; j <img_ht*img_wd;j++)
	    	ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;

	    ofs.close();
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&milliseconds, start, stop);//get the time in milliseconds

	    cout<<"Image saved in "<<milliseconds<<" ms"<<endl;
	    return 0;
	  }