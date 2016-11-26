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
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

// storing values for xyz and lab colorspace images
struct pixel_XYZ
{
  float x;
  float y;
  float z;
};

//store coordinates for each pixel
struct point
{ 
  int x;
  int y;
};

__host__ __device__ point* initial_centre(vector<int> label_vector, int* labelled_ini, int N, int img_wd, point* centers_curr)
{
  int ctr_cnt=0;
  for(vector<int>::iterator it=label_vector.begin();it!=label_vector.end();++it)
  {
    int *p;
    int pixel_count=0;
    float x_mean=0, y_mean=0;
    p=find(labelled_ini, labelled_ini+N,*it);
    while(p!=labelled_ini+N)
    { //cout<<*p<<" FOUND at: "<<p-labelled_ini<<endl;
      int index=p-labelled_ini;
      int x_coord=index%img_wd;
      int y_coord=index/img_wd;
      pixel_count++;
      x_mean+=x_coord;
      y_mean+=y_coord;
      p=find(p+1, labelled_ini+N,*it);
    }
    x_mean=x_mean/pixel_count;
    y_mean=y_mean/pixel_count;
    centers_curr[ctr_cnt].x=floor(x_mean);
    centers_curr[ctr_cnt].y=floor(y_mean);
  }
  return centers_curr;
}

__global__ void squared_elem_add(int* G1_gpu, int* G2_gpu,int* G_gpu,int img_wd, int img_ht)
{
  size_t col=blockIdx.x*blockDim.x + threadIdx.x;
  size_t row=blockIdx.y*blockDim.y + threadIdx.y;

  size_t idx=row*img_wd+col;

  if(col>img_wd || row>img_ht)
    return;

  G_gpu[idx]=G1_gpu[idx]*G1_gpu[idx] + G2_gpu[idx]*G2_gpu[idx];
}

__host__ __device__ int padding(int* labelled, int x_coord, int y_coord, int img_width, int img_height) 
{ int val=0;
  if(x_coord< img_width && y_coord <img_height && x_coord>=0 && y_coord>=0)
  {
    val=labelled[y_coord*img_width+x_coord];
  }
  return val;
}

__global__ void vertical_conv(int* labelled_in, int* labelled_out,int img_wd, int img_ht, float* kernel, int k)
{
  size_t col=blockIdx.x*blockDim.x + threadIdx.x;
  size_t row=blockIdx.y*blockDim.y + threadIdx.y;

  size_t idx=row*img_wd+col;

  float tmp=0;    
  
  if(row<img_ht && col<img_wd){

    for(int l=0;l<k;l++)
    {
      int val=padding(labelled_in, col, (row+l-(k-1)/2), img_wd, img_ht);
      tmp+=val * kernel[l];
    }

    labelled_out[idx]=tmp;
  }
}     


__global__ void horizontal_conv(int* labelled_in, int* labelled_out, int img_wd, int img_ht, float* kernel, int k)
{
  size_t col=blockIdx.x*blockDim.x + threadIdx.x;
  size_t row=blockIdx.y*blockDim.y + threadIdx.y;
  size_t idx=row*img_wd+col;

  float tmp=0;

  if(row<img_ht && col<img_wd)
  {
    for(int l=0; l<k;l++)
    {
      int val=padding(labelled_in, col+ l-(k-1)/2, row, img_wd, img_ht);
      tmp+=val * kernel[l];
    }
    labelled_out[idx]=tmp;
  }
}


__host__ __device__ int max_index(int* array, int size, int x1, int x2, int y1, int y2, int img_wd)
{
  int index=0;
  //finding max values from (X1,y1) to (X2,y2)
  for(int i=0;i<size;i++)
  {
    if(int(i%img_wd)>=x1 && int(i%img_wd)<=x2 && int(i/img_wd)>=y1 && int(i/img_wd)<=y2)
    {  
      if(array[i]>=array[index])
        index=i;
    }
  }
  return index;
}

__host__ __device__ int min_index(int* array, int size, int x1, int x2, int y1, int y2, int img_wd)
{
  int index=0;
  for(int i=0;i<size;i++)
  {
    if(int(i%img_wd)>=x1 && int(i%img_wd)<=x2 && int(i/img_wd)>=y1 && int(i/img_wd)<=y2)
    { 
      if(array[i]<array[index])
        index=i;
    }
  }
  return index;
}

//color space conversion from RGB to XYZ
__host__ __device__ pixel_XYZ* RGB_XYZ(pixel_RGB* img ,int ht ,int wd)
{ 
  pixel_XYZ *XYZ=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ)));

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
__host__ __device__ pixel_XYZ* XYZ_LAB(pixel_XYZ* img ,int ht ,int wd)
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

//label2rgb

int main(int argc, char* argv[])
{
  time_t start=time(NULL);

  if(argc != 2) //there should be three arguments
    return 1; //exit and return an error

  //READING FILE
  
  ifstream infile;
  infile.open(argv[1]);
  string line;

  int img_wd, img_ht;
  int max_pixel_val;
  int line_count=0;

  //line one contains P6, line 2 mentions about gimp version, line 3 stores the height and width
  getline(infile, line);
  istringstream iss1(line);

  //reading first line to check format
  int word;
  string str1;
  iss1>>str1;
  //cout<<"str1="<<str1<<endl;
  
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

  //COLOR CONVERSION
  //RGB->XYZ->CIE-L*ab

  //RGB to XYZ
  pixel_XYZ *Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
  
  //XYZ TO CIE-L*ab
  pixel_XYZ* Pixel_LAB=XYZ_LAB(Pixel_XYZ, img_ht, img_wd);
  
  //IMPLEMENTING SLIC ALGORITHM
  int N = img_ht*img_wd;  //number of pixels in the images
  int K = 400;    //number of superpixels desired

  int S= floor(sqrt(N/K));//size of each superpixel
  float m= 10;    //compactness control constant
  
  int k1=ceil(img_ht/S)*ceil(img_wd/S);
  //initial labelling
  int* labelled_ini = (int*)malloc(N*sizeof(int));  //row major wise storing the labels
  int count=0;

  vector<int> label_vector;
  
  for(int i=0;i<(1+img_ht/S)*(1+img_wd/S);i++)
    label_vector.push_back(i);

  random_shuffle(label_vector.begin(),label_vector.end());
  
  vector<int>::iterator it=label_vector.begin();

  //initialize labels
  for(int i=0; i<img_wd;i=i+S)
  {
    for(int j=0; j<img_ht;j=j+S)
    {
      for(int x=i;x<i+S;x++)
        {
        for(int y=j;y<j+S;y++)
          {
          if(x<img_wd && y<img_ht)
            {
            int idx=y*img_wd+x;
            labelled_ini[idx]=*it;  
            }
          }
        }
      ++it; 
    }
  }

  //get initial cluster centers
  point* centers_curr=(point*)malloc(k1*sizeof(point));
  centers_curr=initial_centre(label_vector, labelled_ini, N, img_wd, centers_curr);

  //perturb centers in a 3x3 neighborhood
  float *K1 = (float *)malloc(3 *sizeof(float)); 
  float *K2 = (float *)malloc(3 *sizeof(float));
  
  K1[0]=0; K1[1]=1.1892; K1[2]=0;
  K2[0]=-0.8409; K2[1]=0; K2[2]=0.8409;

  int *labelled_gpu, *labelled_tmp_gpu, *G1_gpu, *G2_gpu, *G_gpu;

  HANDLE_ERROR(cudaMalloc(&labelled_gpu, N*sizeof(int)));
  
  HANDLE_ERROR(cudaMalloc(&labelled_tmp_gpu, N*sizeof(int)));

  float *K1_gpu, *K2_gpu;

  HANDLE_ERROR(cudaMalloc(&K1_gpu, 3*sizeof(float)));

  HANDLE_ERROR(cudaMalloc(&K2_gpu, 3*sizeof(float)));

  HANDLE_ERROR(cudaMalloc(&G1_gpu, N*sizeof(int)));

  HANDLE_ERROR(cudaMalloc(&G2_gpu, N*sizeof(int)));

  HANDLE_ERROR(cudaMalloc(&G_gpu, N*sizeof(int)));

  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));  //using GPU0

  HANDLE_ERROR(cudaMemcpy(labelled_gpu, labelled_ini, N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(labelled_tmp_gpu, labelled_ini, N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(K1_gpu, K1, 3*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(K2_gpu, K2, 3*sizeof(int), cudaMemcpyHostToDevice));


  int* G=(int*)malloc(N*sizeof(int)); 

  float thread_block=sqrt(prop.maxThreadsPerBlock);
  dim3 DimGrid(ceil(img_wd/thread_block),ceil(img_ht/thread_block),1); //image saved as a 2D grid
  dim3 DimBlock(thread_block,thread_block,1);


  vertical_conv<<<DimGrid,DimBlock>>>(labelled_gpu, labelled_tmp_gpu,img_wd, img_ht,K1_gpu,3);
  horizontal_conv<<<DimGrid, DimBlock>>>(labelled_tmp_gpu, G1_gpu, img_wd, img_ht, K2_gpu, 3);

  vertical_conv<<<DimGrid,DimBlock>>>(labelled_gpu, labelled_tmp_gpu,img_wd, img_ht,K2_gpu,3);
  horizontal_conv<<<DimGrid, DimBlock>>>(labelled_tmp_gpu, G2_gpu, img_wd, img_ht, K1_gpu, 3);

  squared_elem_add<<<DimGrid,DimBlock>>>(G1_gpu,G2_gpu,G_gpu,img_wd,img_ht);

  HANDLE_ERROR(cudaMemcpy(G, G_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));

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


  int num_iterations=100;

  for(int epoch=1; epoch<num_iterations; epoch++)
  {
    
  }

  pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));
  int label_prev_val=labelled_ini[0];

  //getting labelled image

  for(int i=0;i<img_ht*img_wd;i++)
  {
    int label_val=labelled_ini[i];
    rgb[i].r=21*label_val%256;
    rgb[i].g=47*label_val%256;
    rgb[i].b=173*label_val%256;
  }
  
  //OUTPUT STORAGE
  ofstream ofs;
  ofs.open("output.ppm", ofstream::out);
  ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";

  for(int j=0; j <img_ht*img_wd;j++)
    ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;//labelled_ini[j]<<0<<0;//ofs<<Pixel_LAB[j].x<<Pixel_LAB[j].y<<Pixel_LAB[j].z; //write as ascii

  // ofs.close();
  
  return 0;
}