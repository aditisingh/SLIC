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

  int min_index(float* array, int size, int x1, int x2, int y1, int y2, int img_wd)
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

  // __global__ void squared_elem_add(float* G1_gpu, float* G2_gpu, float* G_gpu,int img_wd, int img_ht)
  // {
  //   size_t col=blockIdx.x*blockDim.x + threadIdx.x; //column
  //   size_t row=blockIdx.y*blockDim.y + threadIdx.y; //row

  //   size_t idx=row*img_wd+col;  //index

  //   if(col>img_wd || row>img_ht)
  //     return;

  //   G_gpu[idx]=G1_gpu[idx]*G1_gpu[idx] + G2_gpu[idx]*G2_gpu[idx]; //adding G1 and G2
  // }

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

    if(argc != 2) //there should be three arguments
      return 1; //exit and return an error

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
    cout<<"File read"<<endl;

    //COLOR CONVERSION
    //RGB->XYZ->CIE-L*ab

    //RGB to XYZ
    // time_t t9= time(NULL);
    pixel_XYZ *Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
    
    //XYZ TO CIE-L*ab
    pixel_XYZ* Pixel_LAB=XYZ_LAB(Pixel_XYZ, img_ht, img_wd);
    // time_t t10=time(NULL);

    cout<<"Colorspace conversion done"<<endl;
    //IMPLEMENTING SLIC ALGORITHM
    int N = img_ht*img_wd;  //number of pixels in the images
    int K = 100;    //number of superpixels desired

    int S= floor(sqrt(N/K));//size of each superpixel
    float m= 10;    //compactness control constant
    
    int k1=(1+img_ht/S)*(1+ img_wd/S);//actual number of superpixels
    
    //initialize centers
    time_t t1=time(NULL);	
    point* centers_curr=(point*)malloc(k1*sizeof(point));
    int center_ctr=0;
    for(int j=float(S/2)-1;j<img_ht;j=j+S)
    {
      for(int i=float(S/2)-1;i<img_wd;i=i+S)
      {
        centers_curr[center_ctr].x=i;
        centers_curr[center_ctr].y=j;
        center_ctr++;
      }
    }
    time_t t2=time(NULL);
    cout<<"centres initialized in "<<double(t2-t1)<<" secs"<<endl;

    //perturb centers

    float* Pixel_gray=(float*)malloc(N*sizeof(float));
    
    for(int index=0; index<img_wd*img_ht ;index++)
      Pixel_gray[index]=(Pixel[index].r + Pixel[index].g + Pixel[index].b)/3;

    float *K1 = (float *)malloc(3 *sizeof(float)); 
    float *K2 = (float *)malloc(3 *sizeof(float));
    
    K1[0]=0; K1[1]=1.1892; K1[2]=0;
    K2[0]=-0.8409; K2[1]=0; K2[2]=0.8409;

    float* Pixel_gray_gpu;

    HANDLE_ERROR(cudaMalloc(&Pixel_gray_gpu, N*sizeof(pixel_RGB)));
    
    float *K1_gpu, *K2_gpu,  *G1_gpu, *G2_gpu, *G_gpu, *Pixel_tmp;;

    HANDLE_ERROR(cudaMalloc(&K1_gpu, 3*sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&K2_gpu, 3*sizeof(float)));

    cudaDeviceProp prop;

    HANDLE_ERROR(cudaMalloc(&G1_gpu, N*sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&G2_gpu, N*sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&G_gpu, N*sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&Pixel_tmp, N*sizeof(float)));

    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));  //using GPU0

    HANDLE_ERROR(cudaMemcpy(Pixel_gray_gpu, Pixel_gray, N*sizeof(pixel_RGB), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(K1_gpu, K1, 3*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(K2_gpu, K2, 3*sizeof(float), cudaMemcpyHostToDevice));


    float* G=(float*)malloc(N*sizeof(float)); 

    float thread_block=sqrt(prop.maxThreadsPerBlock);
    dim3 DimGrid(ceil(img_wd/thread_block),ceil(img_ht/thread_block),1); //image saved as a 2D grid
    dim3 DimBlock(thread_block,thread_block,1);

    t1 =time(NULL);
    vertical_conv<<<DimGrid,DimBlock>>>(Pixel_gray_gpu, Pixel_tmp,img_wd, img_ht,K1_gpu,3);
    horizontal_conv<<<DimGrid, DimBlock>>>(Pixel_tmp, G1_gpu, img_wd, img_ht, K2_gpu, 3);

    vertical_conv<<<DimGrid,DimBlock>>>(Pixel_gray_gpu, Pixel_tmp,img_wd, img_ht,K2_gpu,3);
    horizontal_conv<<<DimGrid, DimBlock>>>(Pixel_tmp, G2_gpu, img_wd, img_ht, K1_gpu, 3);

    squared_elem_add<<<DimGrid,DimBlock>>>(G1_gpu,G2_gpu,G_gpu,img_wd,img_ht);

    HANDLE_ERROR(cudaMemcpy(G, G_gpu, N*sizeof(float), cudaMemcpyDeviceToHost));

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
    t2=time(NULL);
    cout<<"centres perturbed in "<<double(t2-t1)<<" secs"<<endl;
	
    ///label initialized to all -1
    int* labels=(int*)malloc(N*sizeof(int));
    float* d=(float*)malloc(N*sizeof(float));
    
    t1=time(NULL);
    for(int idx=0;idx<N;idx++)
    {
     labels[idx]=-1; 
     d[idx]=60000;
   }
   t2=time(NULL);
   cout<<"labels and distance measures initialized in "<<double(t2-t1)<<" secs"<<endl;

   int num_iterations=10;

   for(int epoch=0; epoch<num_iterations; epoch++)
   {
    cout<<"Epoch number "<<epoch<<endl;
    point* centers_prev=centers_curr; //saving current centres, before any recalculation

    t1=time(NULL);
    for(int i=0; i<k1;i++)//for every cluster center
    {
      cout<<"center number: "<<i<<endl;
      int x_center=centers_curr[i].x;
      int y_center=centers_curr[i].y;
      int index_center=y_center*img_wd+x_center;
      //for neighborhood search in 2Sx2S area around the center
      for(int x_coord=x_center-S; x_coord<=x_center+S; x_coord++)
      {
        for(int y_coord=y_center-S; y_coord<=y_coord+S; y_coord++)
        {
          if(x_coord>0 && x_coord<=img_wd && y_coord>0 && y_coord<=img_ht)
        {
          int j=y_coord*img_wd+ x_coord;
          float d_c = pow(pow((Pixel_LAB[index_center].x-Pixel_LAB[j].x),2) + pow((Pixel_LAB[index_center].y-Pixel_LAB[j].y),2) + pow((Pixel_LAB[index_center].z-Pixel_LAB[j].z),2),0.5); //color proximity;
          float d_s = pow(pow(x_coord-x_center,2)+pow(y_coord-y_center,2),0.5); //spatial proximity
          float D=pow(pow(d_c,2)+pow(m*d_s/S,2),0.5);

          if(D<d[j])
          {
            d[j]=D;
            labels[j]=i;
          }
        }
        }
      }
    }
    t2=time(NULL);
    cout<<"Distances calculated for all points neighboured to centres in "<<double(t2-t1)<<" secs"<<endl;

    //update cluster centres
	t1=time(NULL);
     for(int i=0; i<k1;i++)
	{
	int x_mean=0, y_mean=0, count=0, flag=0;
	for(int j=0; j<N; j++)
	{
	if(labels[j]=i)
	{
	int x_coord=j%img_wd;
	int y_coord=j/img_wd;
	x_mean+=x_coord;
	y_mean+=y_coord;
	flag++;
	cout<<i<<" "<<x_mean<<" "<<y_mean<<" "<<count<<endl;
	count++;
	}
	}
	cout<<"count ="<<count<<endl;
	if(flag)
	{
	centers_curr[i].x=x_mean/count;
	centers_curr[i].y=y_mean/count;
	}
	}

	t2=time(NULL);
    cout<<"cluster centers updated in "<<double(t2-t1)<<" secs"<<endl;

    //error calculation
	t1=time(NULL);
    float error= error_calculation(centers_curr, centers_prev,k1);
    t2=time(NULL);
    cout<<"error = "<<error<<" and is calculated in "<<double(t2-t1)<<" secs"<<endl;
   }

    pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));
   
    float alpha=0.4;
    t1=time(NULL);
    for(int i=0;i<img_ht*img_wd;i++)
    {
      int label_val=labels[i];
      rgb[i].r=alpha*((rand()%256)*label_val%255) + (1-alpha)*Pixel[i].r;
      rgb[i].g=alpha*((rand()%256)*label_val%255) + (1-alpha)*Pixel[i].g;
      rgb[i].b=alpha*((rand()%256)*label_val%255) + (1-alpha)*Pixel[i].b;
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
    t2=time(NULL);
    cout<<"Image prepared in "<<double(t2-t1)<<" secs"<<endl;

    //OUTPUT STORAGE
    t1=time(NULL);
    ofstream ofs;
    ofs.open("output1.ppm", ofstream::out);
    ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";

    for(int j=0; j <img_ht*img_wd;j++)
      ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;

    ofs.close();
    t2=time(NULL);
    cout<<"Image saved in "<<double(t2-t1)<<" secs"<<endl;
    return 0;
  }