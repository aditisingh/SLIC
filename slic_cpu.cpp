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
#include <chrono>

using namespace std;


// storing RGB values for rgb colorspace images
struct pixel_RGB
{
  unsigned char r;//red channel
  unsigned char g;//green channel
  unsigned char b;//blue channel
};

// storing values for lab colorspace images
struct pixel_XYZ
{
  float x;//L channel
  float y;//A channel
  float z;//B channel
};

//store coordinates for each pixel
struct point
{ 
  int x;//x coordinates
  int y;//y coordinates
};


int min_index(float* array, int size, int x1, int x2, int y1, int y2, int img_wd) //find the index of min value a given region
{
  int index=(x1+1)+(y1+1)*img_wd; //initialize to the centre index
  for(int x=x1;x<x2;x++)
  {
    for(int y=y1;y<y2;y++)
    {
      if(array[y*img_wd+x]<array[index])
        index=y*img_wd+x;
    }
  }
  return index;
}

//color space conversion from RGB to LAB
pixel_XYZ* RGB_LAB(pixel_RGB* img ,int ht ,int wd)
{ 
  pixel_XYZ *LAB_img=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ))); //declaring same sized output image

  for(int i=0; i<ht*wd;i++)
  {
    //read the RGB channel values
    int R=img[i].r; 
    int G=img[i].g;
    int B=img[i].b;

    //normalize these values
    double var_R=double(R)/255;
    double var_G=double(G)/255;
    double var_B=double(B)/255;

    //linearize it to give XYZ colorspace
    double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
    double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
    double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

    //Normalize XYZ values
    X=X/0.95047;
    Y=Y/1.00000;
    Z=Z/1.088969;

    //Conversion of XYZ to LAB Values
    double Y3=pow(Y,1/3);

    double T=0.008856;//threshold
    double fx=(X>T)?(pow(X,double(1)/3)):(7.787*X+(16/116));
    double fy=(Y>T)?(pow(Y,double(1)/3)):(7.787*Y+(16/116));
    double fz=(Z>T)?(pow(Z,double(1)/3)):(7.787*Z+(16/116));


    double L=(Y>T)?(116*Y3 - 16):(903.3*Y);
    double a = 500 * (fx - fy);
    double b = 200 * (fy - fz);

    //saving the calculations to image
    LAB_img[i].x=L;
    LAB_img[i].y=a;
    LAB_img[i].z=b;
  }
  return LAB_img;
}

//calculating residual error(MSE) between previous and current centres
float error_calculation(point* centers_curr,point* centers_prev,int N)
{
  float err=0; //initialize MSE to zero
  for(int i=0;i<N;i++)
  {
    err+=pow((centers_curr[i].x-centers_prev[i].x),2) + pow((centers_curr[i].y-centers_prev[i].y),2); 
    //squared error between current and previous coordinates
  }

  err=((float)err)/N; //take mean of the squared error
  return err; //
}

int main(int argc, char* argv[])
{
  using namespace std::chrono; //for calculating the CPU processes timing to milliseconds accuracy

  cout<<"Simple Linear Iterative Clustering: CPU IMPLEMENTATION"<<endl<<endl;
  high_resolution_clock::time_point start = high_resolution_clock::now();// store image reading and saving time
  if(argc != 4) //there should be three arguments
  {
    cout<<" program_name image_name num_superpixels control_constant"<<endl;
    return 1; 
  }//exit and return an error

  //READING FILE
  //reading the file line by line

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

  //storing the pixels as 1d images, row major format
  pixel_RGB *Pixel = (pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));

  int pix_cnt=0, cnt=0; // to parse through lines and read r,g,b channels

  getline(infile,line); //this stores max value
  istringstream iss3(line);
  iss3>>word;

  max_pixel_val=word;//max pixel value
  // cout<<max_pixel_val<<endl;
  unsigned int val;

  //read line by line
  while (getline(infile, line))
  {
    istringstream iss4(line);
    for (int i=0; i<=line.length();i++)
    {
      if(pix_cnt<img_ht*img_wd) //if it a valid pixel
      {
        val =((int)line[i]);  //read the current line
        if(cnt%3==0)  //in case of R channel
        {
          Pixel[pix_cnt].r=val; //store R channel value
        }
        else if(cnt%3==1) //in case of G channel
        {
          Pixel[pix_cnt].g=val;//storing G value
        }
        else  
        {
          Pixel[pix_cnt].b=val;//in case of B channel, store it
          pix_cnt++;  //move to next pixel
        }
        cnt++;  //next value read
      }
    }
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2- start);

  cout<<"Image read in "<<time_span.count() <<" s"<<endl; //time taken to read and save the image

  //COLOR CONVERSION
  //RGB->XYZ->CIE-L*ab

  high_resolution_clock::time_point t3=high_resolution_clock::now();//store time to convert RGB to LAB
  //RGB to LAB
  pixel_XYZ* Pixel_LAB=RGB_LAB(Pixel, img_ht, img_wd);//convert RGB to LAB
  high_resolution_clock::time_point t4 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t4-t3);//time for colorspace conversion

  cout<<"colorspace conversion done in "<<time_span.count() <<" s"<<endl;


  //IMPLEMENTING SLIC ALGORITHM
  int N = img_ht*img_wd;  //number of pixels in the images
  int K = atoi(argv[2]);    //number of superpixels desired

  int S= floor(sqrt(N/K));//size of each superpixel
  float m= atof(argv[3]);    //compactness control constant

  int k1=ceil(img_ht*1.0/S)*ceil(img_wd*1.0/S);//actual number of superpixels
  cout<<"Width = "<<img_wd<<", Height = "<<img_ht<<endl;
  cout<<" Number of superpixels is "<<k1<<" each of size "<<S<<" X "<<S<<" and area"<<S*S<<endl;

  //storing centers and initializing them
  point* centers_curr=(point*)malloc(k1*sizeof(point));

  int center_ctr=0;
  high_resolution_clock::time_point t5 =high_resolution_clock::now();//calculated centre initialization time
  //centres are initialized in a regular grid, each separated by S distance to the nearest centre
  // centres start from (S/2,S/2)
  for(int j=S/2;j<S*ceil(img_ht*1.0/S);j=j+S)
  {
    for(int i=S/2;i<S*ceil(img_wd*1.0/S);i=i+S)
    {
      int val1=((i>=img_wd)?(img_wd+j-S)/2:i);//to make sure it doesn't go out of image
      int val2=((j>=img_ht)?(img_ht+i-S)/2:j);//same as above in y coordinate
      //store x and y coordinates into the array
      centers_curr[center_ctr].x=val1; 
      centers_curr[center_ctr].y=val2;
      center_ctr++;
    }
  }
  high_resolution_clock::time_point t6 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t6-t5);//saving the time of centre initialization


  cout<<"centres initialized in "<<time_span.count() <<" s"<<endl;
  ///perturb centers
  high_resolution_clock::time_point t7=high_resolution_clock::now();//to store perturbation time
  float* G=(float*)malloc(N*sizeof(float)); //to store gradient in 3x3 neighborhood
  //gradient is calculated as : G(x, y) = \I(x + 1, y) − I(x − 1, y)|^2+ |I(x, y + 1) − I(x, y − 1)|^2
  for(int i=0; i<img_wd;i++)  //x coordinate
  {
    for(int j=0; j<img_ht;j++)  //y coordinate
    {
      int index=j*img_wd+i; //calculating the index, row major
      //To store L,a, b channels for points (x+1,y),(x-1,y),(x,y+1),(x,y-1)
      float L1, L2, L3, L4, a1, a2, a3, a4, b1, b2, b3, b4; 

      //initializing them to zero, so as to give padding effect when at edges
      L1=L2=L3=L4=a1=a2=a3=a4=b1=b2=b3=b4=0;
      //replace by actual intensities in LAB colorspace when the pixel exists
      if(i+1<img_wd)
        L1=Pixel_LAB[j*img_wd+i+1].x, a1=Pixel_LAB[j*img_wd+i+1].y, b1=Pixel_LAB[j*img_wd+i+1].z;
      if(i-1>0)
        L2=Pixel_LAB[j*img_wd+i-1].x, a2=Pixel_LAB[j*img_wd+i-1].y, b2=Pixel_LAB[j*img_wd+i-1].z;
      if(j+1<img_ht)
        L3=Pixel_LAB[(j+1)*img_wd+i].x, a3=Pixel_LAB[(j+1)*img_wd+i].y, b3=Pixel_LAB[(j+1)*img_wd+i].z;
      if(j-1>0)
        L4=Pixel_LAB[(j-1)*img_wd+i].x, a4=Pixel_LAB[(j-1)*img_wd+i].y, b4=Pixel_LAB[(j-1)*img_wd+i].z;

      //Calculating the gradient
      G[index]=pow(L1-L2,2) + pow(a1-a2,2) + pow(b1-b2,2) + pow(L3-L4,2) + pow(a3-a4,2) + pow(b3-b4,2);
    }
  }

  for(int i=0; i<k1;i++)  //for every cluster center
  {
    //the minimum gradient is needed in the region (x-1,y-1) to (x+1,y+1)
    int x1=centers_curr[i].x-1; 
    int x2=centers_curr[i].x+1;
    int y1=centers_curr[i].y-1;
    int y2=centers_curr[i].y+1;

    int index = min_index(G, N, x1, x2, y1, y2, img_wd); //finding minimum index in this 3x3 search region

    //calculating new x and y coordinates for the centre 
    centers_curr[i].x=(floor)(index%img_wd);
    centers_curr[i].y=(floor)(index/img_wd);

  } 
  high_resolution_clock::time_point t8 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t8-t7); //storing centre perturbation time


  cout<<"centres perturbed in "<<time_span.count() <<" s"<<endl;

  high_resolution_clock::time_point t9=high_resolution_clock::now();

  int* labels=(int*)malloc(N*sizeof(int)); //this will be storing labels for every pixel
  float* d=(float*)malloc(N*sizeof(float)); // this will be storing distance measure of every pixel to its cluster center

  //initializing the labels and distance measures
  for(int idx=0;idx<N;idx++)
  {
    labels[idx]=-1; //unlabelled 
    d[idx]=60000;   //a high value 
  }
  high_resolution_clock::time_point t10 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t10-t9); // storing time for this initialization

  cout<<"labels and distance measures initialized in "<<time_span.count()<<" s"<<endl;


  float error=100;  //initial error value

  point* centers_prev=(point*)(malloc(k1*sizeof(point)));// this will be storing the cluster centres for every previous epoch
  int epoch=0;//initialization epoch

  while(error>1)
  {
    cout<<endl;
    cout<<"Epoch = "<<epoch<<endl;
    //label assignment
    high_resolution_clock::time_point t11=high_resolution_clock::now(); //store label assignment time
    for(int i=0;i<k1;i++)//for every cluster center
    {
      int x_center=centers_curr[i].x; //find x coordinate of the cluster centre
      int y_center=centers_curr[i].y; //find y coordinate of the cluster centre

      int centre_idx=y_center*img_wd+x_center;//find index in image row major form

      for(int x_coord=max(0,x_center-S);x_coord<=min(x_center+S,img_wd-1);x_coord++) //look in 2S x 2S neighborhood 
      { //taking care it doesn't go out of the image
        for(int y_coord=max(0,y_center-S);y_coord<=min(img_ht-1,y_center+S);y_coord++)
        {
          int j=y_coord*img_wd+x_coord; // find global index of the pixel
          float d_c = pow(pow((Pixel_LAB[centre_idx].x-Pixel_LAB[j].x),2) + pow((Pixel_LAB[centre_idx].y-Pixel_LAB[j].y),2) + pow((Pixel_LAB[centre_idx].z-Pixel_LAB[j].z),2),0.5); //color proximity;
          float d_s = pow(pow(x_coord-x_center,2)+pow(y_coord-y_center,2),0.5); //spatial proximity
          float D=pow(pow(d_c,2)+pow(m*d_s/S,2),0.5);//effective distance
          //if it is lesser than current distance, update
          if(D<d[j])
          {
            d[j]=D;//store new distance
            labels[j]=i;//label as the number of cluster centre
          }
        }
      }
    }
    high_resolution_clock::time_point t12 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t12-t11);//store the label assignment time

    cout<<"Labels assigned in "<<time_span.count()<<" s"<<endl;

    for(int i=0; i<k1;i++)
    {
      centers_prev[i].x=centers_curr[i].x; //saving current centres, before any recalculation
      centers_prev[i].y=centers_curr[i].y;
    }
    //update cluster centres
    high_resolution_clock::time_point t13=high_resolution_clock::now();
    for(int i=0; i<k1;i++)//for every cluster centre
    {
      int x_mean=0, y_mean=0, count=0;//mean will store update cluster
      for(int x_coord=max(0,centers_curr[i].x-S);x_coord<=min(centers_curr[i].x+S,img_wd-1);x_coord++) //look in 2S x 2S neighborhood 
      { //taking care it doesn't go out of the image
        for(int y_coord=max(0,centers_curr[i].y-S);y_coord<=min(img_ht-1,centers_curr[i].y+S);y_coord++)
        {
          if(labels[y_coord*img_wd+x_coord]==i) //if the label is the cluster centres, add x and y coordinates to x_mean and y_mean
          {
            x_mean+=x_coord;
            y_mean+=y_coord;
            count++;  //increment the counter
          }
        }
      }
      if(count) //if any counts
      {
        centers_curr[i].x=x_mean/count; //calculate mean x and y coordinate
        centers_curr[i].y=y_mean/count;
      }
    }
    high_resolution_clock::time_point t14 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t14-t13); //store the time taken for centre updation


    cout<<"Centers updated in "<<time_span.count()<<"s"<<endl;
    //finding error
    high_resolution_clock::time_point t15=high_resolution_clock::now();//store error calculation time

    error= error_calculation(centers_curr, centers_prev,k1);  //MSE using previous and current(updated ) values
    high_resolution_clock::time_point t16 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t16-t15);

    cout<<"MSE= "<<error<<" and is calculated in "<<time_span.count()<<endl;
  
    epoch++;

  }

  /////OUTPUT HANDLING////////////

  pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));///this image will store the output

  ///enforce connectivity, to remove stray pixels
  high_resolution_clock::time_point t17=high_resolution_clock::now();
  //for every point, look into its 4 neighbour labels, if all are same and different from pixel's label, change its label
  for(int x=0; x<img_wd; x++)//x coordinates
  {
    for(int y=0; y<img_ht; y++)//y coordinates
    {
      int L_t=labels[max(y-1,0)*img_wd+x];//top pixel label
      int L_b=labels[min(y+1,img_ht)*img_wd+x];//bottom pixel label
      int L_r=labels[y*img_wd+max(img_wd,x+1)];//right pixel label
      int L_l=labels[y*img_wd+min(0,x-1)];//left pixel label

      if(L_t==L_b && L_b==L_r && L_r==L_l)  //if all neighborhood labels are same, force the centre to be the same label
      {
        labels[y*img_wd+x]=L_t;
      }
    }
  }

  high_resolution_clock::time_point t18 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t18-t17);


  cout<<"connectivity enforced in "<<time_span.count() <<" s"<<endl; //Connectivity enforcement time

  //randomly shuffle the labels, to get components of random colors
  random_shuffle(labels,labels+k1);

  high_resolution_clock::time_point t19=high_resolution_clock::now();
  float alpha=0; //Make this non-zero to generate overlay images, value should be between 0.0 and 1.0
  for(int i=0;i<img_ht*img_wd;i++)//for every pixel
  {
    int label_val=labels[i];  //find the label
    //generate overlay image
    rgb[i].r=alpha*(21*label_val%255) + (1-alpha)*Pixel[i].r;
    rgb[i].g=alpha*(47*label_val%255) + (1-alpha)*Pixel[i].g;
    rgb[i].b=alpha*(173*label_val%255) + (1-alpha)*Pixel[i].b;
  }

  //sobel edge detection to draw boundaries
  int valX, valY = 0; //store x and y convolution result
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

  double val1; //to store the graient 
  for(int i=0;i<img_wd;i++)
  {
    for(int j=0;j<img_ht;j++)
    {
      if(i==0||i==img_wd-1||j==0||j==img_ht-1) //pad for edges
      {
        valX=0;
        valY=0;
      }
      else
      {
        valX=0, valY=0;//initialize 
        //convolution in 3x3 neighborhood
        for (int x = -1; x <= 1; x++)
        {
          for (int y = -1; y <= 1; y++)
          {
            valX = valX + labels[i+x+(j+y)*img_wd] * GX[1+x][1+y]; //x convolution
            valY = valY + labels[i+x+(j+y)*img_wd]  * GY[1+x][1+y];//y convolution
          }
        }
      }
      val1=sqrt(valX*valX + valY*valY);//gradient
      if(val1>0)  //edge detected, color the output image black
      {
        rgb[j*img_wd+i].r=0; 
        rgb[j*img_wd+i].g=0;
        rgb[j*img_wd+i].b=0;
      }
    }
  }
  high_resolution_clock::time_point t20 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t20-t19); //storing output preparation time


  cout<<"Output image prepared in "<<time_span.count() <<" s"<<endl;

  //OUTPUT STORAGE
  high_resolution_clock::time_point t21=high_resolution_clock::now();
  ofstream ofs;//output stream
  ofs.open("output_cpu.ppm", ofstream::out);
  ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";//writing magic number, image width, image height and max pixel value

  for(int j=0; j <img_ht*img_wd;j++)
  {
    ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;//writing RGB values
  }

  ofs.close();//closing the output stream

  high_resolution_clock::time_point stop = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(stop- t21);

  cout<<"Output image saved in "<<time_span.count()<<" s"<<endl;
  time_span = duration_cast<duration<double>>(stop- start);

  cout<<"Clustering done in "<<time_span.count() <<" s"<<endl;

  return 0;
}