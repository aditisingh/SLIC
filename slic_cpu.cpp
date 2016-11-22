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

using namespace std;

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

/*point initial_centre(int* label, val)
{
	
}
*/

int max_value(int* array, int size)
{
	int max_val=array[0];
	for(int i=0;i<size;i++)
	{
		cout<<array[i]<<" ";
		if(array[i]>=max_val)
			max_val=array[i];
	}
	return max_val;
}

int min_value(int* array, int size)
{
	int min_val=array[0];
	for(int i=0;i<size;i++)
	{
		if(array[i]<min_val)
			min_val=array[i];
	}
	return min_val;
}

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

//label2rgb

int main(int argc, char* argv[])
{
	time_t start=time(NULL);
	if(argc != 2) //there should be three arguments
	return 1; //exit and return an error

	//READING FILE
	
	ifstream infile;
	infile.open("stop_1.ppm");
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
	int N = img_ht*img_wd;	//number of pixels in the images
	int K = 400;		//number of superpixels desired

	int S= floor(sqrt(N/K));//size of each superpixel
	float m= 10; 		//compactness control constant
	
	int k1=ceil(img_ht/S)*ceil(img_wd/S);
	//initial labelling
	int* labelled_ini = (int*)malloc(N*sizeof(int));	//row major wise storing the labels
	int count=0;

	vector<int> label_vector;
	
	for(int i=0;i<ceil(img_ht/S)*ceil(img_wd/S);i++)
		label_vector.push_back(i);
	
	random_shuffle(label_vector.begin(),label_vector.end());
	
	vector<int>::iterator it=label_vector.begin();
	for(int i=0; i<img_wd;i=i+S)
	{
	    for(int j=0; j<img_ht;j=j+S)
		{
			cout<<i<<" "<<j<<" "<<*it<<endl;
			for(int x=i;x<i+S;x++)
				{
				for(int y=j;y<j+S;y++)
					{
					if(x<img_wd && y<img_ht)
						{
						int idx=y*img_wd+x;
						labelled_ini[idx]=*it;	
						//cout<<i<<" "<<j<<" "<<x<<" "<<y<<" "<<idx<<" "<<labelled_ini[idx]<<endl;	
						}
					}
				}
			++it;
		}
	}
	//cout<<"k1="<<k1<<", N="<<N<<endl;
	//cout<<S<<endl;
	for(int i=0; i<img_wd;i++)
	{	
		for(int j=0; j<img_ht;j++)
		{
			int idx= j*img_wd+i;
			cout<<i<<" "<<j<<" "<<labelled_ini[idx]<<endl;
		}
	}
	//cout<<labelled_ini[3586]<<endl;
	//get initial cluster centers
	point* centers_curr=(point*)malloc(k1*sizeof(point));

	int * p;
	p=find(labelled_ini, labelled_ini+N,30);
	/*while(p)
	{
	cout<<*p<<endl;
	p=find(labelled_ini, labelled_ini+k1,30);
	}*/

	pixel_RGB *rgb=(pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));
	int label_prev_val=labelled_ini[0];

	//getting labelled range
	/*
	int min_val=min_value(labelled_ini,N);
	int max_val=max_value(labelled_ini,N);

	cout<<"min="<<min_val<<", max="<<max_val<<endl;

	int range=(max_val-min_val+1);
	cout<<range<<" "<<pow(range,1/3)<<endl;;

	for(int i=0;i<img_ht*img_wd;i++)
	{
		int label_val=labelled_ini[i];
		rgb[i].r=21*label_val%256;
		rgb[i].g=47*label_val%256;
		rgb[i].b=173*label_val%256;
	}
	*/
	//OUTPUT STORAGE
	ofstream ofs;
	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_pixel_val<<"\n";

	for(int j=0; j <img_ht*img_wd;j++)
	{
		ofs<<rgb[j].r<<rgb[j].g<<rgb[j].b;//labelled_ini[j]<<0<<0;//ofs<<Pixel_LAB[j].x<<Pixel_LAB[j].y<<Pixel_LAB[j].z; //write as ascii
	}

	ofs.close();

	return 0;
}